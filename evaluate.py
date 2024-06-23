from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
import datetime as dt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
import random
from detectron2.engine import DefaultTrainer
import random
import cv2
import json
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import matplotlib.pyplot as plt
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode
import datetime as dt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from functools import reduce
import colorsys
from tqdm import tqdm
import time 

def search(dirname, target="mask"):
    filenames = os.listdir(dirname)
    tmp = []
    for filename in filenames:
        if os.path.isdir(dirname + "/" + filename):
            continue
        elif target not in filename:
            tmp.append(dirname + "/" + filename)
    return tmp


def get_balloon_dicts(img_dir):
    dataset_dicts = []
    for json_file in search(img_dir):
        # json_file = os.path.join(img_dir, "via_region_data.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        for idx, v in enumerate(imgs_anns.values()):
            record = {}

            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            annos = v["region"]
            objs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def find_last(model_dir, key):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        The path of the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model_dir))[1]
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno

        raise FileNotFoundError(
            errno.ENOENT, "Could not find model directory under {}".format(model_dir)
        )
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("model"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        import errno

        raise FileNotFoundError(
            errno.ENOENT, "Could not find weight files in {}".format(dir_name)
        )
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return checkpoint


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = list(
        map(
            lambda color: (
                int(color[0] * 255),
                int(color[1] * 255),
                int(color[2] * 255),
            ),
            colors,
        )
    )
    random.shuffle(colors)
    return colors


class CustomVisualizer(Visualizer):
    def _jitter(self, color):
        return color


for d in ["train", "val"]:
    DatasetCatalog.register("milkthistle_" + d, lambda d=d: [])
    MetadataCatalog.get("milkthistle_" + d).set(thing_classes=[""])
colors = random_colors(1)
for folder in ["val", "train", "test"][:]:
    # train =
    files = []
    print(f"-------------------------------\n{folder}\n----------------------------------------")
    files = search(f"Path to Dataset")
    MetadataCatalog.get("milkthistle_train").set(thing_colors=colors)
    MetadataCatalog.get("milkthistle_val").set(thing_colors=colors)
    balloon_metadata = MetadataCatalog.get("milkthistle_train")

    cfg = get_cfg()
    # logdir = find_last(f"./output/", "milkthistle").rsplit("/", 1)[0]
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("milkthistle_train",)
    cfg.DATASETS.TEST = ("milkthistle_val",)
    cfg.DATALOADER.NUM_WORKERS = 10
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    # )  # Let training initialize from model zoo
    # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 10
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = f"Path to Model"
    # print(MetadataCatalog.get("milkthistle_train").get("thing_colors"))
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(
        f"Path to Model"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    # evaluator = COCOEvaluator("milkthistle_val", output_dir=logdir)
    # val_loader = build_detection_test_loader(cfg, "milkthistle_val")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`
    # dataset_dicts = get_balloon_dicts("/home/oogis/seeds/stride/val")
    # print(search(f"/home/oogis/segmentation/corop/barely/image", target='jpg'))
    print(len(files))
    pbar = tqdm(files[:])
    start = time.time()
    for d in pbar:
        try:
            pbar.set_description(d.replace("_mask", "").split("/")[-1])
            im = cv2.imread(d.replace("_mask", ""))
            v = CustomVisualizer(
                im[:, :, ::-1],
                metadata=balloon_metadata,
                # scale=0.5,
                # remove the colors of unsegmented pixels. This option is only available for segmentation models
                instance_mode=ColorMode.SEGMENTATION,
            )
            # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            outputs = predictor(im)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            filename = d.split("/")[-1].replace("_mask",'').replace(".jpg",'').replace(".png",'')
            pred_masks = (
                outputs["instances"].pred_masks.cpu().numpy().astype(np.uint8) * 255
            )
            # cv2.imwrite(
            #     f"{mo}/{folder}/{filename}.png",
            #     out.get_image()[:, :, ::-1],
            # # )
            # for i in range(len(outputs["instances"].pred_masks)):
            #     cv2.imwrite(
            #         f"{mo}/{folder}/{filename}_mask_{i}.png",
            #         pred_masks[i, :, :],
            #     )
        except KeyboardInterrupt:
            break

    end = time.time()
    sec = (end - start)
    result = dt.timedelta(seconds=sec)
    print(result)
