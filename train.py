# Some basic setup:
# Setup detectron2 logger
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
import copy
import detectron2.data.transforms as T
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
import logging
from detectron2.structures import BoxMode
import datetime as dt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from tqdm import trange, tqdm
import torch
import tracemalloc
import pycocotools

setup_logger()

# import some common libraries

# import some common detectron2 utilities


def search(dirname, target="mask"):
    filenames = os.listdir(dirname)
    tmp = []
    for filename in filenames:
        if os.path.isdir(dirname + "/" + filename):
            continue
        elif target in filename:
            tmp.append(dirname + "/" + filename)
    return tmp


def get_balloon_dicts(img_dir):
    dataset_dicts = []
    leng = -1
    logger = logging.getLogger(__name__)
    logger.info("start {} data load".format(img_dir))
    print()
    img_list = search(img_dir)
    # json_file = os.path.join(img_dir, "annotation.json")
    # f = open(json_file, "r")
    # while True:
    for idx, file in enumerate(tqdm(img_list)):
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        record = {}

        record["file_name"] = file.replace("_mask.png", ".png")
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        objs = []
        py, px = np.where(mask != 0)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        ind = -1
        maxarea = 0
        for cont in range(len(contours)):
            area = cv2.contourArea(contours[cont])
            if area > maxarea:
                ind = cont
                maxarea = area
        px = []
        py = []
        for cont in contours[ind]:
            py.append(int(cont[0][1]))
            px.append(int(cont[0][0]))
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        # mask = pycocotools.mask.encode(mask.astype(np.uint8, order="F"))
        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": 0,  # my dataset starts with 1 as a class label so I do "label-1" to let category id start from 0
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    logger.info("end {} data load".format(img_dir))
    # print(dataset_dicts[-1]["annotations"])
    return dataset_dicts


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        # T.Resize((800, 600)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(
        annos, image.shape[:2], mask_format="bitmask"
    )
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=custom_mapper)
        return build_detection_train_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(f"{cfg.OUTPUT_DIR}/coco_eval", exist_ok=True)
            output_folder = f"{cfg.OUTPUT_DIR}/coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)


for d in ["train", "val"]:
    DatasetCatalog.register(
        f"milkthistle_{i}_" + d,
        lambda d=d: get_balloon_dicts(f"/home/oogis/milkthistle/stride{i}/{d}"),
    )
    MetadataCatalog.get(f"milkthistle_{i}_" + d).set(thing_classes=["Elaiosome"])
balloon_metadata = MetadataCatalog.get(f"milkthistle_{i}_train")

cfg = get_cfg()
logdir = f"LogDIR"
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.DATASETS.TRAIN = (f"milkthistle_{i}_train",)
cfg.DATASETS.TEST = (f"milkthistle_{i}_val",)

cfg.DATALOADER.NUM_WORKERS = 20
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)  # Let training initialize from model zoo
# This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
# 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = []  # do not decay learning rate
# The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 640
# only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 10
cfg.SOLVER.CHECKPOINT_PERIOD = 10
cfg.VIS_PERIOD = 1
cfg.SOLVER.REFERENCE_WORLD_SIZE = 1
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = logdir
# cfg.INPUT.MASK_FORMAT = "bitmask"  # for rle
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CustomTrainer(cfg)
# evaluator = COCOEvaluator("milkthistle_val", output_dir=logdir)
# trainer.test(evaluators=evaluator)
trainer.resume_or_load(resume=True)
trainer.train()
