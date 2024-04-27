# VSCode의 tensorboard extension으로 training log 확인!
# 경로는 work_dirs/config_alarad/*에서 보고싶은 폴더로 경로 기입

from util import DATA_DIR, CONFIG_FILENAME


from mmdet.apis import DetInferencer
import glob
import os
import random
# import supervision as sv
import numpy as np

output_dir = "./output"
# Choose to use a config
config = f"./work_dirs/config_alarad/{CONFIG_FILENAME}"
# Setup a checkpoint file to load
checkpoint = glob.glob(f"./work_dirs/config_alarad/best_coco*.pth")[0]
# checkpoint = f"./work_dirs/{CONFIG_FILENAME[:-3]}/best_coco_bbox_mAP_epoch_21.pth"

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

images = os.listdir(f"{DATA_DIR}images/test/")
image = f"{DATA_DIR}images/test/" + random.choice(images)
while ".txt" in image:
    image = f"{DATA_DIR}images/test/" + random.choice(images)
result = inferencer(image, out_dir=output_dir)

# Show the structure of result dict
from rich.pretty import pprint
pprint(result, max_length=4)



# CONFIDENCE_THRESHOLD = 0.35
# NMS_IOU_THRESHOLD = 0.7

# def callback(image: np.ndarray) -> sv.Detections:
#     result = inferencer(image)
#     detections = sv.Detections.from_mmdetection(result)
#     return detections[detections.confidence > CONFIDENCE_THRESHOLD].with_nms(threshold=NMS_IOU_THRESHOLD)

# ds = sv.DetectionDataset.from_coco(
#     images_directory_path=f"{DATA_DIR}images/test",
#     annotations_path=f"{DATA_DIR}test.json",
# )

# confusion_matrix = sv.ConfusionMatrix.benchmark(
#     dataset = ds,
#     callback = callback
# )

# _ = confusion_matrix.plot(save_plot=output_dir)


# mean_average_precision = sv.MeanAveragePrecision.benchmark(
#     dataset = ds,
#     callback = callback
# )

# print('mAP:', mean_average_precision.map50_95)

# per_class_map = mean_average_precision.per_class_ap50_95.mean(axis=1)
# for class_name, value in zip(ds.classes, per_class_map):
#     print(f"{class_name}: {value:.2f}")