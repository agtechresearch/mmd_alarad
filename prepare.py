import os.path as osp
import mmcv
from mmengine.fileio import dump, load
from util import DATA_DIR
# from mmengine.utils import track_iter_progress

def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    coco_format_json = dict(
        images=data_infos["images"],
        annotations=data_infos["annotations"],
        categories=data_infos["categories"])
    dump(coco_format_json, out_file)

if __name__ == '__main__':
    root = DATA_DIR
    convert_balloon_to_coco(ann_file=f'{root}/annotations/train_STR.json',
                            out_file=f'{root}/train.json',
                            image_prefix=f'{root}/images/train')
    convert_balloon_to_coco(ann_file=f'{root}/annotations/val_STR.json',
                            out_file=f'{root}/val.json',
                            image_prefix=f'{root}/images/val')
    convert_balloon_to_coco(ann_file=f'{root}/annotations/test_STR.json',
                            out_file=f'{root}/test.json',
                            image_prefix=f'{root}/images/test')
