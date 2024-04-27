
selected_type = "tiny"
root_dir = "/mmdetection"

model_types = {
    "x":{
        "config": f'{root_dir}/checkpoints/rtmdet_x_8xb32-300e_coco.py',
        "weight": f'{root_dir}/checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
    },
    "l":{
        "config": f'{root_dir}/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py',
        "weight": f'{root_dir}/checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
    },
    "tiny":{
        "config": f'{root_dir}/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py',
        "weight": f'{root_dir}/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    }
}

CONFIG_FILENAME = "configs.py"
DATA_DIR = '/mmdetection/data/ALARAD_Strawberry_1060_c1/' 
SELECTED_CONFIG = model_types[selected_type]["config"]
SELECTED_WEIGHT = model_types[selected_type]["weight"]