
apt update && apt install wget vim tmux

mkdir ./checkpoints
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth

mim download mmdet --config rtmdet_x_8xb32-300e_coco --dest ./checkpoints
mim download mmdet --config rtmdet_l_swin_b_p6_4xb16-100e_coco --dest ./checkpoints
rtmdet_l_swin_b_p6_4xb16-100e_coco
pip install future tensorboard 
pip install setuptools==59.5.0 fairscale supervision
python prepare.py
python tools/train.py configs.py
python infer.py