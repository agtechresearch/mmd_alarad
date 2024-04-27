# RTMDET model

Read this whole content before start!

---

# 1. Installation

## 1-1. Prepare docker container
Using dockerfile on public repository of MMDetection

```
docker run -it --shm-size 24G --gpus all --name mmd-alarad -v /home/cv_task/ALARAD_Strawberry_Dataset_final:/mmdetection/data  mmdetect 
```

- shm-size: share memory must be increased to handle big model
- gpus: use all gpu
- v: volumn mount links between directory of host and directory of container
  - By this option, you do not need to copy whole data in the container.

## 1-2. At inside docker container
```
apt update && apt install wget vim tmux
pip install future tensorboard 
pip install setuptools==59.5.0 fairscale supervision

mkdir ./checkpoints
```

Following commend is for rtmdet model
```
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth
wget -P ./checkpoints https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth

mim download mmdet --config rtmdet_x_8xb32-300e_coco --dest ./checkpoints
mim download mmdet --config rtmdet_l_swin_b_p6_4xb16-100e_coco --dest ./checkpoints
```

+) If you want to use another models, check `./config_list` file.
I do not recommend the configs not in `./config_list` file but in `./configs` folder. It's really hard to run :(


# 2. Train
1. Create 'COCO' format dataset (refer: `python prepare.py`)
2. Copy the content of config file in `./checkpoints` folder into `./configs.py`
3. Change hyperparameters in `./configs.py`
  - Particulary, be careful following variables setting! You have to check setting correctly whole `./configs.py` line by line...!
    ```py
    max_epochs = 100
    train_batch_size_per_gpu = 4
    num_classes = 7
    data_root = '/mmdetection/data/ALARAD_Strawberry_1060_c1/'
    val_ann_file = '/mmdetection/data/ALARAD_Strawberry_1060_c1/val.json'
    test_ann_file = '/mmdetection/data/ALARAD_Strawberry_1060_c1/test.json'
    load_from = './checkpoints/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth'
    # stage2_num_epochs = 20
    base_lr = 0.004
    work_dir = './work_dirs/config_alarad'
    train_batch_size=4
    classes = (
        'Bud',
        'Flower',
        'Receptacle',
        'Early_fruit',
        'White_fruit',
        '50%_maturity',
        '80%_maturity',
    )
    ```
4. Run on tmux!
    ```
    tmux
    python tools/train.py configs.py
    ```

# 3. Train Result
1. Install vscode extension 'Tensorboard'
2. `Ctrl/Cmd + shift + P` and choose 'Python: Launch TensorBoard'
3. Set directory among `./work_dirs` folder


# 4. Infer
Run following command and check `./output/vis/`
```
python infer.py
```
