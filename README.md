## '21 NIA 욕창과제 모델 최종산출물 코드

### Base Repository: https://github.com/ultralytics/yolov5
Author: Ultralytics
GPL-3.0 license

### Modified by ☘ Molpaxbio Co., Ltd.
Author: jaesik won <jaesik.won@molpax.com>  
Created: 10/12/2021  
Version: 0.5

### 파일 구조 및 설명
```Bash
.
├── data      # YOLOv5 original data utils
│   └── hyps  # hyperparameters
├── models    # YOLOv5 original model utils
│   └── hub   # anchor config
├── nia_utils # utils for nia
├── utils     # YOLOv5 original other utils
└── runs                 # runs
    ├── infer
    │   ├── s3           # infer from s3 trained weight model, s3 valset
    │   └── zip          # infer from zip trained weight model, s3 valset
    ├── test
    │   └── s3           # test from s3 trained weight model, s3 testset
    │   └── zip          # test from zip trained weight model, s3 testset
    └── train
        ├── s3           # train from s3 trainset/val from s3 valset
        │   └── weights  
        └── zip          # train from zip trainset/val from zip valset
            └── weights  
```
  
  
### 사용법
#### 1. 설치 방법
```Bash
git clone https://github.com/molpaxbio-git/21PU_NIA-Final-Product.git
cd 21PU_NIA-Final-Product

# 환경 변수 설정
export AWS_ACCESS_KEY_ID=[사용할 access key id]
export AWS_SECRET_ACCESS_KEY=[사용할 secret access key]
export AWS_BUCKET_NAME=[사용할 bucket 이름]

# 프레임워크 설치
sudo apt-get update
pip install --upgrade pip
pip install --no-cache -r requirements.txt
# optional
# pip install --no-cache torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip install --no-cache -U numpy
```

#### 2. 사용 방법 (소스 코드)
학습 training
| hyp 예시   | value |
|------------|-------|
| image size | 224   |
| batch size | 64    |
| max epoch  | 200   |
```Bash
# cpu
python training.py --img 224 --batch 64 --epochs 200 --device cpu --cfg ./models/sP5.yaml --weights yolov5s.pt --name [생성할 폴더 이름] --project "./runs/train"
# single gpu
python training.py --img 224 --batch 64 --epochs 200 --device 0 --cfg ./models/sP5.yaml --weights yolov5s.pt --name [생성할 폴더 이름] --project "./runs/train"
```
검증 validation
| hyp 예시   | value |
|------------|-------|
| image size | 224   |
| batch size | 64    |
```Bash
# cpu
python infer.py --img 224 --batch 64 --data ./nia_utils/data.yaml --device cpu --weights ./runs/train/[학습 폴더 이름]/weights/best.pt --name [생성할 폴더 이름] --verbose --project "./runs/infer"
# single gpu
python infer.py --img 224 --batch 64 --data ./nia_utils/data.yaml --device 0 --weights ./runs/train/[학습 폴더 이름]/weights/best.pt --name [생성할 폴더 이름] --verbose --project "./runs/infer"
```
테스트 test
| hyp 예시   | value |
|------------|-------|
| image size | 224   |
```Bash
# cpu
python test.py --img 224 --device cpu --weights ./runs/train/[학습 폴더 이름]/weights/best.pt --name [생성할 폴더 이름] --project "./runs/test"
# single gpu
python test.py --img 224 --device 0 --weights ./runs/train/[학습 폴더 이름]/weights/best.pt --name [생성할 폴더 이름] --project "./runs/test"
```

#### 3. 사용 방법 (Bash Scripts)
학습 training
| hyp 예시   | value |
|------------|-------|
| image size | 224   |
| batch size | 64    |
| max epoch  | 200   |
```Bash
$ ./training.sh [생성할 폴더 이름] [cpu 또는 0]
$ image size: 224
$ batch size: 64
$ max epochs: 200
```
검증 validation
| hyp 예시   | value |
|------------|-------|
| image size | 224   |
| batch size | 64    |
```Bash
$ ./infer.sh [학습 폴더 이름] [cpu 또는 0]
$ image size: 224
$ batch size: 64
```
테스트 test
| hyp 예시   | value |
|------------|-------|
| image size | 224   |
```Bash
$ ./test.sh [학습 폴더 이름] [cpu 또는 0]
$ image size: 224
```
