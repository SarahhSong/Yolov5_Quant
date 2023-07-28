# Q_YOLOv5s
## Install the environment
```
conda create -n Qyolov5s python=3.8
conda activate Qyolov5s
pip3 install -r requirements.txt
```
## Data organization
   ```
   ${Q_YOLOv5s}
    -- data
        -- coco
            |-- annotations
                    |--instances_train2017.json
                    |--instances_val2017.json
            |-- images
                    |--train2017
                    |--val2017
   ```
## Set environmental variable
```
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_CACHE_PATH=/data/.cuda_cache
```
## Quantization
### Step1. Train a fp32 model
```
python train.py --mode normal
```
### Step2. Finetune fp32 model with quantization aware training(QAT)
```
python train.py --checkpoint /path/to/Qyolov5s.normal/checkpoint.pkl --mode qat
```
### Step3. Calibration


### Step4. Test QAT model


### Step5. Inference and dump

