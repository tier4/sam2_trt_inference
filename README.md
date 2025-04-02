# sam2_ort_cpp
TensorRT inference framework of SAM2(cpp version) with model convertor from onnx to TRT engine.

## requirements
- Docker
- NVIDIA Container Toolkit
- OpenCV
- TensorRT(8.5 or 8.6)

## set-up environments
- set-up docker
```
docker pull nvcr.io/nvidia/tensorrt:24.01-py3
```

- launch above container and inside
```
sudo apt update
sudo apt install libboost-all-dev
sudo apt install libopencv-dev
```

## build and execute

```
cmake -B build
cd build
make
```

### check arguments
```
./MyTRTSAM2App -h
```

### execute with a folder of images
```
./MyTRTSAM2App encoder.onnx decoder.onnx path_to_image_folder path_to_bbox_txt_folder path_to_save_folder --batch_size 1 --decoder_batch_limit 50
```