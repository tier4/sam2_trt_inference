# sam2_ort_cpp
TensorRT inference framework of SAM2(cpp version) with model convertor from onnx to TRT engine.

## set-up environment
### onnxruntime
1. download packages from https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz
2. install
```
tar -xzf onnxruntime-linux-x64-[version].tgz
sudo cp -r onnxruntime-linux-x64-[version]/include /usr/local/include/onnxruntime
sudo cp -r onnxruntime-linux-x64-[version]/lib /usr/local/lib/onnxruntime
```

## build and execute

```
cmake -B build
cd build
make
```

### check arguments
```
./MyONNXGPUApp -h
```

### execute with a folder of images
```
./MyONNXGPUApp encoder.onnx decoder.onnx path_to_image_folder path_to_bboxtxt_folder path_to_save_folder --batch_size 2 --decoder_batch_limit 50
```