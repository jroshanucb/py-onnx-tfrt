# py-onnx-tfrt

## Git pull this repo
https://github.com/dusty-nv/jetson-inference.git

## Run the docker container
jetson-inference/docker/run.sh
change directory to jetson-inference/python/training/detection/ssd

## Download the images
$ python3 open_images_downloader.py --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit
...
2020-07-09 16:20:42 - Starting to download 6360 images.
2020-07-09 16:20:42 - Downloaded 100 images.
2020-07-09 16:20:42 - Downloaded 200 images.
2020-07-09 16:20:42 - Downloaded 300 images.
2020-07-09 16:20:42 - Downloaded 400 images.
2020-07-09 16:20:42 - Downloaded 500 images.
2020-07-09 16:20:46 - Downloaded 600 images.
...
2020-07-09 16:32:12 - Task Done.

## Download fewer images
$ python3 open_images_downloader.py --max-images=2500 --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit

## Train the data
python3 train_ssd.py --data=data/fruit --model-dir=models/fruit --batch-size=4 --epochs=30

## Convert the pth files to onnx
python3 onnx_export.py --model-dir=models/fruit

## Get the model save files from models\...
Get the model onnx files out from the container (by mounting the local file system)
Or docker commit the image and run it again in interactive mode and copy the onnx file to external filesystem

## Get the tensorflow-trt container
docker pull rdejana/tf-trt-demo
copy the onnx file into this container by mounting the local filesystem

## Run ipynb from the tf-trt container
Run the cell with this code

_print('Converting to TF-TRT FP32...')
max = 3000000000
_conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP32,
                                                               max_workspace_size_bytes=max)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='resnet50_saved_model',
                                    conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir='resnet50_saved_model_TFTRT_FP32')
print('Done Converting to TF-TRT FP32')

## Load the onnx model into tf-trt
Run these functions to train and predict the same images downloaded at the beginning of this exercise

_saved_model_loaded = tf.saved_model.load('resnet50_saved_model_TFTRT_FP32', tags=[tag_constants.SERVING])
predict_tftrt(saved_model_loaded)_
