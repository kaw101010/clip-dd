Run command below for inference on images listed in the txt file. Text file provided.
Place all images in the same directory and run inference.
```python test.py --image_list_path publictest_images.txt --weights_path ../../DF40/DeepfakeBench_DF40/training/clip_large.pth --output_file submission_publictest.txt --batch_size 64 --num_workers 8```


Note: Adapted from https://github.com/SCLBD/DeepfakeBench
