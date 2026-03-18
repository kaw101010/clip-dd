Run command below for inference on images listed in the txt file. Text file provided.
Place all images in the same directory and run inference.
Download model from https://drive.google.com/drive/u/0/folders/1XnOA9NLdUPagNj7lvscdpwZMyoMXZ8tJ
and place it in weights directory


```python test.py --image_list_path publictest_images.txt --weights_path weights/clip_large.pth --output_file submission_publictest.txt --batch_size 64 --num_workers 8```


Note: Adapted from https://github.com/SCLBD/DeepfakeBench
