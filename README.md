# Image Detection

## Model Training

1. Download the Crello dataset from this link: https://storage.cloud.google.com/ailab-public/canvas-vae/crello-dataset-v2.zip
2. Extract the contents of the zip file
3. Copy-paste the dataset_creation.py file inside the extracted directory
4. Modify inside the dataset_creation.py script, the destination path where you want your dataset to be created 
5. python dataset_creation.py
6. git clone https://github.com/meituan/YOLOv6.git
7. cd. YOLOv6
8. pip install -r requirements.txt
9. mkdir weights
10. Since we're going to fine-tune the small version of YOLOv6 model, we'll have to download the pre-trained version of the model first. Download the pre-trained checkpoint from here: https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt
11. Move this downloaded checkpoint inside YOLOv6/weights
12. Modify the data/dataset.yaml file
	a. Change the path to the images train directory. For e.g.: ./dataset/images/train
	b. Change the path to the images val directory. For e.g.: ./dataset/images/val
	c. Change the path to the images test directory. For e.g.: ./dataset/images/test
	d. Change the number of classes "nc". For e.g. if we want to detect "imageElement", "svgElement", and "coloredBackground", set "nc" equal to 3.
	e. Change the names of the classes as per your convenience. For e.g. ['img', 'svg', 'bg']
13python tools/train.py --batch 32 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0

## Model Inference 

Command: python tools/infer.py --weights ./runs/train/exp1/weights/best_ckpt.pt --source ./sample_images/ --device 0 --yaml ./data/dataset.yaml --conf-thres 0.75 --save-txt
