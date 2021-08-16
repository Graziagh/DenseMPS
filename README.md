# DenseMPS
Dense Matrix Product State for Pathological image classification
# Introduction
dataset.py is used to load dataset and preprocess the data 

mps.py is a single-layer tensor network that compresses the input data

densemps.py is a network model based on mps to calculate the classification results of the input data

train.py shows the process of training images

# Environment
Pytorch 3.8

# Run
python main.py --num_epochs 100 --batch_size 512 --data_path XXX 


XXX denotes the path of the data set

# Data
BreaKHis Dataset:  https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/

IDC Dataset:  https://www.kaggle.com/paultimothymooney/breast-histopathology-images

PCam Dataset:  https://github.com/basveeling/pcam

# Acknowledgements
+ Raghavendra Selvan, Erik B Dam, _Tensor Networks for Medical Image Classification_
