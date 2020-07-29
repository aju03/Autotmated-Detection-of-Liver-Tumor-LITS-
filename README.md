# Autotmated-Detection-of-Liver-Tumor-LITS-
Autotmated Detection of Liver Tumor using deep learning with LITS Dataset.

The proposed approach of deep learning model uses a 2D U-Net architecture. 
The U-Net architecture consists of three layers: the contracting/downsampling, the expanding/upsampling and the bottleneck layer.
The dataset was obtained from the Liver Tumor Segmentation (LiTS)challenge organized by ISBI 2017 and MICCAI 2017. 
It is a collection of 130 and 70 CT scans for training and testing respectively.

The backend is a python code (Python 3.7 with Keras 2.0 framework for deep learning). The model used for training the dataset was a 2D-Unet with 9 layers.
90 scans were used for training the model and 10 scans for validation and 30 for testing purposes as only 130 scans had ground truth. 
The training was performed in 3 batches for 30 epochs on Intel i9-9900K processor along with 32 GB RAM and NVIDIA RTX 2070 Super GPU.
