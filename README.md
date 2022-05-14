# Point MLP-Former: Combining Local and Global Receptive Fields in Point Cloud Classification


Full code will be released soon.


# Getting Started
## 1. Requirements
```
PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 11.0;
GCC >= 7.5;
torchvision;
```
## 2. Datasets
<!-- ### ModelNet40 Dataset: 

```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself. -->

### ScanObjectNN Dataset:
```
│data/
│  main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

# Usage
```
bash train.sh
```

# Citation