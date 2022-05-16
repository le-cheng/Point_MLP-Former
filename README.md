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
Start training from scratch:
```
bash train.sh
```
Test:
```
bash test.sh
```
Note: Due to the randomness of the initial point of the farthest point sampling(FPS), the results of the test and training cannot be consistent. We use the training model to test for many times, and we can get different results, or even exceed the training results (85.32 for training, 85.46 for multiple tests). We use the training model to test for many times, and we can get different results, or even exceed the training results (85.32 for training, 85.46 for multiple tests). Training and test logs are saved in `/logs`. If you want to get consistent test results, you can change FPs to:
```
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)*0
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids
```
Fixed initial point is 0, can get 85.01 for testing.

If you have a better solution, I'm glad you can let me know.

# Citation