## HRDecoder: High-Resolution Decoder Network for Fundus Image Lesion Segmentation

### Environment
Our code is based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation).

In this work, we used:
- python=3.8.5
- pytorch=2.0.1
- mmsegmentation=0.16.0
- mmcv=1.7.1

The environment can be installed with:
```
conda create -n hrdecoder python=3.8.5
conda activate hrdecoder

pip install -r requirements.txt
chmod u+x tools/*
```

### Datasets
We carried out experiments on two public datasets, i.e. IDRiD and DDR. For IDRiD, please download the segmentation part from [here](https://ieee-dataport.s3.amazonaws.com/open/3754/A.%20Segmentation.zip?response-content-disposition=attachment%3B%20filename%3D%22A.%20Segmentation.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240315%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240315T053048Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=de00f49e9a770569b25982825c74f0b74a69e40fb965de881a8efca993f5b71f). For DDR dataset, please download the segmentation part from [here](https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC).

Please first change the original structure of datasets and organize like this:
```
IDRiD(segmentation part)
|——images
|  |——train
|  |  |——IDRiD_01.jpg
|  |  |——...
|  |  |——IDRiD_54.jpg
|  |——test
|——labels
|  |——train
|  |  |——EX
|  |  |	 |——IDRiD_01_EX.tif
|  |  |	 |——...
|  |  |  |——IDRiD_54_EX.tif
|  |  |——HE
|  |  |——SE
|  |  |——MA
|  |——test
|  |  |——EX
|  |  |——HE
|  |  |——SE
|  |  |——MA


DDR(segmentation part)
|——images
|  |——test
|  |  |——007-1789-100.jpg
|  |  |——...
|  |  |——20170627170651362.jpg
|  |——val
|  |——train
|——labels
|  |——test
|  |  |——EX
|  |  |  |——007-1789-100.tif
|  |  |  |——...
|  |  |  |——20170627170651362.tif
|  |  |——HE
|  |  |——MA
|  |  |——SE
|  |——val
|  |  |——EX
|  |  |——HE
|  |  |——MA
|  |  |——SE
|  |——train
|  |  |——EX
|  |  |——HE
|  |  |——MA
|  |  |——SE
```

Please note that the original GT with `.tif` format can not be directly used for training, so you can use the following command to convert the labels to `.png` format:
```
python tools/convert_dataset/idrid.py
python tools/convert_dataset/ddr.py
```
Then the structure of dataset will be converted to:
```
IDRiD(segmentation part)
|——images
|  |——train
|  |  |——IDRiD_01.jpg
|  |  |——...
|  |  |——IDRiD_54.jpg
|  |——test
|——labels
|  |——train
|  |  |——IDRiD_01.png
|  |  |——...
|  |  |——IDRiD_54.png
|  |——test


DDR(segmentation part)
|——images
|  |——test
|  |  |——007-1789-100.jpg
|  |  |——...
|  |  |——20170627170651362.jpg
|  |——val
|  |——train
|——labels
|  |——test
|  |  |——007-1789-100.png
|  |  |——...
|  |  |——20170627170651362.png
|  |——val
|  |——train
```

Finally the structure of the project should be like this:
```
HRDecoder
|——configs
|——mmseg
|——tools
|——data
|  |——IDRiD(segmentation part)
|  |  |——images
|  |  |  |——train
|  |  |  |——test
|  |  |——labels
|  |  |  |——train
|  |  |  |——test
|  |——DDR(segmentation part)
|  |  |——images
|  |  |  |——train
|  |  |  |——val
|  |  |  |——test
|  |  |——labels
|  |  |  |——train
|  |  |  |——val
|  |  |  |——test
```

### Training and Testing

To train or test a model using MMsegmentation framework, you can use commands like this:
```
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=18940 tools/dist_train.sh configs/lesion/hrdecoder_fcn_hr48_idrid_2880x1920-slide.py 4
or
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=18940 tools/dist_train.sh configs/lesion/efficient-hrdecoder_fcn_hr48_idrid_2880x1920-slide.py 4

# test
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=17940 tools/dist_test.sh configs/lesion/hrdecoder_fcn_hr48_idrid_2880x1920-slide.py work_dirs/hrdecoder_fcn_hr48_idrid_2880x1920-slide/latest.pth 4 --eval mIoU
or
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=17940 tools/dist_test.sh configs/lesion/efficient-hrdecoder_fcn_hr48_idrid_2880x1920-slide.py work_dirs/efficient-hrdecoder_fcn_hr48_idrid_2880x1920-slide/latest.pth 4 --eval mIoU
```

The trained model will be stored to `work_dirs/`.

In this project, we supply two version of implemention: HRDecoder and Efficient-HRDecoder.

HRDecoder is the version utilized for presenting the results in our paper. While Efficient-HRDecoder is an improved version of HRDecoder to further reduce computational overhead and memory usage. We simply compress the dimension of the extracted features using a 1x1 convolutional layer right after the backbone. This helps save a lot of memory. The compression operation can be replaced with various multi-scale feature fusion methods, such as FPN, CATNet, APPNet, CARAFE++ and so on. We do not focus on this part but on the later HR representation learning module and HR fusion module, so we simply use upsampling and concat in HRDecoder, or use a 1x1 conv in Efficient-HRDecoder. 

### Models and Evaluation

We evaluate our method out IDRiD and DDR test set.

| Dataset  | mIoU | mAUPR | download                                                                                                                                                       |
| ------- | :----: | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IDRiD |  52.20 | 71.34 | [config](https://drive.google.com/file/d/1grh80iE_zHQGo9JVEJIFhVKpF2C9N8ax/view?usp=drive_link); [model](https://drive.google.com/file/d/1tRBNqpbFwnNBGQBMw8UYL8sjJzWlI3O9/view?usp=drive_link) |
| DDR |  32.24 | 49.24 | [config](https://drive.google.com/file/d/1fC8hB4CkUeaXEiY2KtbNJHwUT5HRQknP/view?usp=drive_link); [model](https://drive.google.com/file/d/1CBTHOtxToHQ13H4SZIUBZTwX0zwmIduh/view?usp=drive_link) |

For calculating FLOPs and FPs, you can use the following commands:
```
# calculate FLOPs
CUDA_VISIBLE_DEVICES=0 python tools/get_flops.py configs/lesion/hrdecoder_fcn_hr48_ddr_2048.py

# Calculate FPs
CUDA_VISIBLE_DEVICES=0 python tools/get_fps.py configs/lesion/hrdecoder_fcn_hr48_ddr_2048.py
```
Typically, we calculate FPs on DDR dataset.

### Acknowledgements
Last, we thank these authors for sharing their ideas and source code:
- [MMsegmentation](https://github.com/open-mmlab/mmsegmentation)
- [HRNet](https://github.com/HRNet)
- [HRDA](https://github.com/lhoyer/HRDA)
- [CARAFE++](https://github.com/myownskyW7/CARAFE)
- [CATNet](https://github.com/yeliudev/CATNet)
- [SenFormer](https://github.com/WalBouss/SenFormer)
- [M2MRF](https://github.com/haotianll/M2MRF)

This manuscript was supported in part by the National Key Research and Development Program of China under Grant 2021YFF1201202, the Natural Science Foundation of Hunan Province under Grant 2024JJ5444 and 2023JJ30699, the Key Research and Development Program of Hunan Province under Grant 2023SK2029, the Changsha Municipal Natural Science Foundation under Grant kq2402227, and the Research Council of Finland (formerly Academy of Finland for Academy Research Fellow project) under Grant 355095. The authors wish to acknowledge High Performance Computing Center of Central South University for computational resources.
