# RAS-pytorch
The pytorch code for our TIP paper: [Reverse Attention Based Residual Network for Salient Object Detection](https://ieeexplore.ieee.org/document/8966594)

---

Notice
---
We use ResNet50 as backbone and add the [IoU loss](https://github.com/NathanUA/BASNet)[1] for better performance in this pytorch implementation. The original caffe version is [here](https://github.com/ShuhanChen/RAS_ECCV18). We provide two versions with different training strategies.<br>
v1: The same training strategy with [CPD](https://github.com/wuzhe71/CPD)[2].<br>
v2: The same training strategy with [F3Net](https://github.com/weijun88/F3Net)[3].<br>

Usage
---
Modify the pathes of datasets, then run:<br>
```
Training: python3 train.py
Testing: python3 test.py
```

Performace
---
The codes are tested on Ubuntu 18.04 environment (Python3.6.9, PyTorch1.5.0, torchvision0.6.0, CUDA10.2, cuDNN7.6.5) with RTX 2080Ti GPU.


Pre-trained model & Pre-computed saliency maps
---
v1: model [Baidu](https://pan.baidu.com/s/1O5QsWWOjhPMGOWIiIwSX3A)(bc3k) [Google](https://drive.google.com/file/d/1KHmKrAG1M_C0mYgSD8pz9fDmBn2LtoMJ/view?usp=sharing); smaps [Baidu](https://pan.baidu.com/s/13I2F0dPU5mPmklcxbex0Lw)(kp6t) [Google](https://drive.google.com/file/d/1lT_BkFMuD8kPVkjQRR7HVBDzQnY3VkfB/view?usp=sharing)<br>
v2: <br>

Citation
---
```
@inproceedings{chen2018eccv, 
  author={Chen, Shuhan and Tan, Xiuli and Wang, Ben and Hu, Xuelong}, 
  booktitle={European Conference on Computer Vision}, 
  title={Reverse Attention for Salient Object Detection}, 
  year={2018}
} 
```
```
@article{chen2020tip, 
  author={Chen, Shuhan and Tan, Xiuli and Wang, Ben and Lu, Huchuan and Hu, Xuelong and Fu, Yun}, 
  journal={IEEE Transactions on Image Processing}, 
  title={Reverse Attention Based Residual Network for Salient Object Detection},
  volume={29},  
  pages={3763-3776},
  year={2020}
} 
```

Acknowledgements
---
This code is built on [CPD](https://github.com/wuzhe71/CPD)[2] and [F3Net](https://github.com/weijun88/F3Net)[3]. We thank the authors for sharing their codes.

Reference
---
[1] Xuebin Qin, Zichen Zhang, Chenyang Huang, Chao Gao, Masood Dehghan, Martin Jagersand. BASNet: Boundary-Aware Salient Object Detection. In CVPR, 2019.<br>
[2] Zhe Wu, Li Su, Qingming Huang. Cascaded Partial Decoder for Fast and Accurate Salient Object Detection. In CVPR, 2019.<br>
[3] Jun Wei, Shuhui Wang, Qingming Huang. F3Net: Fusion, Feedback and Focus for Salient Object Detection. In AAAI, 2020.<br>
