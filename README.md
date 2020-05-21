# RAS-pytorch
The pytorch code for our TIP paper: Reverse Attention Based Residual Network for Salient Object Detection. [pdf](https://ieeexplore.ieee.org/document/8966594)

---

Notice
---
We use ResNet50 as backbone and add the [IoU loss](https://github.com/NathanUA/BASNet)[1] for better performance in this pytorch implemention. The original caffe version is [here](https://github.com/ShuhanChen/RAS_ECCV18). We provide two versions with different training strategies.<br>
v1: The same training strategy with [CPD](https://github.com/wuzhe71/CPD)[2].<br>
v2: The same training strategy with [F3Net](https://github.com/weijun88/F3Net)[3].<br>


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

Reference
---
[1] Xuebin Qin, Zichen Zhang, Chenyang Huang, Chao Gao, Masood Dehghan, Martin Jagersand. BASNet: Boundary-Aware Salient Object Detection. In CVPR, 2019.<br>
[2] Zhe Wu, Li Su, Qingming Huang. Cascaded Partial Decoder for Fast and Accurate Salient Object Detection. In CVPR, 2019.<br>
[3] Jun Wei, Shuhui Wang, Qingming Huang. F3Net: Fusion, Feedback and Focus for Salient Object Detection. In AAAI, 2020.<br>
