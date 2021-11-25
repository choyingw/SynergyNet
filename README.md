# SynergyNet
3DV 2021: Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry

Cho-Ying Wu, Qiangeng Xu, Ulrich Neumann, CGIT Lab at University of Souther California

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synergy-between-3dmm-and-3d-landmarks-for/face-alignment-on-aflw)](https://paperswithcode.com/sota/face-alignment-on-aflw?p=synergy-between-3dmm-and-3d-landmarks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synergy-between-3dmm-and-3d-landmarks-for/head-pose-estimation-on-aflw2000)](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000?p=synergy-between-3dmm-and-3d-landmarks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synergy-between-3dmm-and-3d-landmarks-for/face-alignment-on-aflw2000-3d)](https://paperswithcode.com/sota/face-alignment-on-aflw2000-3d?p=synergy-between-3dmm-and-3d-landmarks-for)

[<a href="https://arxiv.org/abs/2110.09772">paper</a>] [<a href="https://choyingw.github.io/works/SynergyNet/index.html">project page</a>]

This paper supersedes the previous version of M3-LRN.

<img src='demo/demo.gif'>

<img src='demo/teaser.png'>

**Advantages:**

&diams; SOTA on all 3D facial alignment, face orientation estimation, and 3D face modeling.<br><br>
&diams; Fast inference with 3000fps on a laptop RTX 2080 Ti.<br><br>
&diams; Simple implementation with only widely used operations.<br><br>


**Evaluation**
(This project is built/tested on Python 3.8 and PyTorch 1.9)

1. Clone

    ```git clone https://github.com/choyingw/SynergyNet```

    ```cd SynergyNet ```

2. Use conda

    ```conda create --name SynergyNet```

    ```conda activate SynergyNet```

3. Install pre-requisite common packages

    ```PyTorch 1.9 (should also be compatiable with 1.0+ versions), Opencv, Scipy, Matplotlib ```

4. Prepare data

Download data [<a href="https://drive.google.com/file/d/1YVBRcXmCeO1t5Bepv67KVr_QKcOur3Yy/view?usp=sharing">here</a>] and
[<a href="https://drive.google.com/file/d/1SQsMhvAmpD1O8Hm0yEGom0C0rXtA0qs8/view?usp=sharing">here</a>]. Extract these data under the repo root.

These data are processed from [<a href="https://github.com/cleardusk/3DDFA">3DDFA</a>] and [<a href="https://github.com/shamangary/FSA-Net">FSA-Net</a>].

Download pretrained weights [<a href="https://drive.google.com/file/d/1BVHbiLTfX6iTeJcNbh-jgHjWDoemfrzG/view?usp=sharing">here</a>]. Put the model under 'pretrained/'


5. Benchmarking

    ```python benchmark.py -w pretrained/best.pth.tar```

Print-out results and visualization under 'results/' (see 'demo/' for some sample references) are shown.

**Training**

1. Download data in last section: 'Evaluation-4 Prepare data'.

2. Download training data from [<a href="https://github.com/cleardusk/3DDFA">3DDFA</a>]: train_aug_120x120.zip and extract the zip file under the root folder (This training size is about 44G).

3. 
    ```bash train_script.sh```

4. Please refer to train_script for hyperparameters, such as learning rate, epochs, or GPU device. 

**TODO**

- [ ] Single-Image inference
- [ ] Add a renderer and 3D face output
- [O] Training script
- [ ] Texture synthesis in the supplementary

We show a comparison with [<a href="https://github.com/YadiraF/DECA">DECA</a>] using the top-3 largest roll angle samples in AFLW2000-3D.

<img src='demo/comparison-deca.png'>

**More Results**

Facial alignemnt on AFLW2000-3D (NME of facial landmarks):

<img src='demo/alignment.png'>

Face orientation estimation on AFLW2000-3D (MAE of Euler angles):

<img src='demo/orientation.png'>

Results on artistic faces: 

<img src='demo/AF-1.png'>

<img src='demo/AF-2.png'>

**Related Project**

[<a href="https://github.com/choyingw/Voice2Mesh">Voice2Mesh</a>] (analysis on relation for voice and 3D face)

**Acknowledgement**

The project is developed on [<a href="https://github.com/cleardusk/3DDFA">3DDFA</a>] and [<a href="https://github.com/shamangary/FSA-Net">FSA-Net</a>]. Thank them for their wonderful work.
