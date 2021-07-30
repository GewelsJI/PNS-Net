# Evaluation Toolbox for Video Polyp Segmentation (VPS)

This code is used for our paper _Progressively Normalized Self-Attention Network for Video Polyp Segmentation_, which is provisionally accepted at MICCAI 2021 conference. Our project page is avaliable at [website](http://dpfan.net/pranet/).

## Features

A simple-to-use toolbox for the evaluation of widely-used metrics (e.g., Dice, IoU, weighted-Fmeasure, Sensitivity, Specificity, Structure measure, Enhanced-alignment measure, and MAE) in medical segmentation, which provide the out-of-the-box functionality.

## Usage

Please set the candidate list of method and dataset in [Line-26](https://github.com/GewelsJI/PNS-Net/blob/7c3996afa3aeacad2316d76b7b4747c8579a9b35/eval/main_VPS.m#L26) and [Line-28](https://github.com/GewelsJI/PNS-Net/blob/7c3996afa3aeacad2316d76b7b4747c8579a9b35/eval/main_VPS.m#L28), respectively. 
You guys first need to install the Matlab first.
To this end， you just run:

```bash
cd eval 
matlab main_VPS.m
```

## Citation

It can only be used for non-comercial purpose. If you like our work, please cite our paper:

    @article{ji2021pnsnet,
        title={PNS-Net: Progressively Normalized Self-Attention Network for Video Polyp Segmentation},
        author={Ji, Ge-Peng and Chou, Yu-Cheng and Fan, Deng-Ping and Chen, Geng and Jha, Debesh and Fu, Huazhu and Shao, Ling},
        journal={MICCAI},
        year={2021}
    }
    
    @article{fan2020pra,
        title={PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
        author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
        journal={MICCAI},
        year={2020}
    }


## 7. Acknowledgements

This code is heavily adopted from the evaluation toolbox from [DAVSOD](https://github.com/DengPingFan/DAVSOD), and many thank to [Deng-Ping Fan](https://scholar.google.com/citations?user=kakwJ5QAAAAJ&hl=en) for his pretty nice work.
