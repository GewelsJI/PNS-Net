# Evaluation Toolbox for Video Polyp Segmentation (VPS)

This code is used for our paper _Progressively Normalized Self-Attention Network for Video Polyp Segmentation_, 
which is provisionally accepted at MICCAI 2021 conference. 
Our project page is at [website](http://dpfan.net/pranet/).

# Usage

Please set the candidate method and dataset name in [Line-27]() and [Line-29](), respectively. 
You guys first need to install the Matlab first.
To this end， you just run 

```bash
cd eval 
matlab main_VPS.m
```

# Citation

It can only be used for non-comercial purpose. If you like our work, please cite our paper:

    @article{ji2021pnsnet,
    title={PNS-Net: Progressively Normalized Self-Attention Network for Video Polyp Segmentation},
    author={Ji, Ge-Peng and Chou, Yu-Cheng and Fan, Deng-Ping and Chen, Geng and Jha, Debesh and Fu, Huazhu and Shao, Ling},
    journal={MICCAI},
    year={2021}
    }

## 7. Acknowledgements

This code is adopted from [DAVSOD](https://github.com/DengPingFan/DAVSOD), and many thank to [Deng-Ping Fan]().