# LEAST: "Local" text-conditioned image style transfer


*Accepted to AI for Content Creation (AI4CC) Workshop at CVPR 2024*

*[Silky Singh](https://silky1708.github.io/), [Surgan Jandial](https://surgan12.github.io/), [Simra Shahid](https://scholar.google.com/citations?user=RXM-KSQAAAAJ&hl=en), [Abhinav Java](https://java-abhinav07.github.io/).*  
Media and Data Science Research (MDSR), Adobe


Project Page: [arXiv]()


![local style transfer teaser](assets/main_qual_results.png)


## Installation


Create a conda environment using the provided `environment.yml` file:  

```
conda env create -f environment.yml  
conda activate least
```



## Getting Started



Download [SAM's `vit-h` checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place it here: `segment-anything/checkpoints/sam_vit_h_4b8939.pth` exactly following the name convention.




## Acknowledgments


This repository is heavily based on [CLIPstyler](https://github.com/cyclomon/CLIPstyler), [LLaVA](https://github.com/haotian-liu/LLaVA) and [Segment Anything](https://github.com/facebookresearch/segment-anything). We thank all the respective authors for open-sourcing their amazing work!



## Citation


If you find our work useful, please consider citing:






