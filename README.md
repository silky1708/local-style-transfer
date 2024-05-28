# LEAST: "Local" text-conditioned image style transfer


*Accepted to AI for Content Creation (AI4CC) Workshop at CVPR 2024*

*[Silky Singh](https://silky1708.github.io/), [Surgan Jandial](https://surgan12.github.io/), [Simra Shahid](https://scholar.google.com/citations?user=RXM-KSQAAAAJ&hl=en), [Abhinav Java](https://java-abhinav07.github.io/).*  
Media and Data Science Research (MDSR), Adobe


Project Page: [arXiv](https://arxiv.org/abs/2405.16330)


![local style transfer teaser](assets/main_qual_results.png)


## Installation


Create a conda environment using the provided `environment.yml` file:  

```
conda env create -f environment.yml  
conda activate least
```



## Getting Started



Download [SAM's `vit-h` checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place it here: `segment-anything/checkpoints/sam_vit_h_4b8939.pth` exactly following the name convention.


A working notebook is provided here: `local_style_transfer.ipynb`. To run the notebook using the environment `least`:  
```
conda install -c anaconda ipykernel  
python -m ipykernel install --user --name=least
```


Given a path to an image and a style description, our method LEAST attempts to constrain the stylization process to the target region in the image, while maintaining the content and structure of the rest of the image.



## Dataset


We collected a set of 25 natural images to perform evaluation of our work against the baselines. The dataset is provided in the `dataset` directory. Please note that the copyrights exist with the owners of these images.




## Acknowledgments


This repository is heavily based on [CLIPstyler](https://github.com/cyclomon/CLIPstyler), [LLaVA](https://github.com/haotian-liu/LLaVA) and [Segment Anything](https://github.com/facebookresearch/segment-anything). We thank all the respective authors for open-sourcing their amazing work!



## Citation


If you find our work useful, please consider citing:   
```
@misc{singh2024least,
      title={LEAST: "Local" text-conditioned image style transfer}, 
      author={Silky Singh and Surgan Jandial and Simra Shahid and Abhinav Java},
      year={2024},
      eprint={2405.16330},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

