# Joint Framework for Fingerprint Synthesis and Reconstruction using Generative Adversarial Network
  
> We utilized the similarity between fingerprint synthesis and reconstruction to propose a joint
framework for both tasks. This framework allows the user to
create a wide range of synthetic fingerprint datasets, which
can then be used in a wide variety of applications. The
framework backbone is a fingerprint generator based on
the Stylegan2 architecture, which is trained on a large
fingerprint dataset. The generator takes as input a latent
vector and transforms it into a realistic fingerprint image.
For fingerprint synthesis, which refers to the generation of
a random fingerprint, all one needs is to feed the generator
with a random normal distributed vector. In order to extend
the backbone for the fingerprint reconstruction task, an
encoder network has been designed to convert minutiae in-
formation into a latent vector so that the trained fingerprint
generator will be able to reconstruct the original fingerprint
from which the minutiae was extracted. In addition, the
proposed framework contains a unique component that
allows the user to control specific visual attributes of the
generated fingerprint.

<p align="center">
<img src="docs/Teaser.png" width="800px"/>
</p>

## Description   
Official Implementation of the paper "Joint Framework for Fingerprint Synthesis and Reconstruction using Generative Adversarial Network".
This repository contains both training and evaluation scripts.

## Recent Updates
**`2021.12.07`**: Initial code release.

## Applications
### Fingerprint Synthesis
Here, we use out framework to generate real-looking fingerprint images using our StyleGAN-based generator trained on NISTSD4 dataset. 
<p align="center">
<img src="docs/Synthesis.png" width="800px"/>
</p>


### Fingerprint Reconstruction
In this application we want to reconstruct a fingerprint image for it's minutiae information. 
<p align="center">
<img src="docs/Reconstruction.png" width="800px"/>
</p>

### Fingerprint Attribute Editor
Here, we are using out Fingerprint attribute editor, based on SeFa [2] algorithm for editing visual attributes of a generated fingerprint.
<p align="center">
<img src="docs/Attribute_Editor.png" width="600px"/>
</p>


## Getting Started
### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3
- torch == 1.6

### Installation
- Clone this repo:
``` 
git clone https://github.com/rafaelbou/pixel2style2pixel.git
cd pixel2style2pixel
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment/fingergen_env.yaml`.

### Inference Notebook
To help visualize the pSp framework on multiple tasks and to help you get started, we provide a Jupyter notebook found in `notebooks/inference_playground.ipynb` that allows one to visualize the various applications of pSp.   
The notebook will download the necessary pretrained models and run inference on the images found in `notebooks/images`.  
For the tasks of conditional image synthesis and super resolution, the notebook also demonstrates pSp's ability to perform multi-modal synthesis using 
style-mixing. 

### Pretrained Models
Please download the pre-trained models from the following links. Each pSp model contains the entire pSp architecture, including the encoder and decoder weights.
| Path | Description
| :--- | :----------
|[fingerprint_synthesis](https://drive.google.com/file/d/1BRqgVCa5PUeR0Unkz2ym13XXebrjJSpP/view?usp=sharing)  | Model trained with the NIST SD14 dataset for fingerprint synthesis.
|[fingerprint_reconstruction](https://drive.google.com/file/d/1qTsTKZW13C1FeMhqahiNdpTWqyBDPIwm/view?usp=sharing)  | Model trained with the NIST SD14 dataset for fingerprint reconstruction from minutiae set.

If you wish to use one of the pretrained models for training or inference, you may do so using the flag `--checkpoint_path`.

## Training
### Preparing your Data
- Currently, we provide support for Synthesis and Reconstruction datasets and experiments.
    - Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. 
    - Refer to `configs/transforms_config.py` for the transforms defined for each dataset/experiment. 
    - Finally, refer to `configs/data_configs.py` for the source/target data paths for the train and test sets
      as well as the transforms.
- If you wish to experiment with your own dataset, you can simply make the necessary adjustments in 
    1. `data_configs.py` to define your data paths.
    2. `transforms_configs.py` to define your own data transforms.

### Training
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.

#### **Synthesis - Fingerprint Generator**
```
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=/path/to/experiment \
--workers=8 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1
```

#### **Reconstruction - Minutiae-to-Vec Encoder**
```
--dataset_type=nist_sd14_mnt
--exp_dir=/hdd/PycharmProjects/fingerprints/models/train_mnt_debug
--workers=6
--batch_size=6
--test_batch_size=6
--test_workers=6
--val_interval=2500
--save_interval=5000
--encoder_type=MntToVecEncoderEncoderIntoW
--lpips_lambda=0.8
--l2_lambda=1
--fingernet_lambda=0
--generator_image_size=256
--style_count=14
--stylegan_weights=/hdd/PycharmProjects/fingerprints/stylegan2-pytorch_origin/checkpoint/480000.pt
--label_nc=1
--input_nc=3
```

### Additional Notes
- See `options/train_options.py` for all training-specific flags. 
- See `options/test_options.py` for all test-specific flags.
- If you wish to resume from a specific checkpoint, you may do so using `--checkpoint_path`.

## Testing
### Inference
Having trained your model, you can use `scripts/inference.py` to apply the model on a set of images.   
For example, 
```
python scripts/inference.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs
```
Additional notes to consider: 
- During inference, the options used during training are loaded from the saved checkpoint and are then updated using the 
test options passed to the inference script. For example, there is no need to pass `--dataset_type` or `--label_nc` to the 
 inference script, as they are taken from the loaded `opts`.


### Multi-Modal Synthesis with Style-Mixing
Given a trained model for conditional image synthesis or super-resolution, we can easily generate multiple outputs 
for a given input image. This can be done using the script `scripts/style_mixing.py`.    
For example, running the following command will perform style-mixing for a segmentation-to-image experiment:
```
python scripts/style_mixing.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=/path/to/experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data/ \
--test_batch_size=4 \
--test_workers=4 \
--n_images=25 \
--n_outputs_to_generate=5 \
--latent_mask=8,9,10,11,12,13,14,15,16,17
``` 
Here, we inject `5` randomly drawn vectors and perform style-mixing on the latents `[8,9,10,11,12,13,14,15,16,17]`.  

Additional notes to consider: 
- To perform style-mixing on a subset of images, you may use the flag `--n_images`. The default value of `None` will perform 
style mixing on every image in the given `data_path`. 
- You may also include the argument `--mix_alpha=m` where `m` is a float defining the mixing coefficient between the 
input latent and the randomly drawn latent.
- When performing style-mixing for super-resolution, please provide a single down-sampling value using `--resize_factors`.
- By default, the images will be saved at resolutiosn of 1024x1024, the original output size of StyleGAN. If you wish to save 
outputs resized to resolutions of 256x256, you can do so by adding the flag `--resize_outputs`.


### Computing Metrics
Similarly, given a trained model and generated outputs, we can compute the loss metrics on a given dataset.  
These scripts receive the inference output directory and ground truth directory.
- Calculating the identity loss: 
```
python scripts/calc_id_loss_parallel.py \
--data_path=/path/to/experiment/inference_outputs \
--gt_path=/path/to/test_images \
```
- Calculating LPIPS loss:
```
python scripts/calc_losses_on_images.py \
--mode lpips
--data_path=/path/to/experiment/inference_outputs \
--gt_path=/path/to/test_images \
```
- Calculating L2 loss:
```
python scripts/calc_losses_on_images.py \
--mode l2
--data_path=/path/to/experiment/inference_outputs \
--gt_path=/path/to/test_images \
```

## Additional Applications
To better show the flexibility of our pSp framework we present additional applications below.

As with our main applications, you may download the pretrained models here: 
| Path | Description
| :--- | :----------
|[Toonify](https://drive.google.com/file/d/1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz/view)  | pSp trained with the FFHQ dataset for toonification using StyleGAN generator from [Doron Adler](https://linktr.ee/Norod78) and [Justin Pinkney](https://www.justinpinkney.com/).

### Toonify
Using the toonify StyleGAN built by [Doron Adler](https://linktr.ee/Norod78) and [Justin Pinkney](https://www.justinpinkney.com/),
we take a real face image and generate a toonified version of the given image. We train the pSp encoder to directly reconstruct real 
face images inside the toons latent space resulting in a projection of each image to the closest toon. We do so without requiring any labeled pairs
or distillation!
<p align="center">
<img src="docs/toonify_input.jpg" width="800px"/>
<img src="docs/toonify_output.jpg" width="800px"/>
</p>

This is trained exactly like the StyleGAN inversion task with several changes:   
- Change from FFHQ StyleGAN to toonifed StyleGAN (can be set using `--stylegan_weights`)
    - The toonify generator is taken from [Doron Adler](https://linktr.ee/Norod78) and [Justin Pinkney](https://www.justinpinkney.com/) 
      and converted to Pytorch using [rosinality's](https://github.com/rosinality/stylegan2-pytorch) conversion script.
    - For convenience, the converted generator Pytorch model may be downloaded [here](https://drive.google.com/file/d/1r3XVCt_WYUKFZFxhNH-xO2dTtF6B5szu/view?usp=sharing).
- Increase `id_lambda` from `0.1` to `1`  
- Increase `w_norm_lambda` from `0.005` to `0.025`  

We obtain the best results after around `6000` iterations of training (can be set using `--max_steps`) 


## Repository structure
| Path | Description <img width=200>
| :--- | :---
| pixel2style2pixel | Repository root folder
| &boxvr;&nbsp; configs | Folder containing configs defining model/data paths and data transforms
| &boxvr;&nbsp; criteria | Folder containing various loss criterias for training
| &boxvr;&nbsp; datasets | Folder with various dataset objects and augmentations
| &boxvr;&nbsp; environment | Folder containing Anaconda environment used in our experiments
| &boxvr; models | Folder containting all the models and training objects
| &boxv;&nbsp; &boxvr;&nbsp; encoders | Folder containing our pSp encoder architecture implementation and ArcFace encoder implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; mtcnn | MTCNN implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
| &boxv;&nbsp; &boxvr;&nbsp; stylegan2 | StyleGAN2 model from [rosinality](https://github.com/rosinality/stylegan2-pytorch)
| &boxv;&nbsp; &boxur;&nbsp; psp.py | Implementation of our pSp framework
| &boxvr;&nbsp; notebook | Folder with jupyter notebook containing pSp inference playground
| &boxvr;&nbsp; options | Folder with training and test command-line options
| &boxvr;&nbsp; scripts | Folder with running scripts for training and inference
| &boxvr;&nbsp; training | Folder with main training logic and Ranger implementation from [lessw2020](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
| &boxvr;&nbsp; utils | Folder with various utility functions
| <img width=300> | <img>

## TODOs
- [ ] Add multi-gpu support

## Credits
**StyleGAN2 implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**MTCNN, IR-SE50, and ArcFace models and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**CurricularFace model and implementation:**   
https://github.com/HuangYG123/CurricularFace  
Copyright (c) 2020 HuangYG123  
License (MIT) https://github.com/HuangYG123/CurricularFace/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  


## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/abs/2008.00951">Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation</a>:

```
@article{richardson2020encoding,
  title={Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation},
  author={Richardson, Elad and Alaluf, Yuval and Patashnik, Or and Nitzan, Yotam and Azar, Yaniv and Shapiro, Stav and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2008.00951},
  year={2020}
}
```
