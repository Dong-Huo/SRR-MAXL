# Learning to Recover Spectral Reflectance from RGB Images

Code for this paper [Learning to Recover Spectral Reflectance from RGB Images](https://arxiv.org/abs/2304.02162)
[Dong Huo](https://dong-huo.github.io/), [Jian Wang](https://jianwang-cmu.github.io/), [Yiming Qian](https://yi-ming-qian.github.io/), [Yee-Hong Yang](http://webdocs.cs.ualberta.ca/~yang/)

## Overview

This paper tackles spectral reflectance recovery (SRR) from RGB images. Since capturing ground-truth spectral reflectance and camera spectral sensitivity are challenging and costly, most existing approaches are trained on synthetic images and utilize the same parameters for all unseen testing images, which are suboptimal especially when the trained models are tested on real images because they never exploit the internal information of the testing images. To address this issue, we adopt a self-supervised meta-auxiliary learning (MAXL) strategy that fine-tunes the well-trained network parameters with each testing image to combine external with internal information. To the best of our knowledge, this is the first work that successfully adapts the MAXL strategy to this problem. Instead of relying on naive end-to-end training, we also propose a novel architecture that integrates the physical relationship between the spectral reflectance and the corresponding RGB images into the network based on our mathematical analysis. Besides, since the spectral reflectance of a scene is independent to its illumination while the corresponding RGB images are not, we recover the spectral reflectance of a scene from its RGB images captured under multiple illuminations to further reduce the unknown. Qualitative and quantitative evaluations demonstrate the effectiveness of our proposed network and of the MAXL.

## Datasets

The datasets utilized in our paper can be downloaded via the links below:
- [Our real dataset](https://dong-huo.github.io/)
- [ICVL](https://icvl.cs.bgu.ac.il/hyperspectral/)
- [TokyoTech](http://www.ok.sc.e.titech.ac.jp/res/MSI/MSIdata59.html)
- [CSS](https://www.gujinwei.org/research/camspec/db.html)

## Training

run ```python crop_patches.py``` before the training to generate the training patches.

pre-training: ```python train.py --num_images number_of_illuminations --dataset_path your_data_path --output_dir your_checkpoint_path --result_dir your_result_path```

meta_training: ```python meta_train.py --num_images number_of_illuminations --dataset_path your_data_path --output_dir your_checkpoint_path --result_dir your_result_path --resume your_pretrained_checkpoint_path```


## Testing on synthetic data

pre-trained model: ``` python train.py --num_images number_of_illuminations --dataset_path your_data_path --output_dir your_checkpoint_path --result_dir your_result_path --resume your_pretrained_checkpoint_path --eval```

meta-trained model: ```python meta_train.py --num_images number_of_illuminations --dataset_path your_data_path --output_dir your_checkpoint_path --result_dir your_result_path --resume your_metatrained_checkpoint_path --eval```

## Testing on real data

pre-trained model: ``` python real_testing_nometa.py --num_images number_of_illuminations --dataset_path your_data_path --output_dir your_checkpoint_path --result_dir your_result_path --resume your_pretrained_checkpoint_path --eval```

meta-trained model: ``` python real_testing_meta.py --num_images number_of_illuminations --dataset_path your_data_path --output_dir your_checkpoint_path --result_dir your_result_path --resume your_metatrained_checkpoint_path --eval```

## Citation

If you use this code and data for your research, please cite our paper.

```
@article{huo2023learning,
  title={Learning to Recover Spectral Reflectance from RGB Images},
  author={Huo, Dong and Wang, Jian and Qian, Yiming and Yang, Yee-Hong},
  journal={arXiv preprint arXiv:2304.02162},
  year={2023}
}
```

If you meet any problems about the implementation, feel free to contact us. Thank you very much!
