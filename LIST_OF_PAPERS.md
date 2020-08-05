# fastMRI publications and preprints

The following is a list of fastMRI publication and preprints with abstracts andlinks to manucripts, code, etc.

## End-to-End Variational Networks for Accelerated MRI Reconstruction
*MICCAI 2020*

[arXiv](https://arxiv.org/abs/2004.06688)

The slow acquisition speed of magnetic resonance imaging (MRI) has led to the development of two complementary methods: acquiring multiple views of the anatomy simultaneously (parallel imaging) and acquiring fewer samples than necessary for traditional signal processing methods (compressed sensing). While the combination of these methods has the potential to allow much faster scan times, reconstruction from such undersampled multi-coil data has remained an open problem. In this paper, we present a new approach to this problem that extends previously proposed variational methods by learning fully end-to-end. Our method obtains new state-of-the-art results on the fastMRI dataset for both brain and knee MRIs.
```BibTeX
@misc{sriram2020endtoend, 
    title={End-to-End Variational Networks for Accelerated MRI Reconstruction}, 
    author={Anuroop Sriram and Jure Zbontar and Tullie Murrell and Aaron Defazio and C. Lawrence Zitnick and Nafissa Yakubova and Florian Knoll and Patricia Johnson}, 
    year={2020}, 
    eprint={2004.06688}, 
    archivePrefix={arXiv}, 
    primaryClass={eess.IV}
}
```

## Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study
*American Journal of Roentgenology*

[publication](https://www.ajronline.org/doi/abs/10.2214/AJR.20.23313)

**Objective**

Deep Learning (DL) image reconstruction has the potential to disrupt the current state of MR imaging by significantly decreasing the time required for MR exams. Our goal was to use DL to accelerate MR imaging in order to allow a 5-minute comprehensive examination of the knee, without compromising image quality or diagnostic accuracy.

**Methods**

A DL model for image reconstruction using a variational network was optimized. The model was trained using dedicated multi-sequence training, in which a single reconstruction model was trained with data from multiple sequences with different contrast and orientations. Following training, data from 108 patients were retrospectively undersampled in a manner that would correspond with a net 3.49-fold acceleration of fully-sampled data acquisition and 1.88-fold acceleration compared to our standard two-fold accelerated parallel acquisition. An interchangeability study was performed, in which the ability of 6 readers to detect internal derangement of the knee was compared for the clinical and DL-accelerated images.

**Results**

The study demonstrated a high degree of interchangeability between standard and DL-accelerated images. In particular, results showed that interchanging the sequences would result in discordant clinical opinions no more than 4% of the time for any feature evaluated. Moreover, the accelerated sequence was judged by all six readers to have better quality than the clinical sequence.

**Conclusions**

An optimized DL model allowed for acceleration of knee images which performed interchangeably with standard images for the detection of internal derangement of the knee. Importantly, readers preferred the quality of accelerated images to that of standard clinical images.

```BibTeX
@article{fastmri2020,
    Author = {Recht, Michael P. and Zbontar, Jure and Sodickson, Daniel K. and Knoll, Florian and Yakubova, Nafissa and Sriram, Anuroop and Murrell, Tullie and Defazio, Aaron and Rabbat, Michael and Rybak, Leon and Kline, Mitchell and Ciavarra, Gina and Alaia, Erin F. and Samim, Mohammad and Walter, William R. and Lin, Dana and Lui, Yvonne W. and Muckley, Matthew and Huang, Zhengnan and Johnson, Patricia and Stern, Ruben and Zitnick, C. Lawrence}, Journal = {American Journal of Roentgenology}, 
    Month = {2020/07/09}, 
    Title = {Using Deep Learning to Accelerate Knee MRI at 3T: Results of an Interchangeability Study}, 
    Year = {2020}
}
```

## GrappaNet: Combining Parallel Imaging With Deep Learning for Multi-Coil MRI Reconstruction
[arXiv](https://arxiv.org/abs/1910.12325) [publication](https://openaccess.thecvf.com/content_CVPR_2020/html/Sriram_GrappaNet_Combining_Parallel_Imaging_With_Deep_Learning_for_Multi-Coil_MRI_CVPR_2020_paper.html) 

Magnetic Resonance Image (MRI) acquisition is an inherently slow process which has spurred the development of two different acceleration methods: acquiring multiple correlated samples simultaneously (parallel imaging) and acquiring fewer samples than necessary for traditional signal processing methods (compressed sensing). Both methods provide complementary approaches to accelerating MRI acquisition. In this paper, we present a novel method to integrate traditional parallel imaging methods into deep neural networks that is able to generate high quality reconstructions even for high acceleration factors. The proposed method, called GrappaNet, performs progressive reconstruction by first mapping the reconstruction problem to a simpler one that can be solved by a traditional parallel imaging methods using a neural network, followed by an application of a parallel imaging method, and finally fine-tuning the output with another neural network. The entire network can be trained end-to-end. We present experimental results on the recently released fastMRI dataset and show that GrappaNet can generate higher quality reconstructions than competing methods for both 4x and 8x acceleration.

```BibTeX
@InProceedings{Sriram_2020_CVPR, 
    author = {Sriram, Anuroop and Zbontar, Jure and Murrell, Tullie and Zitnick, C. Lawrence and Defazio, Aaron and Sodickson, Daniel K.},
    title = {GrappaNet: Combining Parallel Imaging With Deep Learning for Multi-Coil MRI Reconstruction}, 
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
    month = {June}, 
    year = {2020}
}
```

## MRI Banding Removal via Adversarial Training
[arXiv](https://arxiv.org/abs/2001.08699)

MRI images reconstructed from sub-sampled Cartesian data using deep learning techniques often show a characteristic banding (sometimes described as streaking), which is particularly strong in low signal-to-noise regions of the reconstructed image. In this work, we propose the use of an adversarial loss that penalizes banding structures without requiring any human annotation. Our technique greatly reduces the appearance of banding, without requiring any additional computation or post-processing at reconstruction time. We report the results of a blind comparison against a strong baseline by a group of expert evaluators (board-certified radiologists), where our approach is ranked superior at banding removal with no statistically significant loss of detail.

```BibTeX
@misc{defazio2020mri, 
    title={MRI Banding Removal via Adversarial Training}, 
    author={Aaron Defazio and Tullie Murrell and Michael P. Recht}, year={2020}, 
    eprint={2001.08699}, 
    archivePrefix={arXiv}, primaryClass={eess.IV}
}
```

## Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry
[arXiv](https://arxiv.org/abs/1912.01101)

Deep learning approaches to accelerated MRI take a matrix of sampled Fourier-space lines as input and produce a spatial image as output. In this work we show that by careful choice of the offset used in the sampling procedure, the symmetries in k-space can be better exploited, producing higher quality reconstructions than given by standard equally-spaced samples or randomized samples motivated by compressed sensing. 

```BibTeX
@misc{defazio2019offset, 
    title={Offset Sampling Improves Deep Learning based Accelerated MRI Reconstructions by Exploiting Symmetry}, 
    author={Aaron Defazio}, 
    year={2019}, 
    eprint={1912.01101}, 
    archivePrefix={arXiv}, 
    primaryClass={eess.IV}
}
```

## Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge 

*Magnetic Resonance in Medicine*

[publication](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28338) [arXiv](https://arxiv.org/abs/2001.02518) [Code](https://github.com/facebookresearch/fastMRI) 

**Purpose**

To advance research in the field of machine learning for MR image reconstruction with an open challenge.

**Methods**

We provided participants with a dataset of raw k‐space data from 1,594 consecutive clinical exams of the knee. The goal of the challenge was to reconstruct images from these data. In order to strike a balance between realistic data and a shallow learning curve for those not already familiar with MR image reconstruction, we ran multiple tracks for multi‐coil and single‐coil data. We performed a two‐stage evaluation based on quantitative image metrics followed by evaluation by a panel of radiologists. The challenge ran from June to December of 2019.

**Results**

We received a total of 33 challenge submissions. All participants chose to submit results from supervised machine learning approaches.

**Conclusions**

The challenge led to new developments in machine learning for image reconstruction, provided insight into the current state of the art in the field, and highlighted remaining hurdles for clinical adoption.
```BibTeX
@article{doi:10.1002/mrm.28338, 
    Author = {Knoll, Florian and Murrell, Tullie and Sriram, Anuroop and Yakubova, Nafissa and Zbontar, Jure and Rabbat, Michael and Defazio, Aaron and Muckley, Matthew J. and Sodickson, Daniel K. and Zitnick, C. Lawrence and Recht, Michael P.}, 
    Journal = {Magnetic Resonance in Medicine}, 
    Title = {Advancing machine learning for MR image reconstruction with an open competition: Overview of the 2019 fastMRI challenge},
}
```

## fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning
*Radiology: Artificial Intelligence*

[publication](https://pubs.rsna.org/doi/10.1148/ryai.2020190007) [Code](https://github.com/facebookresearch/fastMRI)

A publicly available dataset containing k-space data as well as Digital Imaging and Communications in Medicine image data of knee images for accelerated MR image reconstruction using machine learning is presented.
@article{doi:10.1148/ryai.2020190007, 
Author = {Knoll, Florian and Zbontar, Jure and Sriram, Anuroop and Muckley, Matthew J. and Bruno, Mary and Defazio, Aaron and Parente, Marc and Geras, Krzysztof J. and Katsnelson, Joe and Chandarana, Hersh and Zhang, Zizhao and Drozdzal, Michal and Romero, Adriana and Rabbat, Michael and Vincent, Pascal and Pinkerton, James and Wang, Duo and Yakubova, Nafissa and Owens, Erich and Zitnick, C. Lawrence and Recht, Michael P. and Sodickson, Daniel K. and Lui, Yvonne W.}, 
Journal = {Radiology: Artificial Intelligence},
Title = {fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning}, 
Year = {2020}}



## Reducing Uncertainty in Undersampled MRI Reconstruction With Active Acquisition
*CVPR 2019*

[arXiv](https://arxiv.org/abs/1902.03051)

The goal of MRI reconstruction is to restore a high fidelity image from partially observed measurements. This partial view naturally induces reconstruction uncertainty that can only be reduced by acquiring additional measurements. In this paper, we present a novel method for MRI reconstruction that, at inference time, dynamically selects the measurements to take and iteratively refines the prediction in order to best reduce the reconstruction error and, thus, its uncertainty. We validate our method on a large scale knee MRI dataset, as well as on ImageNet. Results show that (1) our system successfully outperforms active acquisition baselines; (2) our uncertainty estimates correlate with error maps; and (3) our ResNet-based architecture surpasses standard pixel-to-pixel models in the task of MRI reconstruction. The proposed method not only shows high-quality reconstructions but also paves the road towards more applicable solutions for accelerating MRI.
```BibTeX
@InProceedings{Zhang_2019_CVPR,
    author = {Zhang, Zizhao and Romero, Adriana and Muckley, Matthew J. and Vincent, Pascal and Yang, Lin and Drozdzal, Michal},
    title = {Reducing Uncertainty in Undersampled MRI Reconstruction With Active Acquisition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```


## Active MR k-space Sampling with Reinforcement Learning
*MICCAI 2020*

[arXiv](https://arxiv.org/abs/2007.10469)

Deep learning approaches have recently shown great promise in accelerating magnetic resonance image (MRI) acquisition. The majority of existing work have focused on designing better reconstruction models given a pre-determined acquisition trajectory, ignoring the question of trajectory optimization. In this paper, we focus on learning acquisition trajectories given a fixed image reconstruction model. We formulate the problem as a sequential decision process and propose the use of reinforcement learning to solve it. Experiments on a large scale public MRI dataset of knees show that our proposed models significantly outperform the state-of-the-art in active MRI acquisition, over a large range of acceleration factors.

```BibTeX
@misc{pineda2020active,
    title={Active MR k-space Sampling with Reinforcement Learning},
    author={Luis Pineda and Sumana Basu and Adriana Romero and Roberto Calandra and Michal Drozdzal},
    year={2020},
    eprint={2007.10469},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
