# **SABench**


## **Methods**

Spatial alignment uses spatial transcriptomic information to align multiple slices, enables 3D reconstruction from 2D slices and provides an integrative perspective for spatial biology. In this work, we present a benchmark study encompassing 11 alignment methods across datasets with varying characteristics and technological platforms.  
The following methods were included:  
• **PASTE**: 《Alignment and integration of spatial transcriptomics data》  
• **PASTE2**: 《Partial alignment of multislice spatially resolved transcriptomics data》  
• **STAligner**: 《Integrating spatial transcriptomics data across different conditions, technologies and developmental stages》  
• **GPSA**: 《Alignment of spatial genomics data using deep Gaussian processes》  
• **SLAT**: 《Spatial-linked alignment tool (SLAT) for aligning heterogenous slices》  
• **STalign**: 《STalign: Alignment of spatial transcriptomics data using diffeomorphic metric mapping》  
• **CAST**: 《Search and match across spatial omics samples at single-cell resolution》  
• **STAIR**: 《Spatial Transcriptomic Alignment, Integration, and de novo 3D Reconstruction by STAIR》  
• **SPACEL**: 《SPACEL: deep learning-based characterization of spatial transcriptome architectures》  
• **Spateo**: 《Spatiotemporal modeling of molecular holograms》  
• **SANTO**: 《SANTO: a coarse-to-fine alignment and stitching method for spatial omics》

## **Datasets**

To evaluate the alignment methods, we curated a diverse set of benchmarking datasets consisting of 240 tissue slices sourced from published studies and public repositories. These datasets were form different spatial transcriptomics platforms, including 10x Visium, Visium HD, ST, MERFISH, Stereo-Seq, BaristaSeq, STARmap, STARmap PLUS, Slide-seq, Slide-seq V2, Open-ST, Xenium, Xenium 5k, and CosMx. In addition to real data, we generated 35 simulated slice pairs, resulting in a total of 295 alignment tasks for each method. Our article provides links to raw data sources.

## **Overview**

Our evaluation framework encompasses six key aspects:  
(1) Accuracy, assessed through gene-based metrics and landmark-based scores; for datasets with known 3D coordinates, quantitative measures like Mean Absolute Error (MAE) were computed.  
(2) Efficiency, evaluated in terms of computational time and peak memory consumption.  
(3) Robustness, examined by introducing perturbations such as changes in slice overlap ratios and initial rotation angles.  
(4) Downstream performance, analyzed by measuring the impact of alignment on subsequent 3D spatial clustering.  
(5) Challenging scenarios, where we examined method adaptability in three contexts: serial sections, cross-platform, and large-scale datasets alignment—alongside proposing potential solutions to common issues.  
(6) Usability, rated via a standardized scoring system.   
Finally, we provided practical guidelines to users.

## **Tutorial**
We have encapsulated the core functionalities of the framework presented in this study into a practical toolkit and provided detailed tutorials, with the aim of improving the development and application efficiency of related methods. First, the code supports rapid computation of evaluation metrics involved in the manuscript and includes a module for direct comparison with existing alignment methods. This allows developers to evaluate and optimize their algorithms effectively, while also facilitating the integration of new methods into existing benchmarking systems. Furthermore, we offer executable code and detailed operational guidelines for large-scale data solutions, including interactive coarse alignment based on manually annotated landmarks, grid-based downsampling, and resolution restoration. We also include parameter-tuning strategies tailored to different methods to help users better apply the toolkit to real-world data.(see the directory Tutorial)
