# IDMIL
## IDMIL: An alignment-free interpretable deep multiple instance learning (MIL) for predicting disease from whole-metagenomic data
The implementation is in Python. The usage with brief description is provided in the file titled "**Scripts_and_arguments.pdf**". Information about the datasets used in the paper are provided in the file "**dataset_info.txt**".

### Summary
The human body hosts more microbial organisms than human cells. Analysis of this microbial diversity provides key insight into the role played by these microorganisms on human health. Metagenomics is the collective DNA sequencing of coexisting microbial organisms in a host. This has several applications in precision medicine, agriculture, environmental science, and forensics. State-of-the-art predictive models for host phenotype predictions from microbiome rely on alignment and expert-curated reference databases to perform assembly, extensive pruning and microbial profiling on the whole-metagenomic data before prediction. This process is time-consuming and it discards the majority of the sequences for downstream analysis limiting the potential of whole-metagenomics. We formulate the problem of predicting human disease from whole-metagenomic data using Multiple Instance Learning (MIL), a popular supervised learning paradigm. Our proposed alignment-free approach provides higher accuracy in prediction by harnessing the capability of deep convolutional neural network (CNN) within a MIL framework and provides interpretability via neural attention mechanism.

### Method
MIL allows for representing a metagenomic sample as a collection of similar sequences. The MIL formulation combined with the hierarchical feature extraction capability of deep-CNN provides significantly better predictive performance compared to popular existing approaches. The attention mechanism allows for the identification of groups of DNA sequences that are likely to be correlated to diseases providing the much-needed interpretation. Our proposed approach does not rely on alignment, assembly, and manually curated databases; making it fast and scalable for large-scale metagenomic data. We evaluate our method on well-known large-scale metagenomic studies and show that our proposed approach outperforms comparative state-of-the-art methods for disease prediction.
