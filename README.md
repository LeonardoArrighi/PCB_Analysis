# PCBs_Analysis
This GitHub repo presents a Deep Learning model for PCB image classification, achieving high accuracy. We provide a comprehensive dataset and explainability analysis. Our work contributes to the field of image classification and explainable AI. Code and resources are available for use and further research.

In particular the repo contains:
- model: a folder containing everything used to train the model (ResNet50d):
    main: the train
    data: data augmentation tranformations, functions used to load dataset into dataloaders (train-test-validation)
    optimizer: RAdam code
    utils: some useful functions

- notebooks: a folder containing the notebooks used to check and plot results:
    plot: results of the training
    analysis: tests of the model
    masking_images: this notebook contains the passages applied to masking out relevant areas. First of all the areas corresponding to the relevant parts highlighted by CAM-based techniques. Then the areas corresponding to the failed components of the pictures segmented using external tools.