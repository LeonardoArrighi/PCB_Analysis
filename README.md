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