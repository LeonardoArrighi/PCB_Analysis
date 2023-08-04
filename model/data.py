import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms
import numpy as np

# mean and std deviation of the dataset calculated
MEAN = [0.3895, 0.4890, 0.4233]
STD = [0.2877, 0.2396, 0.2902]

# original size
IMAGE_SIZE = 768

# morphological data augmentation transformations applied to the train set 
DEFAULT_TRAIN_TRANSFORM = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                                    transforms.RandomRotation(45),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = MEAN, std = STD),
                                    ])    
# transformations of the test set
DEFAULT_TEST_TRANSFORM = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = MEAN, std = STD),                                  
                                    ])   
    

# function to apply transformations and split dataset
def get_dataset(datadir, test_pct, valid_pct, train_transforms=DEFAULT_TRAIN_TRANSFORM, test_transforms=DEFAULT_TEST_TRANSFORM):
    data = torchvision.datasets.ImageFolder(datadir,       
                    transform=test_transforms)

    test_size = int(len(data) * test_pct)
    valid_size = int(len(data) * valid_pct)

    train_data, test_data, valid_data = torch.utils.data.random_split(data, [len(data)-valid_size-test_size, test_size, valid_size])

    train_data.dataset.transform = train_transforms
    return train_data, test_data, valid_data

# funciton to load dataset into dataloaders
def get_dataloaders(datadir, batch_size_train, batch_size_test=None, train_transforms=DEFAULT_TRAIN_TRANSFORM, test_transform=DEFAULT_TEST_TRANSFORM, test_percentage=0, validation_percentage=.22222):
    train_data, test_data, valid_data = get_dataset(datadir, test_percentage, validation_percentage, train_transforms, test_transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False, num_workers=4)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size_test, shuffle=False, num_workers=4)
    return trainloader, testloader, validloader


    

