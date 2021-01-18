# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Train Model
# 
# #### References
# * https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# * https://pytorch.org/tutorials/beginner/saving_loading_models.html
# * https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# %%



# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import models
import losses
import utils_train
import change_dataset_np
import matplotlib.pyplot as plt
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import helper_augmentations

from IPython.display import clear_output, display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

# Hyperparameters
num_epochs = 50
num_classes = 2
batch_size = 20
img_size = 224
base_lr = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
#num_gpu = torch.cuda.device_count()
num_gpu = 1
batch_size *= num_gpu
base_lr *= num_gpu
print('Number of GPUs Available:', num_gpu)

train_pickle_file = './ChangeNet-arishin/change_dataset_trainCD.pkl'
val_pickle_file = './ChangeNet-arishin/change_dataset_valCD.pkl'
test_pickle_file = './ChangeNet-arishin/change_dataset_testCD.pkl'

# #### Define Transformation

# %%
#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(img_size),
        #transforms.RandomHorizontalFlip(),
        #transforms.ToPILImage(),
        transforms.Resize(img_size),
        #transforms.CenterCrop(img_size),
        #helper_augmentations.SwapReferenceTest(),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
        #helper_augmentations.JitterGamma(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(img_size),
        #transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# #### Load Dataset

# %%
# Create training and validation datasets
train_dataset = change_dataset_np.ChangeDatasetNumpy(train_pickle_file, data_transforms['train'])
val_dataset = change_dataset_np.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'])
test_dataset = change_dataset_np.ChangeDatasetNumpy(test_pickle_file, data_transforms['val'])

image_datasets = {'train': train_dataset, 'val': val_dataset}
# Create training and validation dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}
dataloaders_dict = {'train': train_loader, 'val': val_loader}

# #### Start Tensorboard Interface

# %%
# Default directory "runs"
writer = SummaryWriter()

# #### Initialize Model

# %%
img_reference_dummy = torch.randn(1,3,img_size,img_size)
img_test_dummy = torch.randn(1,3,img_size,img_size)
change_net = models.ChangeNet(num_classes=num_classes)

# Add on Tensorboard the Model Graph
writer.add_graph(change_net, [[img_reference_dummy, img_test_dummy]])

# #### Send Model to GPUs (If Available)
# %%
#if num_gpu > 1:
#    change_net = nn.DataParallel(change_net)
change_net = change_net.to(device)

# #### Load Weights

# %%
checkpointname = './best_model'+str(num_classes)+'CD.pkl'
import os.path

if os.path.exists((checkpointname)):
    checkpoint = torch.load(checkpointname)
    change_net.load_state_dict(checkpoint);
    print('Checkpoint '+checkpointname+' is loaded.')

# #### Initialize Loss Functions and Optimizers

# %%
#criterion = nn.CrossEntropyLoss()
# If there are more than 2 classes the alpha need to be a list
criterion = losses.FocalLoss(gamma=2.0, alpha=0.25)
optimizer = optim.Adam(change_net.parameters(), lr=base_lr)    
sc_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)    

# #### Train Model

# %%
best_model, _ = utils_train.train_model(change_net, dataloaders_dict, criterion, optimizer, sc_plt, writer, device, num_epochs=num_epochs)
torch.save(best_model.state_dict(), './best_model'+str(num_classes)+'CD.pkl')

print("Finished.")

