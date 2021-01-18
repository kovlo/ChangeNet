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

#@interact(idx=widgets.IntSlider(min=0,max=int(len(dataset)-1)/8), inv=widgets.Checkbox())
import cv2
def explore_validation_dataset(idx, inv):
    #best_model.eval()
    change_net.eval()
    imgsize = 224
    
    referenceimg = np.zeros((imgsize,imgsize*8,3))
    testimg=np.zeros((imgsize,imgsize*8,3))
    labelimg=np.zeros((imgsize,imgsize*8))
    outputimg=np.zeros((imgsize,imgsize*8))
    for i in range(0,8):
        sample = dataset[idx+i]
        if not inv:
            reference = sample['reference'].unsqueeze(0).to(device)
            reference_img = sample['reference'].permute(1, 2, 0).cpu().numpy()
            test_img = sample['test'].permute(1, 2, 0).cpu().numpy()
            test = sample['test'].unsqueeze(0).to(device)
        else:
            reference = sample['test'].unsqueeze(0).to(device)
            reference_img = sample['test'].permute(1, 2, 0).cpu().numpy()
            test_img = sample['reference'].permute(1, 2, 0).cpu().numpy()
            test = sample['reference'].unsqueeze(0).to(device)
        
        label = sample['label'].type(torch.LongTensor).squeeze(0).cpu().numpy()
        #label = (sample['label']>0).type(torch.LongTensor).squeeze(0).cpu().numpy()
        
        #pred = best_model([reference, test])
        pred = change_net([reference, test])
        
        #print(pred.shape)
        _, output = torch.max(pred, 1)
        output = output.squeeze(0).cpu().numpy()

        referenceimg[:,i*imgsize:(i+1)*imgsize,:]=reference_img
        testimg[:,i*imgsize:(i+1)*imgsize,:]=test_img
        outputimg[:,i*imgsize:(i+1)*imgsize]=output
        labelimg[:,i*imgsize:(i+1)*imgsize]=label

    """
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 2, 1)
    plt.imshow(referenceimg)
    plt.title('Reference')
    fig.add_subplot(2, 2, 2)
    plt.imshow(testimg)
    plt.title('Test')
    fig.add_subplot(2, 2, 3)
    plt.imshow(labelimg, )
    plt.title('Label')
    fig.add_subplot(2, 2, 4)
    plt.imshow(outputimg)
    plt.title('ChangeNet Output')
    plt.show()
    """

    vline = np.ones((imgsize,1))
    hline = np.ones((1,imgsize*8*2+1))
    outimg = np.hstack((referenceimg[:,:,0],vline,testimg[:,:,0]))
    outimg= np.vstack((outimg,hline,np.hstack((labelimg,vline,outputimg))))

    image_sizeH = 128
    image_sizeW = 1024
    referenceimgRES = cv2.resize(referenceimg[:,:,0],(image_sizeH,image_sizeW))
    testimgRES = cv2.resize(testimg[:,:,0],(image_sizeH,image_sizeW))
    outputimgRES = cv2.resize(outputimg,(image_sizeH,image_sizeW))

    evaloutImg=np.vstack((referenceimgRES,testimgRES,outputimgRES))
#   plt.imshow(outimg)
#    plt.show()
    plt.imsave("./trainoutput/"+str(idx)+".png",outimg)
    print(idx)

test_pickle_file = './ChangeNet-arishin/change_dataset_testCD.pkl'
test_dataset = change_dataset_np.ChangeDatasetNumpy(test_pickle_file, data_transforms['val'])
dataset = val_dataset
#dataset = train_dataset
dataset = test_dataset

for idx in range (0,int(len(dataset)/8)):
    explore_validation_dataset(idx, False)

print("Finished.")

