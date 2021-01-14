# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Dataset Explore
# This dataset is organized in different folders, but we're actually interested on a pair of input images and the expected label that highlight differences between the 2 input images

# %%
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pickle
import cv2 

train_file_list = './ChangeNet-arishin/trainCD.txt'
val_file_list = './ChangeNet-arishin/valCD.txt'
base_data_dir = './'
train_pickle_file = './ChangeNet-arishin/change_dataset_trainCD.pkl'
val_pickle_file = './ChangeNet-arishin/change_dataset_valCD.pkl'

validation_set = {}
training_set = {}

image_size  = 256
image_sizeH = 128
image_sizeW = 1024

# #### Parse the path files

# %%
train_file_list = [line.rstrip('\n').split() for line in open(train_file_list)]
val_file_list = [line.rstrip('\n').split() for line in open(val_file_list)]
print('Length Training Set:', len(train_file_list))
print('Length Validation Set:', len(val_file_list))
size_train = len(train_file_list)
size_validation = len(val_file_list)

# #### Load Validation Set On Memory
# %%
idx2 = 0
for idx, entry in enumerate(tqdm(val_file_list[:750])):
    # Load the reference, test and label images
    trio_img = cv2.imread(base_data_dir + entry[0],cv2.IMREAD_UNCHANGED)
    trio_img=np.array(trio_img)

    reference_img = trio_img[0:image_sizeH,:]/1000/42.42
    #reference_img = np.tile(reference_img[:,:,np.newaxis],(1,1,3))

    test_img = trio_img[image_sizeH:2*image_sizeH,:]/1000/42.42
    #test_img = np.tile(test_img[:,:,np.newaxis],(1,1,3))

    label_img = trio_img[2*image_sizeH:,:]
    label_img = (label_img>0).astype(int)

    # Populate validation dictionary with tupple (reference,test,label)
    for i in range(0,8):
        # Populate training dictionary with tupple (reference,test,label)
        validation_set[idx2] = reference_img[:,i*128:(i+1)*128], test_img[:,i*128:(i+1)*128], label_img[:,i*128:(i+1)*128]   
        idx2+=1

print('Saving Pickle Validation Set')
with open(val_pickle_file, 'wb') as handle:
    pickle.dump(validation_set, handle, protocol=4)
# #### Load Training Set On Memory
idx2 = 0
for idx, entry in enumerate(tqdm(train_file_list[:5000])):
    trio_img = cv2.imread(base_data_dir + entry[0],cv2.IMREAD_UNCHANGED)   
    trio_img=np.array(trio_img)

    reference_img = trio_img[0:image_sizeH,:]/1000/42.42
    #reference_img = np.tile(reference_img[:,:,np.newaxis],(1,1,3))
    
    test_img = trio_img[image_sizeH:2*image_sizeH,:]/1000/42.42
    #test_img = np.tile(test_img[:,:,np.newaxis],(1,1,3))

    label_img = trio_img[2*image_sizeH:,:]
    label_img = (label_img>0).astype(int)
    
    reference_PIL   = np.tile(reference_img[:,:,np.newaxis],(1,1,3))
    import torchvision
    t   = torchvision.transforms.ToPILImage()
    reference_PIL=t(reference_PIL)

    for i in range(0,8):
        # Populate training dictionary with tupple (reference,test,label)
        training_set[idx2] = reference_img[:,i*128:(i+1)*128], test_img[:,i*128:(i+1)*128], label_img[:,i*128:(i+1)*128]   
        idx2+=1
    
print('Saving Pickle Training Set')
with open(train_pickle_file, 'wb') as handle:
    pickle.dump(training_set, handle, protocol=4)
print('Saved.')

