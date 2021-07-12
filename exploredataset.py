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
from PIL import ImageMath   

def explore():
    train_file_list = './trainCD.txt'
    val_file_list =  './valCD.txt'
    test_file_list = './testCD.txt'

    base_data_dir = ''
    train_pickle_file = './change_dataset_trainCD.pkl'
    val_pickle_file = './change_dataset_valCD.pkl'
    test_pickle_file = './change_dataset_testCD.pkl'


    validation_set = {}
    training_set = {}
    test_set = {}

    image_size  = 256
    image_sizeH = 128
    image_sizeW = 1024

    # #### Parse the path files

    # %%
    #train_file_list = [line.rstrip('\n').split() for line in open(train_file_list)]
    #val_file_list = [line.rstrip('\n').split() for line in open(val_file_list)]
    test_file_list = [line.rstrip('\n').split() for line in open(test_file_list)]
    #print('Length Training Set:', len(train_file_list))
    #print('Length Validation Set:', len(val_file_list))
    print('Length Test Set:', len(test_file_list))
    #size_train = len(train_file_list)
    #size_validation = len(val_file_list)
    size_test = len(test_file_list)

    def ReadImage(FileName):
        # Load the reference, test and label images
        trio_pil = Image.open(FileName).convert('I;16')

        #reference_img = (trio_img[0:image_sizeH,:]/1000/42.42)
        #reference_img = np.tile(reference_img[:,:,np.newaxis],(1,1,3))
        reference_pil = trio_pil.crop(box=(0,0*image_sizeH,image_sizeW,1*image_sizeH)).convert('F')

        #test_img = trio_img[image_sizeH:2*image_sizeH,:]/1000/42.42
        #test_img = np.tile(test_img[:,:,np.newaxis],(1,1,3))
        test_pil = trio_pil.crop(box=(0,1*image_sizeH,image_sizeW,2*image_sizeH)).convert('F')

        label_pil = trio_pil.crop(box=(0,2*image_sizeH,image_sizeW,3*image_sizeH)).convert("L")

        reference_pil = ImageMath.eval("(x/1000/42.42*255)",x=reference_pil).convert('L')
        test_pil = ImageMath.eval("(x/1000/42.42*255)",x=test_pil).convert('L')
        label_pil = Image.eval(label_pil,lambda x: 1 if x>0 else 0)

        return reference_pil,test_pil,label_pil

    """
    # #### Load Validation Set On Memory
    # %%
    idx2 = 0
    plt.figure
    for idx, entry in enumerate(tqdm(val_file_list[:2000])):
        reference_pil,test_pil,label_pil= ReadImage(base_data_dir + entry[0])

        # Populate validation dictionary with tupple (reference,test,label)
        for i in range(0,8):
            # Populate training dictionary with tupple (reference,test,label)
            reference_pilPart = reference_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))
            test_pilPart=test_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))
            label_pilPart=label_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))

            validation_set[idx2] = reference_pilPart,test_pilPart,label_pilPart   
            reference_pilPart,test_pilPart,label_pilPart=0,0,0
            idx2+=1

    print('Saving Pickle Validation Set')
    with open(val_pickle_file, 'wb') as handle:
        pickle.dump(validation_set, handle, protocol=4)
    validation_set={}
    print('Done')

    # #### Load Training Set On Memory
    idx2 = 0
    for idx, entry in enumerate(tqdm(train_file_list[:10000])):
        reference_pil,test_pil,label_pil= ReadImage(base_data_dir + entry[0])

        # Populate validation dictionary with tupple (reference,test,label)
        for i in range(0,8):
            # Populate training dictionary with tupple (reference,test,label)
            reference_pilPart = reference_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))
            test_pilPart=test_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))
            label_pilPart=label_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))

            training_set[idx2] = reference_pilPart,test_pilPart,label_pilPart   
            idx2+=1
    
    print('Saving Pickle Training Set')
    with open(train_pickle_file, 'wb') as handle:
        pickle.dump(training_set, handle, protocol=4)
    print('Saved.')
    training_set={}
    """
    # #### Load Test Set On Memory
    idx2 = 0
    for idx, entry in enumerate(tqdm(test_file_list)):
        reference_pil,test_pil,label_pil= ReadImage(base_data_dir + entry[0])

        # Populate validation dictionary with tupple (reference,test,label)
        for i in range(0,8):
            # Populate training dictionary with tupple (reference,test,label)
            reference_pilPart = reference_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))
            test_pilPart=test_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))
            label_pilPart=label_pil.crop(box=(i*image_sizeH,0,image_sizeH*(i+1),image_sizeH))

            test_set[idx2] = reference_pilPart,test_pilPart,label_pilPart  
            """
            plt.subplot(3,8,i+1)
            plt.imshow(reference_pilPart)
            plt.subplot(3,8,i+1+8)
            plt.imshow(test_pilPart)
            plt.subplot(3,8,i+1+8+8)
            plt.imshow(label_pilPart)
            """
            idx2+=1
        #plt.show()
    print('Saving Pickle Training Set')
    with open(test_pickle_file, 'wb') as handle:
        pickle.dump(test_set, handle, protocol=4)
    print('Saved.')
    test_set={}
