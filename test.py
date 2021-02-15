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
import time

import os.path
import cv2

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
#num_gpu = torch.cuda.device_count()
num_gpu = 1
print('Number of GPUs Available:', num_gpu)

num_classes = 2
img_size = 224

train_pickle_file = './change_dataset_trainCD.pkl'
val_pickle_file = './change_dataset_valCD.pkl'
test_pickle_file = './change_dataset_testCD.pkl'

checkpointname = './best_model'+str(num_classes)+'CD.pkl'

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(img_size),
        #transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Create training and validation datasets
train_dataset = change_dataset_np.ChangeDatasetNumpy(train_pickle_file, data_transforms['val'])
val_dataset  = change_dataset_np.ChangeDatasetNumpy(val_pickle_file, data_transforms['val'])
test_dataset = change_dataset_np.ChangeDatasetNumpy(test_pickle_file, data_transforms['val'])

change_net = models.ChangeNet(num_classes=num_classes)
change_net = change_net.to(device)

if os.path.exists((checkpointname)):
    checkpoint = torch.load(checkpointname)
    change_net.load_state_dict(checkpoint);
    print('Checkpoint '+checkpointname+' is loaded.')

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def explore_validation_dataset(idx, inv):
    change_net.eval()

    
    referenceimg = np.zeros((img_size,img_size*8,3))
    testimg=np.zeros((img_size,img_size*8,3))
    labelimg=np.zeros((img_size,img_size*8))
    outputimg=np.zeros((img_size,img_size*8))
    for i in range(0,8):
        actidx = idx*8+i
        print(actidx)
        sample = dataset[actidx]
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
        
        pred = change_net([reference, test])
        
        #print(pred.shape)
        _, output = torch.max(pred, 1)
        output = output.squeeze(0).cpu().numpy()

        referenceimg[:,i*img_size:(i+1)*img_size,:]=reference_img
        testimg[:,i*img_size:(i+1)*img_size,:]=test_img
        outputimg[:,i*img_size:(i+1)*img_size]=output
        labelimg[:,i*img_size:(i+1)*img_size]=label

    vline = np.ones((img_size,1))
    hline = np.ones((1,img_size*8*2+1))
    outimg = np.hstack((referenceimg[:,:,0],testimg[:,:,0]))
    outimg= np.vstack((outimg,np.hstack((labelimg,outputimg))))

    image_sizeH = 128
    image_sizeW = 1024
    referenceimgRES = cv2.resize(referenceimg[:,:,0],(image_sizeW,image_sizeH))
    testimgRES = cv2.resize(testimg[:,:,0],(image_sizeW,image_sizeH))
    labelimgRES = cv2.resize(labelimg,(image_sizeW,image_sizeH))
    outputimgRES = cv2.resize(outputimg,(image_sizeW,image_sizeH))
    
    evaloutImg = np.hstack((referenceimgRES,testimgRES))
    evaloutImg= np.vstack((evaloutImg,np.hstack((labelimgRES,outputimgRES))))
    cv2.imwrite(outdir+"change_net_"+str(idx).zfill(4)+".png",evaloutImg*255)


dataset = val_dataset
#dataset = train_dataset
dataset = test_dataset
outdir = "./trainoutput/"
check_dir(outdir)
starttime = time.time()
for idx in range (0,min(1000,int(len(dataset)/8))):
    explore_validation_dataset(idx, False)

endtime = time.time()
avgtime = (endtime-starttime)/len(dataset)/8
with open((outdir+"avgtime.time"), 'w') as f2:
    f2.write(str(avgtime))
    f2.write("\n")
print("Avg execution time: "+str(avgtime))