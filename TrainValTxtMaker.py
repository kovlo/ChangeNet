import os
from pathlib import Path
import random
import glob

train_file_list = './trainCD.txt'
val_file_list =  './valCD.txt'
test_file_list = './testCD.txt'
"""
## Train Val txt
rootdir = Path('/home/lorant/Projects/data/Change3D/Change3D_dyn/')

with open(train_file_list, 'w') as tr, open(val_file_list, 'w') as vl:

    #images = os.listdir(rootdir)
    images = glob.glob(str(rootdir)+'/*.png')
    images.sort()
    print(len(images))

    for gtimname in images:
        if random.random()<0.8:
            f=tr
        else:
            f=vl

        savedirpath = Path(rootdir.parent.stem)/rootdir.stem
        a=str(savedirpath/gtimname)
        f.write(a+"\n")
print('Train Val done!')
"""
## Test txt
def createTxt(rootdir):

    #images = os.listdir(rootdir)
    images = glob.glob(str(rootdir)+'/*.png')
    images.sort()
    print(len(images))
    
    with open(test_file_list, 'w') as tst:

        for gtimname in images:
            f=tst

            savedirpath = Path(rootdir.parent.stem)/rootdir.stem
            a=str(savedirpath/gtimname)
            f.write(a+"\n")
    print('Test done!')

#rootdir = Path('/home/lorant/Projects/data/Change3D/Change3D_stat_4class_cut_final/imgs_multi_dyn/')
#createTxt(rootdir)