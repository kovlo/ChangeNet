import os
from pathlib import Path
import random

train_file_list = './trainCD.txt'
val_file_list =  './valCD.txt'
test_file_list = './testCD.txt'

## Train Val txt
rootdir = Path('/home/lorant/Projects/data/ChangeNet/combined/train/')

with open(train_file_list, 'w') as tr, open(val_file_list, 'w') as vl:

    images = os.listdir(rootdir)
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

## Test txt

rootdir = Path('/home/lorant/Projects/data/ChangeNet/combined/test/')

with open(test_file_list, 'w') as tst:
   
    images = os.listdir(rootdir)
    images.sort()
    print(len(images))

    for gtimname in images:
        f=tst

        savedirpath = Path(rootdir.parent.stem)/rootdir.stem
        a=str(savedirpath/gtimname)
        f.write(a+"\n")
print('Test done!')