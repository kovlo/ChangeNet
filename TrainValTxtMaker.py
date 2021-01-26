import os
from pathlib import Path
import random


traintxt = 'trainCD.txt'
valtxt  = 'valCD.txt'
## Train Val txt
rootdir = Path('/home/lorant/Projects/data/ChangeNet/combined/train/')

with open(traintxt, 'w') as tr, open(valtxt, 'w') as vl:

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
print('Done!')

## Test txt
testtxt = 'testCD.txt'

rootdir = Path('/home/lorant/Projects/data/ChangeNet/combined/test/')

with open(testtxt, 'w') as tst:
   
    images = os.listdir(rootdir)
    images.sort()
    print(len(images))

    for gtimname in images:
        f=tst

        savedirpath = Path(rootdir.parent.stem)/rootdir.stem
        a=str(savedirpath/gtimname)
        f.write(a+"\n")
print('Done!')