import os
from pathlib import Path
import random

rootdir = Path('/home/lorant/Projects/data/cikk2/combined/train/')

traintxt = 'trainCD.txt'
testtxt  = 'valCD.txt'
valtxt  = 'valCD.txt'

with open(traintxt, 'w') as tr, open(testtxt, 'w') as vl:

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
rootdir = Path('/home/lorant/Projects/data/cikk2/combined/test/')


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