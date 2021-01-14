import os
from pathlib import Path
import random

rootdir = Path('/home/lorant/Projects/data/cikk2/combined/train/')

traintxt = 'trainCD.txt'
testtxt  = 'valCD.txt'

#subdirs = os.listdir(rootdir)
#subdirs.sort()

with open(traintxt, 'w') as tr, open(testtxt, 'w') as vl:

    #for actfolder in subdirs:
    #    print(actfolder)
        #images = os.listdir(rootdir/actfolder/'GT')
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