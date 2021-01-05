import os
from pathlib import Path

rootdir = Path('/home/lorant/Projects/data/cikk2/VL-CMU-CD-dataset/raw/')

traintxt = 'train1.txt'
testtxt  = 'val1.txt'

subdirs = os.listdir(rootdir)
subdirs.sort()

with open(traintxt, 'w') as tr, open(testtxt, 'w') as vl:

    for actfolder in subdirs:
        print(actfolder)
        images = os.listdir(rootdir/actfolder/'GT')
        images.sort()
        print(len(images))
        if len(images)>5:            
            f=tr
        else:
            f=vl

        for gtimname in images:
            imID = gtimname[2:4]
            savedirpath = Path(rootdir.parent.stem)/rootdir.stem
            a=str(savedirpath/actfolder/'RGB'/('2_'+imID+'.png'))
            b=str(savedirpath/actfolder/'RGB'/('1_'+imID+'.png'))
            c=str(savedirpath/actfolder/'GT'/gtimname)
            f.write(a+" "+b+" "+c+"\n")
print('Done!')