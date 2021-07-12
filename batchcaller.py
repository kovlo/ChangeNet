import glob
import pathlib as Path
from TrainValTxtMaker import *
from exploredataset import *
from test import *


SourceFolderRoot = "/media/lorant/DATA_TRANSFER/test_roboustness2/*/"
FolderList = glob.glob(str(SourceFolderRoot))
FolderList.sort()

for testFolder in FolderList:
    if len(testFolder.split('-'))>1:
        continue
    
    testFolderPath=Path(testFolder)
    testFolderPath= testFolderPath/"set"

    outFolderPath = Path(testFolder)
    outFolderPath = outFolderPath/(outFolderPath.stem+"-ChangeNet")
    if outFolderPath.exists():
        continue
    print(testFolder)
    
    createTxt(testFolderPath)
    explore()
    runtest(str(outFolderPath)+"/")
