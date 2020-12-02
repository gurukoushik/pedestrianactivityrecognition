This folder includes all the file needed to convert JAAD data to YOWO compatible format

To create the files follows these step:

1. Clone JAAD github 

```
$ git clone https://github.com/ykotseruba/JAAD.git
```
2. Copy `CreateGroundTruthsDatabaseForYOWO.py` to `JAAD` folder

```
$ cp ./pedestrianactivityrecognition/JAAD_to_YOWO/CreateGroundTruthsDatabaseForYOWO.py -d ./JAAD/
```
3. Run CreateGroundTruthsDatabaseForYOWO.py 
this command will generate groundthruth for train, test, and validation set in JAAD. 
