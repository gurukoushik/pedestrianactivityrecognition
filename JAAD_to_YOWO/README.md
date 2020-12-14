This folder includes all the file needed to convert JAAD data to YOWO compatible format

To create the files follows these step:

1. Clone JAAD github 

```
$ git clone https://github.com/ykotseruba/JAAD.git
```

if you want to crate multiple class use the following commands
2. Copy `CreateGroundTruthsDatabaseForYOWO.py` to `JAAD` folder

```
$ cp ./pedestrianactivityrecognition/JAAD_to_YOWO/CreateGroundTruthsDatabaseForYOWOMultipleCLASS.py -d ./JAAD/
```
3. Run CreateGroundTruthsDatabaseForYOWOMultipleCLASS.py 
this command will generate groundthruth for train, test, and validation set in JAAD. 


the following 2 step are wrote for creating crossing and not crossing groundtruths and reduce the bbox by 4 only

2. Copy `CreateGroundTruthsDatabaseForYOWO_2classes.py` to `JAAD` folder

```
$ cp ./pedestrianactivityrecognition/JAAD_to_YOWO/CreateGroundTruthsDatabaseForYOWO_2classes.py -d ./JAAD/
```
3. Run CreateGroundTruthsDatabaseForYOWO_2classes.py 
this command will generate groundthruth for train, test, and validation set in JAAD. 
