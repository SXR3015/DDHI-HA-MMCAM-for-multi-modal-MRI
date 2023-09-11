# dual-domain-cross-attention-hierarchical-alignments-mmscorecam-for-multi-modal-MRI
A framework of hierarchical alignments, dual-domain cross-attention,mmscorecam

## __Environment__  
torch ==1.13.1  
numpy == 1.22.3  
nibabel == 1.10.2  
torchcam == 0.3.2  
torchvision == 0.14.1  

## extract the imaging features
To run the model, you need to extract the sfc, dfc, and alff by matlab. You can use the batch opearation of spm12 to finish this. Then, you need to use panda to gerenate FA. But you can also use FSL instead. In data file, we list some expample files. 

## run the model

### __Create k fold csv file__  
generate_csv.py
### train and validate the model 
train.py
### test the model 
test.py

## generate the avtivation map in multi-modal MRI  
### replace the py files in initial torchcam with the files in torcham filefolder in this respority, including
activation.py
core.py
### run test.py
test.py
