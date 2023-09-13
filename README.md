# DDCA-HA-mmscorecam-for-multi-modal-MRI
A framework of hierarchical alignments, dual-domain cross-attention,mmscorecam

## __Environment__  
torch ==1.13.1  
numpy == 1.22.3  
nibabel == 1.10.2  
torchcam == 0.3.2  
torchvision == 0.14.1  
einops == 0.6.0  
python == 3.9.0  
imageio == 2.31.1  
## extract the imaging features
To run the model, you need to extract the sfc, dfc, and alff by Matlab. You can use the batch operation of spm12 to finish this. Then, you need to use Panda to generate FA. But you can also use FSL instead. Additionally, the input data shape might influence the kernel size of avgpooling, you need to change the kernel size, if has bugs.  

## run the model

### __Create k fold csv file__  
generate_csv.py
### train and validate the model 
train.py
### test the model 
test.py

## generate the avtivation map in multi-modal MRI  
### replace the py files in initial torchcam package with the files in $\color[rgb]{1,0,1} method$ folder in this respority, including

activation.py  
core.py
### run test.py  
test.py
### results
see the images folder
