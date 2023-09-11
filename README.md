# dual-domain-cross-attention-hierarchical-alignments-mmscorecam-for-multi-modal-MRI
A framework of hierarchical alignments, dual-domain cross-attention

The environment is  
torch ==1.13.1  
numpy == 1.22.3  
nibabel == 1.10.2  
torchcam == 0.3.2  
torchvision == 0.14.1  

To run the model, you need to extract the sfc, dfc, and alff by matlab. Then, you need to use panda to gerenate FA. In data file, we list some expample files.
Create k fold csv file  
generate_csv.py
