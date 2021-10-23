# RDNet
The project was implemented aims to realize multi-types rock detection and classification, names RDNet. And the method was optimized based on YOLO-V3, the paper could be download from "https://arxiv.org/abs/1804.02767", and some tricks learned from it.
# Contents
- [1.Environment](#environment)

- [2.Preparation](#preparation)

- [3.Training](#training)

- [4.Demo](#demo)
# Environment
    - Driver: NVIDIA-Linux-x86_64-470.63.01.run
    - CUDA: cuda_11.4.2_470.57.02_linux.run
    - Torch 1.7.0+
    - python 3.8
# Preparation
1. Get the code. We will call the cloned directory as `$RDNet`.  
2. Prepare the dataset, we spyder the data from [National Infrastructure of Mineral, Rock and Fossil for Science and Technology.](http://www.nimrf.net.cn/en/english)
3. Prepare the data basic structure, the .txt file include `cls`, `center_x`, `center_y`, `w`, `h`, and then transform all labeled txt files to COCO format saved as json file.
        
        $Data/                           # RootPath  
        
        $Data/JPEGImages                 # include images.   
        
        $Data/Annotations                # include .txt files. 
        
        $Data/*.json                     # include .json files. 
# Training
1. Train your model on PASCAL VOC Format.
                
        cd $RDNet
        python3 train.py
                
2. Train results, it will create '.pth' model, loss log file and evaluation log file.
                
        # It will create model definition files and save snapshot models in:
        #   - $RDNet/weights/'{}_{}.pth'
        # the loss log and evaluation log saved in:
        #   - $RDNet/log/'{}'/loss.txt'.format(lstrftime('%b%d-%H'))
        #   - $RDNet/log/'{}'/result.txt'.format(lstrftime('%b%d-%H'))
# Demo
1. Visualization.  
                
        cd $RDNet
        python3 demo.py
![aug_00315](https://user-images.githubusercontent.com/33689425/138543133-28440cff-1d91-4e3e-a207-4a05097ad6cb.jpg)

        
        
