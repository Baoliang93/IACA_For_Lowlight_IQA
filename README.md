# IACA_For_Lowlight_IQA
Code for "Gap-closing Matters: Perceptual Quality Evaluation and  Optimization of Low-Light Image Enhancement"  
  
![image](datasets/SQUARE-LOL/flow.pdf)


# Environment
* Python 3.6.7
* Pytorch 1.6.0  Cuda V9.0.176 Cudnn 7.4.1

# Running
* Download the SQUARE-LOL dataset from [MEGA](https://pan.baidu.com/s/1pyl5Yz4opPdoACnqSWLXsw), and put all the downloaded files in the path: "./datasets/SQUARE-LOL/".
* Download the pre-trained models from [MEGA](https://pan.baidu.com/s/1pyl5Yz4opPdoACnqSWLXsw), and put all the downloaded files in the path: "./codes/TMM2023-IACA-Release/checkpoints/".

* Train:  
  `python  ./codes/TMM2023-IACA-Release/IQA_Main.py`

* Test:  
  `python   ./codes/TMM2023-IACA-Release/IQA_Test.py`
    
* Demo:    
   `python   ./codes/TMM2023-IACA-Release/Demo.py`
   
# Details
* Apex install:
```
  git clone https://github.com/NVIDIA/apex
  cd apex
  python3 setup.py install
```
* Waiting

