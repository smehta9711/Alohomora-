
## Phase 1: Shake My boundary

### Steps to run the code
To run the PBLite boundary detection, use the following command:
- Open 'smehta1_hw0/Phase1/Code' and run the command given below.

python3 Wrapper.py



## Phase 2: Deep Dive on Deep Learning

### Steps to run the code:
### Training:
 For training the model, open 'smehta1_hw0/Phase2/Code/Train.py' and uncomment the model which we want to train.
 Below is an example which shows changes for training on First Neural Network (which is named CIFAR10Model in my code). Also, remember to make similar change in 'Test.py' for testing.
```Python
    model = CIFAR10Model()
    # model = ResNet18()
    # model = ResNeXt()
    # model = DenseNet()

```
 Now, run the command given below from path 'smehta1_hw0/Phase2/Code'.


python3 Train.py 


The file test.py was not used in this assignment explicitly. The file Train.py function involves training and testing functionality.