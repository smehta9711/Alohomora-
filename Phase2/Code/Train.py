#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
# import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model, ResNet18, ResNeXt, DenseNet
from Misc.MiscUtils import *
from Misc.DataUtils import *
from Test import ConfusionMatrix

from customDataset import CustomImageDataset
from torch.utils.data import DataLoader

# Don't generate pyc codes
sys.dont_write_bytecode = True

def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Generate a batch of images and labels with proper gradient tracking
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        ImageNum += 1
        
        # Get image and label
        I1, Label = TrainSet[RandIdx]
        
        # Convert image to float and enable gradients
        if isinstance(I1, torch.Tensor):
            I1 = I1.float()
        else:
            I1 = torch.tensor(I1, dtype=torch.float32)
        
        # Convert label to long tensor
        if isinstance(Label, torch.Tensor):
            Label = Label.long()
        else:
            Label = torch.tensor(Label, dtype=torch.long)

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(Label)
    
    # Stack batches
    I1Batch = torch.stack(I1Batch)
    LabelBatch = torch.stack(LabelBatch)
    
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath, TestSet=None):
    """
    Training operation with memory optimization
    """
    # Initialize the model
    model = CIFAR10Model() 
    # model = ResNet18()
    # model = ResNeXt()
    # model = DenseNet()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer with reduced learning rate
    Optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.1)  #Added decaying learning rate to increase accuracy      

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=NumEpochs,eta_min=1e-6) #Added decaying learning rate to increase accuracy

    # Tensorboard
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    
    train_epoch_loss = []
    train_epoch_accs = []
    test_epoch_accs = []

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        train_losses = []
        train_accs = []
        
        model.train()
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Clear GPU cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            
            # Move batch to device
            Batch = [b.to(device) for b in Batch]
            
            # Forward pass
            results = model.validation_step(Batch)
            LossThisBatch = results['loss']
            BatchAccuracy = results['acc']

            # Backward pass
            Optimizer.zero_grad()
            LossThisBatch.backward()
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            Optimizer.step()

            # Record metrics
            train_losses.append(LossThisBatch.item())
            train_accs.append(BatchAccuracy.item())
            
            # Checkpointing
            if PerEpochCounter % SaveCheckPoint == 0:
                SaveName = CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                torch.save({
                    'epoch': Epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': Optimizer.state_dict(),
                    'loss': LossThisBatch
                }, SaveName)
                print('\n' + SaveName + ' Model Saved...')

            # # Tensorboard logging
            # Writer.add_scalar('LossEveryIter', LossThisBatch.item(), Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # Writer.add_scalar('Accuracy', BatchAccuracy.item(), Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # Writer.flush()

        # Epoch metrics
        epoch_loss = np.mean(train_losses)
        epoch_acc = np.mean(train_accs)
        
        # Update learning rate
        scheduler.step()

        train_epoch_loss.append(epoch_loss)
        train_epoch_accs.append(epoch_acc)

        Writer.add_scalar('Epoch/TrainLoss', epoch_loss, Epochs)
        Writer.add_scalar('Epoch/TrainAccuracy', epoch_acc, Epochs)
        Writer.flush()

        print(f"\nEpoch {Epochs+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # Save epoch model
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({
            'epoch': Epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            'loss': epoch_loss
        }, SaveName)

        # Testing if test set provided
        if TestSet:
            test_accs = []
            LabelsTrue = []  # New: Collect true labels
            LabelsPred = []  # New: Collect predicted labels
            model.eval()
            with torch.no_grad():
                for _ in range(len(TestSet) // MiniBatchSize):
                    TestBatch = GenerateBatch(TestSet, None, ImageSize, MiniBatchSize)
                    TestBatch = [b.to(device) for b in TestBatch]
                    images, labels = TestBatch
                    
                    # New: Collect predictions and labels
                    preds = torch.argmax(model(images), dim=1)
                    LabelsTrue.extend(labels.cpu().numpy())
                    LabelsPred.extend(preds.cpu().numpy())
                    
                    # Evaluate batch accuracy
                    results = model.validation_step(TestBatch)
                    test_accs.append(results['acc'].item())
            
            # Calculate and log test accuracy
            test_acc = np.mean(test_accs)
            test_epoch_accs.append(test_acc)
            print(f"Test Accuracy: {test_acc:.4f}")
            Writer.add_scalar('Epoch/TestAccuracy', test_acc, Epochs)
            
            # New: Generate confusion matrix
            print("\nConfusion Matrix:")
            ConfusionMatrix(LabelsTrue, LabelsPred)

    
    PlotMetrics(train_epoch_loss, train_epoch_accs, test_acc=test_epoch_accs)
    # ConfusionMatrix(model, TrainSet, title="Training Set")
    
    return train_epoch_loss, train_epoch_accs, test_epoch_accs

def PlotMetrics(train_loss, train_acc, test_acc=None):
    epochs = range(1, len(train_loss) + 1)

    # Plot loss
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, train_acc, label='Train Accuracy')
    if test_acc:
        plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=20, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:16')
    Parser.add_argument('--MiniBatchSize', type=int, default=100, help='Size of the MiniBatch to use, Default:50')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    # TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                     download=True, transform=ToTensor())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    transform = transforms.Compose([
    # transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),         # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])


    train_img_dir_path = "/home/sarthak_m/ComputerVision/HW0_Alohomora/YourDirectoryID_hw0/Phase2/CIFAR10/Train"
    train_annotations_file_path = "/home/sarthak_m/ComputerVision/HW0_Alohomora/YourDirectoryID_hw0/Phase2/Code/TxtFiles/LabelsTrain.txt"

    train_dataset = CustomImageDataset(annotations_file=train_annotations_file_path, img_dir=train_img_dir_path, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size= MiniBatchSize, shuffle=False)

    # print(train_dataset.img_labels)
    # print(train_dataset)

    test_img_dir_path = "/home/sarthak_m/ComputerVision/HW0_Alohomora/YourDirectoryID_hw0/Phase2/CIFAR10/Test"
    test_annotations_file_path ="/home/sarthak_m/ComputerVision/HW0_Alohomora/YourDirectoryID_hw0/Phase2/Code/TxtFiles/LabelsTest.txt"

    test_dataset = CustomImageDataset(annotations_file=test_annotations_file_path, img_dir=test_img_dir_path, transform=transform)
    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath= "./data/", CheckPointPath= CheckPointPath)

    ImageSize = [32, 32, 3]
    # NumTrainSamples = len(train_dataset)
    NumTrainSamples = len(train_dataset)

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(base_path= "../data/",CheckPointPath= CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, train_dataset, LogsPath, TestSet=test_dataset)

    
if __name__ == '__main__':
    main()
 
