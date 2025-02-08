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

import torch.nn as nn
import torch.nn.functional as F
import torch

def accuracy(outputs, labels):
    """Calculate accuracy for a batch"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    #################################################
        #Fill your loss function of choice here!
    #################################################
    criterion = nn.CrossEntropyLoss()
    return criterion(out, labels)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        # Ensure inputs have proper types and gradients
        images = images.float().requires_grad_()
        labels = labels.long()
        
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels)         # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        # Ensure inputs have proper types and gradients
        images = images.float()
        labels = labels.long()
        
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)           # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss, 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))


class CIFAR10Model(ImageClassificationBase):
    def __init__(self, InputSize=32, OutputSize=10):
        """
        Inputs: 
        InputSize - Size of the Input features
        OutputSize - Number of classes (10 for CIFAR-10)
        """
        super().__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(3, InputSize, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(InputSize)                                                 #Added Batch Normalization for increasing accuracy
        self.conv2 = nn.Conv2d(InputSize, 2*InputSize, kernel_size=3, stride=1, padding=1)   
        self.bn2 = nn.BatchNorm2d(2*InputSize)                                               #Added Batch Normalization for increasing accuracy
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear((2*InputSize) * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)                                                        #Increased the layers for improved model
        self.fc3 = nn.Linear(64, 32)                                                         #Increased the layers for improved model
        self.fc4 = nn.Linear(32, OutputSize)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)                                                     #Added dropout for better generalisation and avoid overfitting
    
    def forward(self, x):
        """
        Input:
        x - Input tensor of shape (batch_size, 3, 32, 32)
        
        Output:
        x - Output tensor of shape (batch_size, num_classes)
        """

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    

class ResNet18(ImageClassificationBase):
    def __init__(self, InputSize=32, OutputSize=10):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.shortcut1 = nn.Sequential()
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # self.flatten = nn.Flatten()

        self.fc = nn.Linear(128, OutputSize)
    
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        idx_1 = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += idx_1

        x = F.relu(x)

        idx_2 = self.shortcut2(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        x += idx_2

        x = F.relu(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNeXt(ImageClassificationBase):

    def __init__(self, InputSize=32, OutputSize=10, cardinality=32):
        """
        Inputs: 
        InputSize - Size of the Input features
        OutputSize - Number of classes (10 for CIFAR-10)
        cardinality - Number of parallel paths
        """
        super().__init__()
        
        # First layer
        self.conv1 = nn.Conv2d(3, InputSize, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(InputSize)
        
        # First ResNeXt block - calculate widths carefully
        width = InputSize  # Width for each group
        self.resnext1_conv1 = nn.Conv2d(InputSize, width, kernel_size=1)
        self.resnext1_bn1 = nn.BatchNorm2d(width)
        self.resnext1_conv2 = nn.Conv2d(width, width, kernel_size=3, 
                                       padding=1, groups=cardinality)
        self.resnext1_bn2 = nn.BatchNorm2d(width)
        self.resnext1_conv3 = nn.Conv2d(width, width*2, kernel_size=1)
        self.resnext1_bn3 = nn.BatchNorm2d(width*2)
        
        # Shortcut for first block
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(InputSize, width*2, kernel_size=1),
            nn.BatchNorm2d(width*2)
        )
        
        # Second ResNeXt block
        self.resnext2_conv1 = nn.Conv2d(width*2, width*2, kernel_size=1)
        self.resnext2_bn1 = nn.BatchNorm2d(width*2)
        self.resnext2_conv2 = nn.Conv2d(width*2, width*2, kernel_size=3, 
                                       padding=1, groups=cardinality)
        self.resnext2_bn2 = nn.BatchNorm2d(width*2)
        self.resnext2_conv3 = nn.Conv2d(width*2, width*4, kernel_size=1)
        self.resnext2_bn3 = nn.BatchNorm2d(width*4)
        
        # Shortcut for second block
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(width*2, width*4, kernel_size=1),
            nn.BatchNorm2d(width*4)
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input features for first FC layer
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        fc_input_size = width*4 * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, OutputSize)
        
        # Dropout for regularization
        # self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        """
        Input:
        x - Input tensor of shape (batch_size, 3, 32, 32)
        Output:
        x - Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))  # 32x32
        x = self.pool(x)                      # 16x16
        
        # First ResNeXt block
        identity1 = x
        x = F.relu(self.resnext1_bn1(self.resnext1_conv1(x)))
        x = F.relu(self.resnext1_bn2(self.resnext1_conv2(x)))
        x = self.resnext1_bn3(self.resnext1_conv3(x))
        x += self.shortcut1(identity1)
        x = F.relu(x)
        x = self.pool(x)                      # 8x8
        
        # Second ResNeXt block
        identity2 = x
        x = F.relu(self.resnext2_bn1(self.resnext2_conv1(x)))
        x = F.relu(self.resnext2_bn2(self.resnext2_conv2(x)))
        x = self.resnext2_bn3(self.resnext2_conv3(x))
        x += self.shortcut2(identity2)
        x = F.relu(x)
        x = self.pool(x)                      # 4x4
        
        # Fully connected layers
        x = torch.flatten(x, 1)
        # x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4):
        """
        Dense Layer with bottleneck.
        Args:
            num_input_features: Number of input channels
            growth_rate: How many features to add
            bn_size: Bottleneck size multiplier
        """
        super().__init__()
        
        # First BN-ReLU-Conv(1x1)
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 
                              kernel_size=1, stride=1, bias=False)
        
        # Second BN-ReLU-Conv(3x3)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # First bottleneck layer
        new_features = F.relu(self.bn1(x), inplace=True)
        new_features = self.conv1(new_features)
        
        # Second conv layer
        new_features = F.relu(self.bn2(new_features), inplace=True)
        new_features = self.conv2(new_features)
        
        # Concatenate input with new features
        return torch.cat([x, new_features], 1)

class DenseNet(ImageClassificationBase):
    def __init__(self, InputSize=32, OutputSize=10, growth_rate=12, block_config=[6, 12, 24]):
        """
        DenseNet Model for CIFAR-10.
        Args:
            InputSize: Initial feature size
            OutputSize: Number of classes
            growth_rate: Features added by each layer
            block_config: Number of layers in each dense block
        """
        super().__init__()
        
        # Initial number of features
        num_features = 2 * growth_rate

        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block 1
        self.dense1 = nn.ModuleList()
        for i in range(block_config[0]):
            self.dense1.append(DenseLayer(num_features, growth_rate))
            num_features += growth_rate
            
        # Transition 1
        self.trans1_bn = nn.BatchNorm2d(num_features)
        self.trans1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block 2
        self.dense2 = nn.ModuleList()
        for i in range(block_config[1]):
            self.dense2.append(DenseLayer(num_features, growth_rate))
            num_features += growth_rate
            
        # Transition 2
        self.trans2_bn = nn.BatchNorm2d(num_features)
        self.trans2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dense Block 3
        self.dense3 = nn.ModuleList()
        for i in range(block_config[2]):
            self.dense3.append(DenseLayer(num_features, growth_rate))
            num_features += growth_rate
        
        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_features)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, OutputSize)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution
        out = self.conv1(x)               # 32x32
        out = F.relu(self.bn1(out))
        out = self.maxpool(out)           # 16x16
        
        # Dense Block 1
        for layer in self.dense1:
            out = layer(out)
        out = F.relu(self.trans1_bn(out))
        out = self.trans1_pool(out)       # 8x8
        
        # Dense Block 2
        for layer in self.dense2:
            out = layer(out)
        out = F.relu(self.trans2_bn(out))
        out = self.trans2_pool(out)       # 4x4
        
        # Dense Block 3
        for layer in self.dense3:
            out = layer(out)
        out = F.relu(self.final_bn(out))
        
        # Classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out