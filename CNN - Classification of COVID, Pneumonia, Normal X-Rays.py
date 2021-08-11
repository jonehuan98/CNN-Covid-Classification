import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 100
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imageDataLoader(directory:str, train=True, singleChannel=False):
    covidData = directory + "\\Covid"
    normalData = directory + "\\Normal"
    pneumoniaData = directory + "\\Viral Pneumonia"
    dataLabels =  {covidData:0, normalData:1, pneumoniaData:2}
    dataset = []

    covidCount = 0
    normalCount = 0
    pneumoniaCount = 0

    for label in dataLabels:
        for f in os.listdir(label):
            path = os.path.join(label, f)
            img = cv2.imread(path)            
            img = np.array(img)
            img = np.moveaxis(img,-1,0)
            dataset.append([img, dataLabels[label]])

            if label == covidData:
                covidCount += 1
            elif label == normalData:
                normalCount += 1
            elif label == pneumoniaData:
                pneumoniaCount += 1
            
    print("Number of samples, train = ", train)
    print("Covid Samples: ", covidCount)
    print("Normal Samples: ", normalCount)
    print("Pneumonia Samples: ", pneumoniaCount)

    np.random.shuffle(dataset)
    
    if train:
        validationSplit = 0.2
        validationSize = int(len(dataset)*validationSplit)
        validationData = dataset[-validationSize:]
        trainingData = dataset[:-validationSize]
        loadedTrainingData = DataLoader(dataset = trainingData, batch_size= 2, shuffle=False)
        loadedValidationData = DataLoader(dataset = validationData, batch_size= 2, shuffle=False)
        return loadedTrainingData, loadedValidationData
    
    else:
        loadedTestData = DataLoader(dataset= dataset, batch_size =2, shuffle=False)
        return loadedTestData

def plotLosses(trainHistory, valHistory, title):
    x = np.arange(1, len(trainHistory) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(x, trainHistory, color="blue", label="Training loss", linewidth=2)
    plt.plot(x, valHistory, color="green", label="Validation loss", linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of training and validation loss: " + str(title))
    plt.show()

class SimpleConvolutionalNetwork(nn.Module):
    def __init__(self, imgSize, extraLayer = False, singleChannel = False):
        super(SimpleConvolutionalNetwork,self).__init__()
        self.imgSize = imgSize
        self.extraLayer = extraLayer
        self.singleChannel = singleChannel

        if singleChannel:
            self.conv1 = nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1) # size 32 for 32x32
        else:
            self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1) # size 32 for 32x32 , size 800 for 800x800

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # size 16 for 32x32, size 400 for 800x800
        self.conv2 = nn.Conv2d(18, 18, kernel_size=3 ,stride = 1, padding=1) # size is 16 for 32x32, size is 400 for 800x800
        if imgSize == 32:
            if extraLayer:
                self.fc1 = nn.Linear(18*8*8, 64) # size is 8
            else:
                self.fc1 = nn.Linear(18*16*16, 64) # size is 16

        elif imgSize == 800:
            if extraLayer:
                self.fc1 = nn.Linear(18*200*200, 64) # size is 200
            else:
                self.fc1 = nn.Linear(18*400*400, 64) # size is 400

        self.fc2 = nn.Linear(64, 3)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        if self.extraLayer:
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            if self.imgSize == 32:
                x = x.view(-1, 18*8*8)
            elif self.imgSize == 800:
                x = x.view(-1, 18*200*200)
        else:
            if self.imgSize == 32:
                x = x.view(-1, 18*16*16)
            elif self.imgSize == 800:
                x = x.view(-1, 18*400*400)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, dataset, valset, n_epochs, learning_rate):
        trainLoader = dataset
        valLoader = valset

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # for plotting the loss
        trainHistory = []
        valHistory = []

        bestError = np.inf
        bestModelPath = "best_model.pth"

        # Move model to gpu if possible
        self.to(device)

        for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times
            runningLoss = 0.0
            totalTrainLoss = 0

            for inputs, labels in trainLoader:

                # Move tensors to correct device
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                # for single channel, take only first channel
                if self.singleChannel:
                    inputs = (inputs[:,0,:,:]).unsqueeze(1)


                labels = torch.tensor(labels, dtype = torch.long, device = device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                runningLoss += loss.item()
                totalTrainLoss += loss.item()

            trainHistory.append(totalTrainLoss / len(trainLoader))

            totalValLoss = 0
            # Do a pass on the validation set
            # We don't need to compute gradient,
            # we save memory and computation using torch.no_grad()
            with torch.no_grad():
              for inputs, labels in valLoader:
                  # Move tensors to correct device
                  inputs, labels = inputs.to(device), labels.to(device)
                  inputs = inputs.float()

                  # for single channel, take only first channel
                  if self.singleChannel:
                    inputs = inputs[:,0,:,:].unsqueeze(1)

                  labels = torch.tensor(labels, dtype = torch.long, device = device)
                  # Forward pass
                  predictions = self(inputs)
                  valLoss = criterion(predictions, labels)
                  totalValLoss += valLoss.item()

            valHistory.append(totalValLoss / len(valLoader))
            # Save model that performs best on validation set
            if totalValLoss < bestError:
                bestError = totalValLoss
                torch.save(self.state_dict(), bestModelPath)

        # Load best model
        self.load_state_dict(torch.load(bestModelPath))

        return trainHistory, valHistory

    def test(self, loadedTestData, name=""):
        self.to(device)
        correct = 0
        total = 0
        for inputs, labels in loadedTestData:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            # for single channel, take only first channel
            if self.singleChannel:
                inputs = inputs[:,0,:,:].unsqueeze(1)

            labels = torch.tensor(labels, dtype = torch.long, device = device)
            outputs = self(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        accuracy = 100 * float(correct) / total
        print('Accuracy of {} network , {} images: {:.2f} %'.format(name, total, accuracy))


######################################################################################################################################
if __name__=='__main__':
    # 32x32, 3 channels
    path = "C:\\Users\\J\\Desktop\\UOL SEM 2\\534 Applied AI\\AAI Assignment 3"
    # train
    print("32x32 Images")
    train32, val32 = imageDataLoader(path+"\\covid19_dataset_32_32\\train")
    test32 = imageDataLoader(path+"\\covid19_dataset_32_32\\test", train=False)

    network32 = SimpleConvolutionalNetwork(32, extraLayer=False)
    trainHistory, valHistory = network32.train(train32, val32, 30, 0.001)
    plotLosses(trainHistory, valHistory, "3x32x32, 1 conv layer")
    
    ######################################################################################################################################
    # 32x32, single channel
    singleNetwork32 = SimpleConvolutionalNetwork(32, extraLayer=False, singleChannel=True)
    trainHistory, valHistory = singleNetwork32.train(train32, val32, 30, 0.001)
    plotLosses(trainHistory, valHistory, "1x32x32, 1 conv layer")
    
    ######################################################################################################################################
    # 32x32, 3 channels, 2 conv
    newNetwork32 = SimpleConvolutionalNetwork(32, extraLayer=True)
    trainHistory, valHistory = newNetwork32.train(train32, val32, 30, 0.001)
    plotLosses(trainHistory, valHistory, "3x32x32, 2 conv layer")

    ######################################################################################################################################
    # 800x800, 3 channels
    print("800x800 Images")
    train800, val800 = imageDataLoader(path+"\\covid19_dataset_800_800\\train")
    test800 = imageDataLoader(path+"\\covid19_dataset_800_800\\test", train=False)

    network800 = SimpleConvolutionalNetwork(800, extraLayer=False)
    trainHistory, valHistory = network800.train(train800, val800, 30, 0.001)
    plotLosses(trainHistory, valHistory, "3x800x800, 1 conv layer")

    ######################################################################################################################################
    # 800x800, 3 channels, 2 conv
    newNetwork800 = SimpleConvolutionalNetwork(800, extraLayer=True)
    trainHistory, valHistory = newNetwork800.train(train800, val800, 30, 0.001)
    plotLosses(trainHistory, valHistory, "3x800x800, 2 conv layer")
    
    ######################################################################################################################################
    # test accuracy
    network32.test(test32, "3x32x32, 1 conv layer")
    singleNetwork32.test(test32, "1x32x32, 1 conv layer")
    newNetwork32.test(test32, "3x32x32, 2 conv layer")
    network800.test(test800, "3x800x800, 1 conv layer")
    newNetwork800.test(test800, "3x800x800, 2 conv layer")