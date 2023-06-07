import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import nn, optim

# downloading the dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])

dataTrain = MNIST(root="./", download=True, train=True, transform = transform)
dataTest = MNIST(root="./", download=True, train=False, transform = transform)

# loading the data and labels
trainloader = torch.utils.data.DataLoader(dataTrain, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(dataTest, batch_size=64, shuffle=True)

# build the network
model = nn.Sequential(nn.Linear(784, 128), # input layer
                      nn.ReLU(),
                      nn.Linear(128, 64), # hidden layer
                      nn.ReLU(),
                      nn.Linear(64, 10),   # output layer
                      nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss() #loss function
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9) #optimization function

epochs = 10
steps = 0

# training test
for e in range(epochs):

    #training part
    test_loss = 0
    for images, labels in trainloader:    
        images = images.view(images.shape[0], -1) # flatten the images to 784 vector
        optimizer.zero_grad()
        output = model(images) # passes the images through the model
        loss = criterion(output, labels) # calculates the NLL loss
        loss.backward() # Backpropagating
        optimizer.step() # Updates the weight

        test_loss += loss.item()

    print("Epoch: ", e, "Training loss: ", test_loss/len(trainloader))
    print("\n")

    # validation part
    validation_loss = 0
    accuracy = 0
    with torch.no_grad(): #turn off gradients for validation as saves memory
        for images,labels in testloader:
            images = images.view(images.shape[0], -1)
            output = model(images)
            loss = criterion(output, labels)
            validation_loss += loss.item()

            ps = torch.exp(output) # probability distribution
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Epoch: ", e, "Validation loss: ", validation_loss/len(testloader))
        print("\n")
        print("Epoch: ", e, "Accuracy: ", accuracy/len(testloader)*100)
        print("\n")
        
