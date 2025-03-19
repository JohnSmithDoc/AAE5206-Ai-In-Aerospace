import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)
        self.type = 'MLP'

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        self.type = 'CNN'

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def create_dataloader():
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data',
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor())

    test_dataset = torchvision.datasets.MNIST(root='data',
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False)

    return train_loader, test_loader


def train(train_loader, model, criterion, optimizer, num_epochs):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            if model.type == 'MLP':
                images = images.reshape(-1, 28 * 28)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))


def test(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if model.type == 'MLP':
                images = images.reshape(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    ### step 1: prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    ### step 2: instantiate neural network and design model
    model = NeuralNet()
    # model = ConvNet()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ### step 3: train the model
    train(train_loader, model, criterion, optimizer, num_epochs=5)

    ### step 4: test the model
    test(test_loader, model)





