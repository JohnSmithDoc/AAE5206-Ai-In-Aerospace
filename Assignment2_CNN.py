import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # (M - K + 2pd) / s + 1 = (28 - 12 + 0) / 2 + 1 = 9
            # 32 * (9*9)
            nn.Conv2d(1, 32, kernel_size=12, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # (M - K + 2pd) / s + 1 = (9-2 + 0)/2 + 1 = 4
            # 所以输出是 32 * （4*4）
            # nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            # 因为上一次池化输出后，output是 32 * （4*4）所以我们需要使用zero-padding
            # nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1, padding_mode="zeros"),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU())

        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes
        self.type = 'CNN'

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)

        # 每次forward只初始化一次
        if self.fc1 is None:
            self.fc1 = nn.Linear(out.size(1), 512)
            print(out.size(1))
        out = self.fc1(out)

        # 每次forward只初始化一次
        if self.fc2 is None:
            self.fc2 = nn.Linear(512, self.num_classes)
        out = self.fc2(out)

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

                if loss.item() < 0.005:
                    # 保存模型及其参数，退出
                    torch.save(model.state_dict(), 'model.pth')
                    break

    # 运行完成的话，保存模型及其参数，退出
    torch.save(model.state_dict(), 'model.pth')


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
    model = ConvNet()

    # Loss and optimizer
    # CrossEntropyLoss自动包含了softmax
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ### step 3: train the model
    train(train_loader, model, criterion, optimizer, num_epochs=5)

    ### step 4: test the model
    # 加载模型参数
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # 切换到评估模式
    test(test_loader, model)





