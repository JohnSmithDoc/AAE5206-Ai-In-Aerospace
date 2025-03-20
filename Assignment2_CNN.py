#   Student Number: 24066688g
#   Student   Name: Yang Guodong

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# temp solution
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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
            # 如果上一次进行了池化输出，输出后，output是 32 * （4*4）此时我们需要使用zero-padding，如果上一次不是池化输出，就没必要padding了
            # nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1, padding_mode="zeros"),

            # stride为1，暂不padding
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # 按照要求，和全连接层连接之前再经过一层Relu
            nn.ReLU())

        # 不直接计算全连层从卷积层最后得到的输入个数，不然每次变动都要认为计算，所以我们直接在前向传播时动态计算
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


def test_in_training(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the training images: {} %'.format(100 * correct / total))

def train(train_loader, model, criterion, optimizer, num_epochs, save_param: bool = True):
    # Train the model
    total_step = len(train_loader)

    # 初始的loss设为一个较大的值
    last_loss = 100

    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                test_in_training(train_loader, model)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, step + 1, total_step, loss.item()))

                # 每100次同步检查一下，loss变小就保存一下此次的模型参数
                if loss.item() < last_loss:
                    model_params = model.state_dict()

                last_loss = loss.item()

    # 训练完成的话，如果需要保存模型及其参数，则保存
    if save_param:
        torch.save(model_params, 'model.pth')

def test(test_loader, model):
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# 可视化卷积核
def visualize_kernels(kernels):
    # 正常应该是32
    num_kernels = kernels.shape[0]
    # 正常应该只一层
    num_channels = kernels.shape[1]
    size_x = kernels.shape[2]
    size_y = kernels.shape[3]

    # 穿件4*8的子图网格，每个子图12*12，实际应当会画出32个
    fig, axes = plt.subplots(4, 8, figsize=(size_x, size_y))
    for i in range(num_kernels):
        for j in range(num_channels):
            ax = axes[i // 8, i % 8]
            ax.imshow(kernels[i, 0].cpu().numpy(), cmap='gray')
            ax.axis('off')
    plt.show()


if __name__ == '__main__':
    ### step 1: prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    ### step 2: instantiate neural network and design model
    model = ConvNet()

    # Loss and optimizer
    # CrossEntropyLoss自动包含了softmax
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 临时切换要做什么
    training: bool = True
    testing: bool = True

    # 同时训练并紧接着评估
    if training and testing:
        ### step 3: train the model
        train(train_loader, model, criterion, optimizer, num_epochs=20)
        ### step 4: 加载刚刚训练好的最好的模型参数，并test the model
        model.load_state_dict(torch.load('model.pth'))
        model.eval()  # 切换到评估模式
        test(test_loader, model)

    # 不训练，只评估，这时我们加载之前保存的最新的模型文件进行测试
    elif not training and testing:
        # 加载保存的使loss最小的模型参数，然后测试
        # 因为使用了动态计算神经元个数，所以这里需要通过一次前向传播来初始化全连接层，这样才能正常加载对应的模型参数，临时先使用一次训练来代替，但不保存训练参数
        train(train_loader, model, criterion, optimizer, num_epochs=1, save_param=False)
        model.load_state_dict(torch.load('model.pth'))
        model.eval()  # 切换到评估模式
        test(test_loader, model)

    else:
        print("do nothing")

    # 获取第一个卷积层的卷积核
    conv1 = model.layer1[0]  # 访问 nn.Sequential 容器中的第一个卷积层
    kernels = conv1.weight.data
    # 检查卷积核的形状
    print(f"Conv1 kernels shape: {kernels.shape}")
    # 可视化第一个层的卷积核
    visualize_kernels(kernels)
