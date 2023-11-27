import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self,config):
        super(CNN, self).__init__()
        # Params 
        num_channels = config['num_channels']
        num_classes = config['num_classes']
        num_filters = config['num_filters']
        time_pool_size = int(config["resizewidth"]/32)
        dropout_prob = config['dropout_prob']
        # CNN architecture 
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(num_filters, 2 * num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(4 * num_filters, 4 * num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(),
            nn.Conv2d(4 * num_filters, 4 * num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * num_filters),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(time_pool_size, 1)),

            nn.Dropout(dropout_prob),
            nn.Flatten(),
            nn.Linear(11520, num_classes),
            
        )
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    # Forward pass of model
    def forward(self, x, label = None):
        if label is not None:
            loss = self.criterion(x, label)
            return loss
        else :
            return self.model(x)
def choose_optim(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler=lr_scheduler.MultiStepLR(optimizer,milestones=[15],gamma=0.1)
    return optimizer, scheduler
