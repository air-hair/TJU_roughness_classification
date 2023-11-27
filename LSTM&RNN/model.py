import torch.nn as nn
import torch


class LSTM_model(nn.Module):
    # Initialize model with config parameters
    def __init__(self, config):
        super(LSTM_model, self).__init__()
        input_size = config["input_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        output_size = config["class_num"]
        self.mode = config["mode"]
        # Define LSTM, RNN or combination based on mode
        if self.mode == "LSTM":
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.mode == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.mode == "LSTM+RNN":
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        # Linear layer and activations
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.layernorm = nn.LayerNorm(input_size)
        self.relu = nn.ReLU()
        # Loss function
        self.loss = nn.functional.cross_entropy
    # Forward pass of model
    def forward(self, x, target = None):
        # x: [B, I]
        # target: [B]
        if self.mode == "LSTM":
            output, (_ , _) = self.lstm(x)
            output = self.relu(output)
            output = self.fc1(output)
        elif self.mode == "RNN":
            output, _ = self.rnn(x)
            output = self.relu(output)
            output = self.fc1(output)
        elif self.mode == "LSTM+RNN":
            output, (_ , _) = self.lstm(x)
            output, _ = self.rnn(output)
            output = self.relu(output)
            output = self.fc1(output)
      
      
        
        if target is not None:
            loss = self.loss(output, target.squeeze())
            return loss
        else:
            return output
    


# Choose optimizer
def choose_optimizer(config, model):
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        raise Exception("optimizer not supported: {}".format(config["optimizer"]))