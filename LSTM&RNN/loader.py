import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import random

class DataProcess:
    def __init__(self, config):
        # Initialize with config
        self.raw_data_path = config['raw_data_path']
        self.file_num = config['file_num']
        self.config = config
        # Empty lists to store processed data
        self.train_data = []
        self.valid_data = []

        self.data_processing()
        
        
        # Process data files 
    def data_processing(self):
        for i, path in enumerate(self.raw_data_path):
            self.get_data(path, i)
        return
    # Read individual file  
    def get_data(self, path, label):
        self.data = []
        for file in range(self.file_num):
            current_path = path + str(file) + '.csv'
            _data = pd.read_csv(current_path)
            # Pad data
            padded_data = self.padding(_data)
            label = torch.LongTensor([label])
            # Add label and shuffle
            self.data.append([padded_data,label])
        random.shuffle(self.data)
        self.train_data.extend(self.data[:int(self.config["train_valid_ratio"]*self.file_num)])
        self.valid_data.extend(self.data[int(self.config["train_valid_ratio"]*self.file_num):])
        
        return
    # Pad tensor to fixed size
    def padding(self, data):
        # padding
        data = torch.FloatTensor(data['1'])
        data = F.pad(data, (0, 2500-tuple(data.shape)[0]), "constant", 0)
        
        return data

   
class DataGenerator:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def load_data(config):
    dg = DataProcess(config)
    train_data, valid_data = dg.train_data, dg.valid_data
    train_dg = DataGenerator(train_data)
    valid_dg = DataGenerator(valid_data)
    train_dl = DataLoader(train_dg, batch_size=config['batch_size'], shuffle=True)
    valid_dl = DataLoader(valid_dg, batch_size=config['batch_size'], shuffle=False)
    return train_dl, valid_dl
    

if __name__ == "__main__":
    from config import Config
    dg = DataProcess(Config)

