import pandas as pd

import numpy as np
import random

class DataProcess:
    def __init__(self, config):
        self.raw_data_path = config['raw_data_path']
        self.file_num = config['file_num']
        self.config = config
        self.train_data = []
        self.train_label = []
        self.valid_data = []
        self.valid_label = []
        self.data_processing()
    
        

    def data_processing(self):
        # Loop through file paths
        for i, path in enumerate(self.raw_data_path):
            self.get_data(path, i)
        return
    # Load and process file
    def get_data(self, path, label):
        self.data = []
        
        for file in range(self.file_num):
            current_path = path + str(file) + '.csv'
            
            _data = pd.read_csv(current_path)
            padded_data = self.padding(_data)
            label = np.array(label)
            self.data.append([padded_data,label])
        random.shuffle(self.data)
        rows = self.data[:int(self.config["train_valid_ratio"]*self.file_num)]
        self.train_data.extend([row[0] for row in rows])
        self.train_label.extend([row[1] for row in rows])
        rows = self.data[int(self.config["train_valid_ratio"]*self.file_num):]
        self.valid_data.extend([row[0] for row in rows])
        self.valid_label.extend([row[1] for row in rows])
        
        return
    # Pad time series data
    def padding(self, data):
        data = np.array(data['1']) 
        data = np.pad(data, (0, 2500-data.shape[0]), constant_values=(0))
        return data

   
    
def load_data(config):
    dg = DataProcess(config)
    train_data, valid_data, train_label, valid_label = dg.train_data, dg.valid_data, dg.train_label, dg.valid_label

    return train_data, valid_data, train_label, valid_label
    

if __name__ == "__main__":
    from config import Config
    train_data, valid_data, train_label, valid_label = load_data(Config)
 