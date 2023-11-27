
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self,  valid_label, pred):
        self.valid_label = valid_label
        self.pred = pred
        self.show_result()
    # Print evaluation metrics   
    def show_result(self):
        
        # Generate confusion matrix
        confusion = confusion_matrix(self.valid_label, self.pred)
        
        # Calculate accuracy 
        acc = accuracy_score(self.valid_label, self.pred)
        
        # Calculate precision
        precision = precision_score(self.valid_label, self.pred, average='macro') 
        
        # Calculate recall
        recall = recall_score(self.valid_label, self.pred, average='macro')
    
        # Calculate F1 score
        f1 = f1_score(self.valid_label, self.pred, average='macro')
    
        # Print evaluation metrics
        print(confusion, 
             "accuracy: ", acc,  
             "precision: ", precision,
             "recall score:", recall,
             "f1 score: ", f1)

        
