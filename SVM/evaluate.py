
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


class Evaluator:
    def __init__(self,  valid_label, pred):
        self.valid_label = valid_label
        self.pred = pred
        self.show_result()
    def show_result(self):
        confusion = confusion_matrix(self.valid_label, self.pred)
        acc = accuracy_score(self.valid_label, self.pred)
        precision = precision_score(self.valid_label, self.pred, average='macro')
        recall = recall_score(self.valid_label, self.pred, average='macro')
        f1 = f1_score(self.valid_label, self.pred, average='macro')
        print(confusion, "accuracy: ",acc, 
              "precision: ",precision,
               "recall score:", recall,
                "f1 score: ", f1)
