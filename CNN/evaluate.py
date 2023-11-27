import torch
import torch.nn as nn
import torch.optim as optim 
class Evaluator:
    # Initialize with trained model and config
    def __init__(self, model, config):
        self.model = model
        self.config = config
    # Evaluate trained model on test set
    def evaluate(self, test_loader):
        # Set model to eval mode
        self.model.eval()
    
        # Initialize lists to store predictions and true labels
        self.y_pred = []
        self.y_true = []
        # Iterate over test dataloader
        with torch.no_grad():
            for images, labels in test_loader:
                # Move data to GPU if available
                if torch.cuda.is_available():  
                    images, labels = images.cuda(), labels.cuda()

                # Forward pass and get predictions
                outputs = self.model(images)
            
                # Get top-1 class from model output
                _, predicted = torch.max(outputs.data, 1)
                print("predicted:", predicted)
                print("label",labels)
                print(predicted==labels)
                self.y_pred.extend(predicted.cpu().numpy())
                self.y_true.extend(labels.cpu().numpy())
        return self.y_true, self.y_pred