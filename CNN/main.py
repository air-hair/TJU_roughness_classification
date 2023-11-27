from model import CNN, choose_optim
from config import Config
from loader import dataProcess
from evaluate import Evaluator
import torch.nn as nn
import torch.optim as optim 
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def main(config):
    # Load data
    data = dataProcess(config)
    train_loader, test_loader = data.get_data()
  
    # Initialize model
    model = CNN(config)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer, scheduler = choose_optim(model, config)
    evaluator = Evaluator(model, config)

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        # Forward and backward pass on batches
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()    
            optimizer.zero_grad()
            outputs = model(images)
            loss = model(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('loss:{}'.format(loss))
      
        # Evaluate on test set  
        y_true, y_pred = evaluator.evaluate(test_loader)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Accuracy: {accuracy:.4f}")
    # Print metrics
    confusion = confusion_matrix(evaluator.y_true, evaluator.y_pred)
    acc = accuracy_score(evaluator.y_true, evaluator.y_pred)
    precision = precision_score(evaluator.y_true, evaluator.y_pred,average='macro')
    recall = recall_score(evaluator.y_true, evaluator.y_pred,average='macro')
    f1 = f1_score(evaluator.y_true, evaluator.y_pred,average='macro')
    print(confusion, "accuracy: ",acc, 
              "precision: ",precision,
               "recall score:", recall,
                "f1 score: ", f1)
    # save model
    torch.save(model.state_dict(), 'model.ckpt')


if __name__ == '__main__':
    main(Config)