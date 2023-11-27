from config import Config
from loader import load_data
from model import LSTM_model, choose_optimizer
from evaluate import Evaluator
import torch
import logging
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import random

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def main(config):
    # load data
    train_data, valid_data = load_data(config)

    # build model
    model = LSTM_model(config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("CUDA is available")
        model = model.cuda()

    # choose optimizer
    optimizer = choose_optimizer(config, model)

    # build evaluator
    evaluator = Evaluator(config, model, valid_data, logger)

   

    # train model
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        logger.info("epoch %d begin" % (epoch + 1))
        for i, (x, y) in enumerate(train_data):
  
            if cuda_flag:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            if i % 4 == 0:
                logger.info("epoch %d, batch %d, loss: %f" % (epoch + 1, i, loss.item()))
            train_loss.append(loss.item())
        logger.info("epoch %d, average loss: %f" % (epoch + 1, np.mean(train_loss)))
        evaluator.evaluate(epoch)
    confusion = confusion_matrix(evaluator.y_true, evaluator.y_pred)
    acc = accuracy_score(evaluator.y_true, evaluator.y_pred)
    precision = precision_score(evaluator.y_true, evaluator.y_pred,average='macro')
    recall = recall_score(evaluator.y_true, evaluator.y_pred,average='macro')
    f1 = f1_score(evaluator.y_true, evaluator.y_pred,average='macro')
    print(config["mode"])
    print(confusion, "accuracy: ",acc, 
              "precision: ",precision,
               "recall score:", recall,
                "f1 score: ", f1)





if __name__ == "__main__":
    main(Config)
