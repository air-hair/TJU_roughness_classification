from config import Config
from loader import load_data
from model import KNN_model
from evaluate import Evaluator
import torch
import logging
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)

def main(config):
    # Load training and validation data from files
    train_data, valid_data, train_label, valid_label = load_data(Config)
    
    # Initialize KNN model
    model = KNN_model(config["mode"], config["n_neighbors"])
    
    # Train model on training data
    model = model.train(train_data, train_label)
    
    # Make predictions on validation data
    pred = model.predict(valid_data)
    
    # Log model used
    logger.info(config["mode"])
    
    # Evaluate model performance
    evaluator = Evaluator(valid_label, pred)


  





if __name__ == "__main__":
    main(Config)
