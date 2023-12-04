# TJU_roughness_classification
Machine learning models used in the paper "A flexible piezoelectric PVDF/MXene pressure sensor for roughness discrimination"

This repository contains code used for classifying roughness in sensor data obtained from flexible sensors. The classification is performed using various machine learning models including CNN, RNN, LSTM, KNN, and SVM.

## Project Structure

The repository is organized into separate folders for each model used in the classification task. Each model folder contains the following files:

- `config.py`: File for configuring parameters specific to the model.
- `loader.py`: Module for loading and preprocessing the sensor data.
- `model.py`: File defining the architecture and implementation of the model.
- `evaluate.py`: Module for evaluating the predictions and performance metrics.
- `main.py`: Main script for training the model and making predictions.

## Folder Structure

- `CNN/`: Contains files related to the Convolutional Neural Network model.
- `LSTM&RNN/`: Contains files related to the Long Short-Term Memory model and the Recurrent Neural Network model.
- `KNN/`: Contains files related to the K-Nearest Neighbors model.
- `SVM/`: Contains files related to the Support Vector Machine model.

## Usage

To utilize this codebase, follow these steps:

1. Navigate to the specific model folder (`CNN/`, `RNN/`, etc.).
2. Configure parameters in `config.py` if necessary.
3. Run `main.py` to train the model and make predictions.

Please ensure you have the required dependencies installed before running the code. 

## Requirements

The code is implemented in Python and requires the following dependencies:

- **torch** (Version 1.12.1)
- **scikit-learn** (Version 1.0.2)
- **numpy** (Version 1.20.0)
- **pandas** (Version 1.4.4)



## Citation

If you find this code useful for your work, you are welcome to reference it directly via its GitHub repository link.


## License

This repository does not currently have a specific license. All rights are reserved. Without a specified license, the code is protected by default under the applicable copyright laws. Users are allowed to view and fork this repository, but any unauthorized use, reproduction, or distribution of the code is prohibited without explicit permission from the author(s).


## Contact Information

If you have any questions, suggestions, or inquiries regarding the code or its usage, feel free to contact me through:

- Email: 1582669020@qq.com



