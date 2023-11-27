
import torch
import numpy as np

class Evaluator:
    def __init__(self, config, model, data, logger):
        self.data = data
        self.model = model
        self.config = config
        self.logger = logger
        

    def evaluate(self, epoch):
        """
        Evaluate the model on the given data.
        """
        self.y_pred = []
        self.y_true = []
        self.logger.info("Test the %d epoch accuracy" % (epoch + 1))
        self.state_dict = {"correct" : 0, "wrong" : 0}
        self.model.eval()
        for i, (data, label) in enumerate(self.data):
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            with torch.no_grad():
                pred = self.model(data)
            self.write_state(pred, label)
        acc = self.show_state()
        return acc

    def write_state(self, pred, label):
        assert len(pred) == len(label)

        for pred, label in zip(pred, label):
            self.y_pred.append(np.array(torch.argmax(pred).cpu()).item())
           
            self.y_true.extend(np.array(label.cpu()))
            if torch.argmax(pred) == label:
                self.state_dict["correct"] += 1
            else:
                self.state_dict["wrong"] += 1
        return

    def show_state(self):
        self.logger.info("Accuracy：%f" % (self.state_dict["correct"] / (self.state_dict["correct"] + self.state_dict["wrong"])))
        return self.state_dict["correct"] / (self.state_dict["correct"] + self.state_dict["wrong"])
