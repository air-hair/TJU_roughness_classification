
from sklearn.svm import SVC

class SVM_model:
    def __init__(self, mode):
        
        if mode == "cubicSVM":
            self.classifier = SVC(kernel="poly", degree=3)
        elif mode == "quadraticSVM":
            self.classifier = SVC(kernel="poly", degree=2)
        elif mode == "fineGuassianSVM":
            self.classifier = SVC(kernel="rbf", gamma="scale")
        elif mode == "coarseGuassianSVM":
            self.classifier = SVC(kernel="rbf", gamma=8)
        
    def train(self, train_x, train_y):
        self.classifier.fit(train_x, train_y)
        return self.classifier

    def predict(self, test_x, classifier):
        y_pred = classifier.predict(test_x)
        return y_pred

      
      



