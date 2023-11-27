
from sklearn.neighbors import KNeighborsClassifier

class KNN_model:
    def __init__(self, mode, n_neighbors):
        
        if mode == "fineKNN":
            self.classifier = KNeighborsClassifier(
                                               n_neighbors= n_neighbors,
                                               algorithm= 'ball_tree',
                                               leaf_size= 30,
                                               p =1)
        elif mode == "weightedKNN":
            self.classifier = KNeighborsClassifier(
                                               n_neighbors= n_neighbors,
                                               weights= "distance")
        elif mode == "cubicKNN":
            self.classifier =  KNeighborsClassifier(
                                               n_neighbors= n_neighbors,
                                               metric='euclidean',
                                               p = 3)       
        elif mode == "cosineKNN":
            self.classifier = KNeighborsClassifier(n_neighbors= n_neighbors,
                                                    metric='cosine', 
                                                    weights= "distance")
        elif mode == "mediumKNN":
            self.classifier = KNeighborsClassifier(n_neighbors= n_neighbors * 2,
                                                   algorithm= 'kd_tree',
                                                   leaf_size= 60)
        elif mode == "coarseKNN":
            self.classifier = KNeighborsClassifier(n_neighbors= n_neighbors * 4,
                                                   algorithm= 'kd_tree',
                                                   leaf_size= 100)
        
    def train(self, train_x, train_y):
        self.classifier.fit(train_x, train_y)
        return self.classifier

    def predict(self, test_x, classifier):
        y_pred = classifier.predict(test_x)
        return y_pred

      
      



