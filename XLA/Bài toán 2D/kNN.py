import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score

#Input Data
X =  np.array([
    [158, 58],
    [158, 59],
    [158, 63],
    [160, 59],
    [160, 60],
    [163, 60],
    [163, 61],
    [160, 64],
    [163, 64],
    [165, 61],
    [165, 62],
    [165, 65],
    [168, 62],
    [168, 63],
    [168, 66],
    [170, 63],
    [170, 64],
    [170, 68],
])
Y = np.array(["R", "R", "R", "R", "R", "R", "R", "G", "G", "G", "G", "G", "G", "G", "G", "G", "G", "G"])

#kNN 
def kNN(X, Y):
    #Choose Test and Train
    print("Number of Data:    {}".format(len(X)))
    print("Number of classes: {}  are {}".format(len(np.unique(Y)), np.unique(Y))) #np.unique(Y): name of class
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=9)
    print(X_train)
    #Plot
    for i, sample in enumerate(X_train):
        if y_train[i] == 'R':
            plt.scatter(X_train[i, 0], X_train[i, 1], s=40, marker='_', color='red')
        else:
            plt.scatter(X_train[i, 0], X_train[i, 1], s=40, marker='+', color='green')
    for i, sample in enumerate(X_test):
        plt.scatter(X_test[i, 0], X_test[i, 1], s=20, marker='x', color='black')     
    plt.axis('equal')
    plt.show()

    #Calc distance
    clf = neighbors.KNeighborsClassifier(n_neighbors = 2, p = 1)#p = 2 the standard Euclidean distance
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Result:  {}".format(y_pred))
    print("Reality: {}".format(y_test))
    print("Accuracy of {}NN: {}%".format(2, (100*accuracy_score(y_test, y_pred))))
    
    #Plot
    for i, sample in enumerate(X_train):
        if y_train[i] == 'R':
            plt.scatter(X_train[i, 0], X_train[i, 1], s=40, marker='_', color='red')
        else:
            plt.scatter(X_train[i, 0], X_train[i, 1], s=40, marker='+', color='green')
    for i, sample in enumerate(X_test):
        if y_pred[i] == 'R':
            plt.scatter(X_test[i, 0], X_test[i, 1], s=40, marker='_', color='red')
        else:
            plt.scatter(X_test[i, 0], X_test[i, 1], s=40, marker='+', color='green')
    for i, sample in enumerate(y_pred):
        if sample != y_test[i]:
            circle1 = plt.Circle((X_test[i, 0], X_test[i, 1]), .2, color='black', fill = False)
            plt.gcf().gca().add_artist(circle1)
    plt.show()                                                                                                                                                                                                                                                                                                                                                                      

#Main
if __name__ == "__main__":
    kNN(X, Y)