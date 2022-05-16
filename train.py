# Load libraries
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.externals import joblib
import numpy as np

# Load the iris data
iris = datasets.load_iris()

# Create a matrix, X, of features and a vector, y.
X, y = iris.data, iris.target

# We put a Percepton class from the other file - laboratory4.ipynb
class Perceptron:
    
    #n_iter - numer of iterations -> all rows of our data 10 times
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    #what are the wages of our model?
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        #as many as our iterator
        for _ in range(self.n_iter):
            errors = 0
            #checking how many rows are good
            for xi, target in zip(X,y): #connect data with target
                update=self.eta*(target-self.predict(xi)) #computing prediction and checking it -> target minus predict must be equal to zero
                self.w_[1:] += update *xi #update of wages
                self.w_[0] += update 
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    #scalar product of row of data with a wage
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    #our step function, when our where is bigger than zero, than we have class 1
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)

#creating model
model = Perceptron()
model.fit(X, y)

# Save the trained model as a pickle string.
saved_model = pickle.dumps(model)

# Save the model as a pickle in a file
joblib.dump(model, 'model.pkl')