import numpy as np

class Classifier:
    def __init__(self, training_data: np.ndarray, training_labels: np.ndarray, training_parameters: dict):
        self.training_data = training_data
        self.training_labels = training_labels.T
        self.training_parameters = training_parameters

        print(self.training_data.shape, self.training_labels.shape)

        # Initialize parameters (m, c) with zeros
        self.m = np.zeros((1, training_data.shape[1]))
        self.c = float(0)

    def sigmoid(self, x: np.ndarray):
        """Returns sigmoid of numpy array.

        Args:
            x (numpy array): Array for which sigmoid should be calculated.

        Returns:
            numpy array: Array with sigmoid applied.
        """
        return 1/(1+np.exp(-x))


    def cost(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """Returns cost of model.

        Args:
            y (numpy array): Array of correct labels.
            y_hat (numpy array): Array of predicted labels.

        Returns:
            ndarray: Cost of model.
        """
        return (-1 / y.shape[0]) * np.sum(np.dot(y, np.log(y_hat + 1e-5).T) + np.dot((1 - y), np.log(1 - y_hat + 1e-5).T), axis = 1)
    

    def predict(self, X: np.ndarray):
        """Predicts labels for given data using the formula: y = m*x + c.

        Args:
            X (numpy array): Numpy array containing all the images which need to be classified.

        Returns:
            numpy array: Labels for given data.
            numpy array: Numerical prediction ranging from 0 to 1.
        """
        
        prediction = self.sigmoid(np.dot(self.m, X.T) + self.c)
        label = np.where(prediction > 0.5, 1, 0)
        return label, prediction


    def train(self, callback = None):
        """Trains the model for given number of epochs"""
        epochs = int(self.training_parameters['epochs'])
        learning_rate = float(self.training_parameters['learning-rate'])
        debug_print_rate = int(self.training_parameters['log-rate'])

        for epoch in range(1, epochs + 1):
            cost = self.train_one_iteration(self.training_data, self.training_labels, learning_rate)
            if epoch % debug_print_rate == 0 and callback is not None:
                callback(epoch, cost)
                print(f'Epoch: {epoch}, Cost: {cost}')


    def train_one_iteration(self, X: np.ndarray, Y: np.ndarray, learning_rate: float):
        """Trains the model for 1 iteration.

        Args:
            X (numpy array): Numpy array of images to be used for training
            Y (numpy array): Numpy array of labels for images for training
            learning_rate (float): Learning rate for gradient descent.
        """
    
        # Predict
        _, prediction = self.predict(X)
        cost = self.cost(Y, prediction)
        cost = np.squeeze(np.array(cost))

        # Update parameters
        dm = (1 / X.shape[0]) * np.dot((prediction - Y), X)
        dc = (1 / X.shape[0]) * (prediction - Y)
        self.m = self.m - learning_rate * dm
        self.c = (self.c - learning_rate * dc)[0][0]
        return cost