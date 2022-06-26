import h5py
import numpy as np
from PIL import Image

class Classifier:
    def __init__(self, m, c, epochs, learning_rate, log_rate, category_1, category_2, training_data = None, training_labels = None):
        self.m = m
        self.c = c
        self.training_parameters = {
            'epochs': epochs,
            'learning-rate': learning_rate,
            'log-rate': log_rate,
            'category-1': category_1,
            'category-2': category_2
        }
        if training_data is not None:
            self.training_data = training_data
            self.training_labels = training_labels
        else:
            self.training_data = None
            self.training_labels = None

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

        if callback is not None:
            callback(-1, -1)


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

    def test(self, img_path):
        print(img_path, type(img_path))
        img = Image.open(img_path)
        img = img.resize([150, 150])
        img = np.array(img)
        img = img.reshape(150*150*3)
        img = img / 255

        return self.predict(img)


    def to_hdf(self, fname, store_training_data = False):
        with h5py.File(fname, 'w') as f:
            model_group = f.require_group('model')
            model_group.clear()

            model_group.create_dataset('m', data=self.m)
            model_group.create_dataset('c', data=self.c)
            
            training_parameters_group = model_group.require_group('training-parameters')
            training_parameters_group.create_dataset('epochs', data=self.training_parameters['epochs'])
            training_parameters_group.create_dataset('learning-rate', data=self.training_parameters['learning-rate'])
            training_parameters_group.create_dataset('log-rate', data=self.training_parameters['log-rate'])
            training_parameters_group.create_dataset('category-1', data=self.training_parameters['category-1'])
            training_parameters_group.create_dataset('category-2', data=self.training_parameters['category-2'])

            if store_training_data:
                training_data_group = model_group.require_group('training-data')
                training_data_group.create_dataset('data', data=self.training_data)
                training_data_group.create_dataset('labels', data=self.training_labels)


    @classmethod
    def from_hdf(cls, fname):
        with h5py.File(fname, 'r') as f:
            model_group = f['model']
            m = model_group['m'][()]
            c = model_group['c'][()]
            training_parameters_group = model_group['training-parameters']
            epochs = training_parameters_group['epochs'][()].decode('utf-8')
            learning_rate = training_parameters_group['learning-rate'][()].decode('utf-8')
            log_rate = training_parameters_group['log-rate'][()].decode('utf-8')
            category_1 = training_parameters_group['category-1'][()].decode('utf-8')
            category_2 = training_parameters_group['category-2'][()].decode('utf-8')
            if 'training-data' in model_group:
                training_data_group = model_group['training-data']
                training_data = training_data_group['data'][()]
                training_labels = training_data_group['labels'][()]

                return cls(m, c, epochs, learning_rate, log_rate, category_1, category_2, training_data, training_labels)

            return cls(m, c, epochs, learning_rate, log_rate, category_1, category_2)

    
    @classmethod
    def new_model(cls, training_data: np.ndarray, training_labels: np.ndarray, training_parameters: dict):
        training_labels = training_labels.T

        # Initialize parameters (m, c) with zeros
        m = np.zeros((1, training_data.shape[1]))
        c = float(0)

        return cls(m, c, training_parameters['epochs'], training_parameters['learning-rate'], training_parameters['log-rate'], training_parameters['category-1'], training_parameters['category-2'], training_data, training_labels)
