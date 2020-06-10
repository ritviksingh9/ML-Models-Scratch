import numpy as np
import random
import json

class NN: 
    def __init__(self, layers_sizes, activation = 'sigmoid', cost_function = 'quadratic'):
        ''' 
            (List(Integer), String) --> None
            layers_sizes represent the number of nodes in each layer of the neural network.  
            The program assumes that the first layer is the input layer and the last one is the output layer.
            By default, it assumes a sigmoid activation function.
        '''
        self.layers_sizes = layers_sizes
        self.initialize_weights()
        self.function_activation = activation
        self.cost_function = cost_function

    def initialize_weights(self):
        '''
            (None) --> None
            This function randomly assigns the weights and biases such that the mean is 0 and a standard 
            deviation of 1/sqrt(number of connections to a neuron).
        '''
        #By convention there are no bias units for the input layer
        self.weights = [np.random.randn(self.layers_sizes[i], self.layers_sizes[i-1])/np.sqrt(self.layers_sizes[i-1]) for i in range(1, len(self.layers_sizes))]
        self.biases = [np.random.randn(i, 1) for i in self.layers_sizes[1:]]

    def forward_prop(self, input_layer):
        '''
            (List(Float)) --> List(Float)
            Takes in the input layer and propogates it through the network and returns the output layer.
        '''
        output_layer = input_layer
        for i, j in zip(self.weights, self.biases):
            output_layer = self.activation_function(np.dot(i, output_layer)+j)
        return output_layer

    def activation_function(self, z):
        '''
            (List(Float)) --> List(Float)
            Returns the activation function calculated on the given vector.
        '''
        if(self.function_activation == 'sigmoid'):
            return 1.0 / (1.0+np.exp(-z))
        elif(self.function_activation == 'tanh'):
            return np.tanh(z)
        elif(self.function_activation == 'relu'): #
            return np.maximum(0, z)

    def activation_function_derivative(self, z):
        '''
            (List(Float)) --> List(Float)
            Returns the derivative of the activation function calculated on the given vector.
        '''
        if(self.function_activation == 'sigmoid'):
            return self.activation_function(z)*(1-self.activation_function(z))
        elif(self.function_activation == 'tanh'):
            return 1-self.activation_function(z) ** 2
        elif(self.function_activation == 'relu'):
            return self.activation_function(z) / z

    def train(self, training_data, epochs, batch_size, learning_rate, test_data = None, lambda_term = 0):
        '''
            (List(Tuple), Integer, Integer, Float, List(Tuple) --> None
            Performs Stochastic Gradient Descent in order to train the network.  
            training_data and test_data are list of tuples containing the training inputs and the desired output.
            If test_data is provided, it will evaluate the accuracy of the network on it for every epoch.
        '''
        for i in range(epochs):
            random.shuffle(training_data)
            #creating a bunch of small batches of the training data
            batches = [training_data[j:j+batch_size] for  j in range(0, len(training_data), batch_size)]
            for k in batches:
                self.update_batch(k, learning_rate, lambda_term, len(training_data))
            if test_data:
                print("Epoch {} : {} / {}".format(i, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch "+str(i))

    def update_batch(self, batch, learning_rate, lambda_term, n):
        '''
            (List(Tuple), Float) --> None
            Updates the weights and biases using gradient descent and backpropogation from the small batch.
        '''
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]

        for x, y in batch:
            delta_gradient_weights, delta_gradient_biases = self.backprop(x, y)
            gradient_weights = [i+j for i, j in zip(gradient_weights, delta_gradient_weights)]
            gradient_biases = [i+j for i, j in zip(gradient_biases, delta_gradient_biases)]

        self.weights = [(1-learning_rate*lambda_term/n)*w-(learning_rate/len(batch))*nw for w, nw in zip(self.weights, gradient_weights)]
        self.biases = [b-(learning_rate/len(batch))*nb for b, nb in zip(self.biases, gradient_biases)]

    def backprop(self, x, y):
        '''
            (List(Float), List(Float)) --> Tuple(List, List)
            Returns a tuple the gradient of the cost function with respect to the weights and biases
            for each of the layers in the network.
        '''
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        
        activation = x
        activations = [x]
        z_vectors = []
        for i, j in zip(self.weights, self.biases):
            z = np.dot(i, activation) + j
            z_vectors.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        
        delta = self.error_delta_output(activations[-1], y, z_vectors[-1])
        gradient_weights[-1] = np.dot(delta, activations[-2].transpose())
        gradient_biases[-1] = delta

        for i in range(2, len(self.layers_sizes)):
            z = z_vectors[-i]
            derivative = self.activation_function_derivative(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * derivative
            gradient_weights[-i] = np.dot(delta, activations[-i-1].transpose())
            gradient_biases[-i] = delta
        
        return (gradient_weights, gradient_biases)

    def error_delta_output(self, output_layer, y, z_vector):
        '''
            (List(Float), List(Float), List(Float)) --> List(Float)
            This function calculates the error delta between the calculated output and the desired output
        '''
        if(self.cost_function == 'quadratic'):
            return self.delta_output(output_layer, y) * self.activation_function_derivative(z_vector)
        elif(self.cost_function == 'cross_entropy'):
            return self.delta_output(output_layer, y)

    def delta_output(self, output_layer, y):
        '''
            (List(Float), List(Float)) --> List(Float)
            Calculates the error between the calculated output and the desired output.
        '''
        return (output_layer-y)
    
    def evaluate(self, test_data):
        '''
            (List(Tuple)) --> Int
            Taking in the test_data, it returns the amount of times the network correctly evaluates the test data.
        '''
        test_result = [(np.argmax(self.forward_prop(x)), y) for (x, y) in test_data]
        return sum((x == y) for (x, y) in test_result)

    def save(self, file_name):
        '''
            (String) --> None
            This function takes in the name of the file and write the details of the Neural Network to a .json 
            file with that name
        '''
        data = {'layers_sizes': self.layers_sizes, 'weights': [i.tolist() for i in self.weights], 'biases': [i.tolist() for i in self.biases], 'cost_function': self.cost_function}
        with open(file_name, 'w') as outfile:
            json.dump(data, outfile) 

    def load(self, file_name):
        '''
            (string) --> None
            This function takes in the name of the file and loads the attributes into the NN object
        '''
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
            self.cost_function = data['cost_function']
            self.layers_sizes = data['layers_sizes']
            self.weights = [np.array(i) for i in data['weights']]
            self.biases = [np.array(i) for i in data['biases']]