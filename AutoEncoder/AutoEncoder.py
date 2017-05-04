import numpy as np

class SingleAutoEncoder(object):
    """
        This class define the implementation of the single auto-encoder
        It just contain two layer.
        The first weight matrix is encoder, and the last one is decoder
        You shouldn't use this class directly!!
    """
    # essential parameter
    hidden_num = -1     # The number of the neuron in this auto-encoder
    _x = []             # Training data
    epoch = 50
    eta = 0.05
    period = -1

    # Encoder and Decoder weight
    w1 = None           # The weight matrix of encoder
    w2 = None           # The weight matrix of decoder

    # Forward process variable
    s1 = None           # The result after the 1st matrix computation
    x1 = None           # The result after the hyperbolic tangent in the 1st layer
    s2 = None           # The result after the 2nd matrix computation

    # Backward process variable
    delta1 = None       # The derivation of the error in the hidden layer
    delta2 = None       # The derivation of the error in the last layer

    def __init__(self, _hidden, epoch=50, eta=0.5, period=10):
        """
            Constructor

            Arg:    _hidden     - The number of neuron in this layer
                    epoch       - The training times
                    eta         - Learning rate
                    period      - The period that you want to see the error log
        """
        self.hidden_num = _hidden
        self.epoch = epoch
        self.eta = eta
        self.period = period

    def assignX(self, _x):
        """
            Assign the training data

            Arg:    _data_list  - The list of the training data
        """
        self._x = _x

    def fit(self):
        """
            Fit the auto-encoder and return the list of the encode weight
            The method to initialize the weight is 0.5-shiftting about the random distribution

            Ret:    The encoder weight matrix
        """
        input_dim = np.shape(self._x)[1]
        self.w1 = np.random.random([self.hidden_num, input_dim]) - 0.5
        self.w2 = np.random.random([input_dim, self.hidden_num]) - 0.5

        for i in range(self.epoch):
            for j in range(len(self._x)):
                self.forward(self._x[j])
                if i % self.period == 0:
                    print "err: ", self.error(self._x[j])
                self.backward(self._x[j])
                self.update(self._x[j])
        return self.w1


    def forward(self, _input):
        """
            The forward process to compute the final vector

            Arg:    _input  - The data vector
        """
        self.s1 = np.matmul(self.w1, _input)
        self.x1 = np.tanh(self.s1)
        self.s2 = np.matmul(self.w2, self.x1)

    def error(self, _input):
        """
            Return the error value
            The method to validate the value is square error

            Arg:    _input  - The data vector
        """
        diff_vec = _input - self.s2
        return np.sum(np.square(diff_vec))

    def printDetail(self):
        """
            Print the detail
            This function should be add by yourself if you want to trace the detail
        """
        print "s1:"
        print self.s1
        print "s2:"
        print self.s2
        print "delta1:"
        print self.delta1
        print "delta2:"
        print self.delta2

    def backward(self, _input):
        """
            The backward process to compute the gradient

            Arg:    _input  - The data vector
        """
        self.delta2 = 2 * np.subtract(self.s2, _input)
        sech_2 = 1 / np.square(np.cosh(self.s1))
        self.delta1 = np.matmul(self.w2.T, self.delta2) * sech_2

    def update(self, _input):
        """
            Update the weight matrix

            Arg:    _input  - The data vector
        """
        self.w2 = self.w2 - self.eta * np.matmul(self.delta2, self.x1.T)
        self.w1 = self.w1 - self.eta * np.matmul(self.delta1, _input.T)

class AutoEncoder(object):
    """
        This class is the interface of the auto-encoder
        You can assign the different number of layer and different neuron in each layer

        There're three process you should do:
            1. create the object
            2. assign the data
            3. fit!
        You would get the list of the weight matrix after fitting toward the data
    """
    # Variable
    list_hidden_layer = []      # The list that contain the number of hidden neuron in each layer
    weight = []                 # The list to contain the weight matrix of each encoder
    _data_list = []             # Training data

    # Single auto-encoder variable
    epoch = -1
    eta = -1
    period = -1

    def __init__(self, _hidden, epoch=50, eta=0.5, period=10):
        """
            Constructor

            Arg:    _hidden     - The list that contain the number of neuron in each layer
                    epoch       - The training times
                    eta         - Learning rate
                    period      - The period that you want to see the error log
        """
        self.list_hidden_layer = _hidden
        self.epoch = epoch
        self.eta = eta
        self.period = period

    def assignX(self, _data_list):
        """
            Assign the training data

            Arg:    _data_list  - The list of the training data, it should follow the format that shown below:

                    [
                        [ [1], [-1], [1] ]
                    ]
        """
        self._data_list = _data_list

    def fit(self):
        """
            Fit the auto-encoder and return the list of the encode weight

            Ret:    The list of the encoder weight
        """
        for i in range(len(self.list_hidden_layer)):
            # Construct the Single auto-encoder object
            print "\n< start to train", i, " encoder >\n"
            single_encoder = SingleAutoEncoder(self.list_hidden_layer[i], \
                epoch=self.epoch, eta=self.eta, period=self.period)

            # Assign training data and fit the encoder
            single_encoder.assignX(self._data_list)
            w = single_encoder.fit()
            self.weight.append(w)

            # Feature Transform
            _data_after_transform = []
            for _data in self._data_list:
                _data_after_transform.append(np.tanh(np.matmul(w, _data)))
            self._data_list = np.asarray(_data_after_transform)
        return self.weight