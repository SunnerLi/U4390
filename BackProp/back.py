import numpy as np

w = [   0, 
        np.asarray([[0.1, 0.2], [0.9, 0.8]]), 
        np.asarray([[0.1, -0.1], [0.2, 0.1], [0.3, 0.5]]), 
        np.asarray([[-2.0, 1.0, 3.0]])
    ]
"""
w = [   0,
        np.asarray([[0.2, 0.3], [0.8, 0.7]]),
        np.asarray([[1.0, -1.0]])
    ]
"""

class Net(object):
    weight = []
    depth = -1
    s = []
    x = []
    delta = []
    eta = 0
    should_print = False

    def __init__(self, depth, eta=0.05, should_print=False):
        """
            Constructor of NNet

            Arg:    depth           - The depth of the network
                    eta             - The learning rate
                    should_print    - Should print the log to the screen
        """
        self.depth = depth
        self.weight = [0] * (depth + 1)
        self.eta = eta
        self.should_print = should_print

    def assignWeight(self, index, weight):
        """
            Assign weight matrix to the specific layer

            Arg:    index   - The index of the layer you want to assign
                    weight  - The weight numpy object
        """
        self.weight[index] = weight

    def forward(self, _input):
        """
            Forward process of the back propagation algorithm

            Arg:    _input  - The training vector
        """
        self.s = []
        self.x = []
        self.s.append([0])
        self.x.append(_input[np.newaxis].T)
        for i in range(1, self.depth + 1):            
            _s = np.matmul(self.weight[i], self.x[i - 1])
            _x = np.tanh(_s)
            self.s.append(_s)
            self.x.append(_x)         
            
            if self.should_print:
                print 's', i, '=', self.weight[i], '*', self.x[i-1], '=', _s
                print 'x', i, '= tan( ', _s, '=', _x

    def backward(self, tag):
        """
            Backward process of the back propagation algorithm

            Arg:    tag     - The output that except the network print
        """
        self.delta = [0] * (self.depth + 1)
        self.delta[self.depth] = -2 * ( tag - self.s[self.depth] )

        if self.should_print:
            print 'delta L: ', self.delta[self.depth]

        for i in range(self.depth-1, 0, -1):
            tanh_diff = 1/np.square(np.cosh(self.s[i]))            
            self.delta[i] = np.matmul(np.transpose(self.weight[i+1]), self.delta[i+1]) * tanh_diff
            if self.should_print:
                print 'delta', i, ':', self.weight[i+1], '*', self.delta[i+1], '*', tanh_diff, '=', self.delta[i]

    def update(self):
        """
            Update process of the back propagation algorithm
        """
        print '\n<update>'
        for i in range(1, self.depth+1):
            if self.should_print:
                print 'W', i, '=', self.weight[i], '-', self.delta[i], '*', self.x[i-1].T, '=', self.weight[i] - (self.eta * np.matmul(self.delta[i], self.x[i-1].T)), '\n'
            self.weight[i] -= (self.eta * np.matmul(self.delta[i], self.x[i-1].T))

if __name__ == '__main__':
    net = Net(3, should_print=True)
    for i in range(4):
        net.assignWeight(i, w[i])

    print net.weight
    for i in range(2):
        net.forward(np.asarray([1, -1]))
        print ""
        net.backward([0])
        net.update()
        print net.weight