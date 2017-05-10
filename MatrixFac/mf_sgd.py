import numpy as np
import random

"""
    This program demonstrate the process of matrix factorization with stotastic gradient decent technique
"""

# Create the matrix factorization network 
x = np.eye(3)
v = np.asarray([[1.0, 3.0, 2.0], [2.0, 4.0, 1.0]], dtype=float)
w = np.asarray([[2.0], [2.0]], dtype=float)
y = np.asarray([[10, 5, 7]])

# Learning parameters
eta = 0.0001126
iterators = 100000


def printFormula(i, n, m, v_n, w_m, residual):
    """
        Print the detail of matrix factorization network 

        Arg:    i           - The index of the iterator
                n           - The random index of input vector
                m           - The random index of output vector
                v_n         - The specific transform weight toward n
                w_m         - The combination weight toward m
                residual    - The difference between ground-truth and prediction
    """
    print '\n------<time: ', i+1, '>------------'
    print 'pick n=', n, '\tm=', m
    print '\nV', n, '='
    print v_n
    print '\nW', m, '^T='
    print w_m
    print 'residual: ', residual
    print '\nV', n, '\'= V', n, '+ eta * r', n, m, '* W', m, '='
    print v_n +  eta * residual * np.expand_dims(w_m, -1)
    print '\nW', m, '\'= W', m, '+ eta * r', n, m, '* V', n, '='
    print np.expand_dims(w_m, -1) +  eta * residual * v_n

# Stotastic gradient decent process
print "--< Matrix Factorization >--"
for i in range(iterators):

    # Determine the index of input and output
    n, m = np.shape(y)[1], np.shape(y)[0]
    n_random = random.randint(0, n-1)
    m_random = random.randint(0, m-1)

    # Compute the residual and new weights
    residual = y[m_random][n_random] - np.matmul(w.T[m_random], np.expand_dims(v.T[n_random], -1))
    _v = v[:, n_random] + (eta * residual * w[:, m_random])
    _w = w[:, m_random] + (eta * residual * v[:, n_random])

    # Print detail (You should uncomment the next line)
    #printFormula(i, n_random, m_random, np.expand_dims(v.T[n_random], -1), w[:, m_random], residual)

    # Update weight and print error rate
    v[:, 0] = _v
    w[:, m_random] = _w
    err_rate = np.sum(np.square(y - np.matmul(w.T, v)))
    if i % 10000 == 0:
        print 'err: ', err_rate

# print the final prediction
print 'ground-truth: ', y
print 'prediction  : ', np.matmul(w.T, v)