import numpy as np

"""
    This program demonstrate the process of matrix factorization with alternating least square algorithm
"""

iterators = 200

def printDetail(i, w, y):
    """
        Print the detail of matrix factorization network

        Arg:    i   - The index of the iterator
                w   - The feature combination weight 
                y   - The ground-truth vector
    """
    print "------<iter: ", i, ' >-------'
    print 'v = w * y ='
    print w, '*', y, '='
    print np.matmul(w, y), '\n'
    print 'w = {y * [v]\'}\' ='
    print '{', y, '*', np.linalg.pinv(np.matmul(w, y)), '}^T = '
    print np.matmul(y, np.linalg.pinv(np.matmul(w, y))).T, '\n'

# Initialize the network
v = np.asarray([[1, 0, 1], [0, 1, 0]])
w = np.asarray([[2], [3]])
x = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y = np.asarray([[1, 2, 3]])

# Another more complex network
"""
v = np.asarray([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
w = (np.random.random([3, 4]) - 0.5) * 100  
x = np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
y = np.asarray([[90, 50, 75, 20, 100], [85, 50, 45, 30, 100], [90, 60, 75, 20, 100], [80, 60, 75, 40, 100]])
"""

# Tango iteration!
for i in range(iterators):
    # Print detail (You should uncomment the next line)
    #printDetail(i, w, y)

    # Update weight
    v = np.matmul(w, y)
    w = np.matmul(y, np.linalg.pinv(np.matmul(w, y))).T
    print 'error: ', np.sum(np.square(y - np.matmul(w.T, v)))