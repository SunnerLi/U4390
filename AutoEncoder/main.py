from AutoEncoder import AutoEncoder
import numpy as np

x = [
        [[-1], [1], [1], [1]],
        [[1], [1], [1], [1]],
    ]
x = np.asarray(x)

# Build the auto-encoder
auto_encoder = AutoEncoder([3, 2], eta=0.05)
auto_encoder.assignX(x)
weights = auto_encoder.fit()

# Print the parameters
for i in range(len(weights)):
    print weights[i]