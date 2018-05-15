import numpy as np
#import gzip
#from struct import unpack
import time


class NeuralNet():
    def __init__(self):
        self.layers = []

    # Rectified Linear Unit activation
    def relu(self, x):
        return np.maximum(x, 0)

    # Softmax activation with a shift towards 0 using the maximum value in the input vector
    # To Avoid overflow (infinity) in exp.
    def softmax(self, x):
        norm = x - np.max(x)
        return np.exp(norm) / np.sum(np.exp(x), axis=0)

    # Used to flatten matrices and combine channels so that they can be passed into fully connected layers.
    def flatten(self, x):
        dim = x[0][0].shape[0]
        flattened = []
        for i in range(len(x)):
            flat = x[i][0].reshape(dim**2)
            for j in range(1, len(x[0])):
                flat = np.concatenate((flat, x[i][j].reshape(dim ** 2)))
            flattened.append(flat)
        return flattened

    # 2x2 max pooling
    def max_pool(self, conv_mat):
        output = []
        # Calculate the dimensions of max pooled matrix
        conv_dim = conv_mat[0][0].shape[0]
        pool_dim = conv_dim//2
        dropped = conv_dim%2

        for sample in conv_mat:
            # List to contain all pooled matrices for a given sample
            pooled = []
            for channel in sample:
                # Channel specific matrix
                pooled_channel = []
                # For row in the convolved matrix
                for i in range(0, conv_dim - dropped, 2):
                    # For column in the convolved matrix
                    for j in range(0, conv_dim - dropped, 2):
                        # Find maximum value of 2X2 submatrix
                        pooled_channel.append(channel[i:i+2, j:j+2].max())
                pooled.append(np.asarray(pooled_channel).reshape(pool_dim, pool_dim))
            output.append(pooled)
        return output

    # Convolutional Layer
    def cnn(self, x, filter_size, size_in, size_out, strides=1):
        # Initiate filters of correct size, matching the depth of the input and bias matching output depth
        w = [[np.random.normal(0, 0.1, filter_size**2).reshape(filter_size, filter_size) for i in range(size_in)] for j in range(size_out)]
        b = np.full(size_out, 0.1)

        # Calculate the size of the output matrix. Assume matrix is square.
        conv_dim = x[0][0].shape[0] - filter_size + 1

        # Holds convolved matrices for all samples
        convolved = []
        # For each sample in the input
        for sample in x:
            v = [[] for d in range(size_out)]
            # For row in the convolved matrix
            for i in range(conv_dim):
                # For column in the convolved matrix
                for j in range(conv_dim):
                    # For all input channels, store the submatrices of the input matrix that have the same coordinates
                    sample_sub = []
                    for layer in sample:
                        sample_sub.append(layer[i:i+filter_size, j:j+filter_size])

                    # Multiples the filter/weight for each channel by all submatrices in the channel
                    for filters in range(len(w)):
                        v_temp = np.matmul(sample_sub, w[filters])
                        # Sums across all channels all calculated matrices that correspond to the same coordinates
                        v_out = np.sum(v_temp)
                        v[filters].append(v_out)

            # Add bias to all calculated matrices
            biased = [v[i] + b[i] for i in range(size_out)]
            biased = [np.asarray(biased[i]).reshape(conv_dim, conv_dim) for i in range(size_out)]
            convolved.append(biased)

        return convolved

    # Fully connected layer
    def fc_layer(self, input, size_in, size_out):
        # Initialize weight and bias
        w = np.random.normal(0, 0.1, size_in*size_out).reshape(size_in, size_out)
        b = np.full(size_out, 0.1)

        act = np.matmul(input, w) + b

        return act

    # Cross Entropy
    def xent(self, labels, calc):
        log_likelihood = -np.log(calc, labels)
        loss = np.sum(log_likelihood) / 100
        return loss




# Load in MNIST data
inFile = np.load('../mnistDataNumpy.npz')

start_time = time.time()

# Create list of all training samples and their corresponding label
inputs = []
inLabels = []
i = 0
for sample in inFile['train']:
    inputs.append([sample])

# One-hot encode label values
for label in inFile['trainLabel']:
    y = np.zeros(10)
    y[int(label)] = 1
    inLabels.append(y)





convNN = NeuralNet()

conv_layer = convNN.cnn(inputs[:100], 10, 1, 32)
print("Conv Layer (minutes): " + str((time.time() - start_time)/60))
act = convNN.relu(conv_layer)
print("Activation Layer (minutes): " + str((time.time() - start_time)/60))
pool = convNN.max_pool(act)
print("Pooling Layer (minutes): " + str((time.time() - start_time)/60))


conv_layer2 = convNN.cnn(pool, 5, 32, 16)
print("Conv Layer 2 (minutes): " + str((time.time() - start_time)/60))
act2 = convNN.relu(conv_layer2)
print("Activation Layer 2 (minutes): " + str((time.time() - start_time)/60))
pool2 = convNN.max_pool(act2)
print("Pooling Layer 2 (minutes): " + str((time.time() - start_time)/60))


flattened = convNN.flatten(pool2)
print("Flattening (minutes): " + str((time.time() - start_time)/60))


print("flattened[0]:",  flattened[0])
fc1 = convNN.fc_layer(flattened, len(flattened[0]), 1024)
print("Fully Connected 1 (minutes): " + str((time.time() - start_time)/60))
act_fc = convNN.relu(fc1)
print("Activation Layer - FC (minutes): " + str((time.time() - start_time)/60))
fc2 = convNN.fc_layer(act_fc, len(act_fc[0]), 10)
print("Fully Connected 2 (minutes): " + str((time.time() - start_time)/60))
act_fc2 = convNN.softmax(fc2)
print("Activation Layer - FC2 (minutes): " + str((time.time() - start_time)/60))
#print(act_fc2)
#print(act_fc2[0])


loss = convNN.xent(inLabels[:100], act_fc2)
print("Cross Entropy (minutes): " + str((time.time() - start_time)/60))
#print(act_fc2)
#print(act_fc2[0])



'''
# Code below pulls image and label data from .gz files, formats them in np.arrays,
# and saves all np.arrays into a file with compressed .npz format

def read_img(file):
    with gzip.open('../mnist_data/' + file, 'rb') as img:
        img.read(4)
        num_img = unpack('>I', img.read(4))[0]
        rows = unpack('>I', img.read(4))[0]
        cols = unpack('>I', img.read(4))[0]
        x = np.zeros((num_img, rows, cols))

        for i in range(num_img):
            for row in range(rows):
                for col in range(cols):
                    pixel = unpack('>B', img.read(1))[0]
                    x[i][row][col] = pixel
        return x

def read_label(file):
    with gzip.open('../mnist_data/' + file, 'rb') as label:
        label.read(4)
        num_labels = unpack('>I', label.read(4))[0]
        y = np.zeros((num_labels, 1))

        for i in range(num_labels):
            lbl = unpack('>B', label.read(1))[0]
            y[i] = lbl
    return y


img_file = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']

mnistTrain = read_img(img_file[0])
mnistTest = read_img(img_file[1])

label_file = ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

mnistTrainLabel = read_label(label_file[0])
mnistTestLabel = read_label(label_file[1])

np.savez_compressed('../mnistDataNumpy', train = mnistTrain, trainLabel = mnistTrainLabel, test = mnistTest, testLabel = mnistTestLabel)
'''