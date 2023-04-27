#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2020 The TensorFlow Authors.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # MNIST classification

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/quantum/tutorials/mnist"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/mnist.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/quantum/blob/master/docs/tutorials/mnist.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/quantum/docs/tutorials/mnist.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This tutorial builds a quantum neural network (QNN) to classify a simplified version of MNIST, similar to the approach used in <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al</a>. The performance of the quantum neural network on this classical data problem is compared with a classical neural network.

# ## Setup

# In[ ]:


#get_ipython().system('pip install tensorflow==2.7.0')



# Install TensorFlow Quantum:

# In[ ]:


#get_ipython().system('pip install tensorflow-quantum==0.7.2')


# In[ ]:


# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)


# Now import TensorFlow and the module dependencies:

# In[ ]:


import tensorflow as tf
import tensorflow_quantum as tfq
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import cirq
import sympy
import numpy as np
import seaborn as sns
import collections
import os
# visualization tools
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


# ## 1. Load the data
# 
# In this tutorial you will build a binary classifier to distinguish between the digits 3 and 6, following <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> This section covers the data handling that:
# 
# - Loads the raw data from Keras.
# - Filters the dataset to only 3s and 6s.
# - Downscales the images so they fit can fit in a quantum computer.
# - Removes any contradictory examples.
# - Converts the binary images to Cirq circuits.
# - Converts the Cirq circuits to TensorFlow Quantum circuits. 

# ### 1.1 Load the raw data

# Load the MNIST dataset distributed with Keras. 
G = ["5"]#"4", "5", "6", "7"]#, "1"]#, "2", "3", "4", "5", "6", "7", "8", "9"]#,"1","2","3"]
stringGPUs = []
for i in G:
    GPS = "GPU:" + i
    stringGPUs.append(GPS)
print(stringGPUs)
print(",".join(G))
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(G)

if len(stringGPUs) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=stringGPUs)#, cross_device_ops=tf.distribute.NcclAllReduce())
else:
    strategy = tf.distribute.OneDeviceStrategy(device=stringGPUs[0])

print("Number of devices: {}".format(strategy.num_replicas_in_sync))



input_shape_class = (28,28,1)
Q = 16  # use Q values when np.sqrt(Q) = interger numbers  
#DI = [int(np.sqrt(Q))]
DI = [4,4]
#print(DI)
EPOCHS = 3
BATCH_SIZE = 32
THRESHOLD = 0.5
EXP = "ResultsTestMnistCIAandQIA_Q" + str(Q)
pathsave = os.path.join(os.getcwd(), EXP)
if not os.path.exists(pathsave):
    os.mkdir(pathsave)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))


# Filter the dataset to keep just the 3s and 6s,  remove the other classes. At the same time convert the label, `y`, to boolean: `True` for `3` and `False` for 6. 

# In[ ]:


def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y


# In[ ]:


x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

print("Number of filtered training examples:", len(x_train))
print("Number of filtered test examples:", len(x_test))


# Show the first example:

# In[ ]:


print(y_train[0])
plt.figure()
plt.imshow(x_train[0, :, :, 0])
plt.colorbar()
plt.savefig(os.path.join(pathsave,"PlotMnistSaved.png"))

# ### 1.2 Downscale the images

# An image size of 28x28 is much too large for current quantum computers. Resize the image down to 4x4:

# In[ ]:


x_train_small = tf.image.resize(x_train, (DI[0],DI[0])).numpy()
x_test_small = tf.image.resize(x_test, (DI[0],DI[1])).numpy()


# Again, display the first training example—after resize: 

# In[ ]:


print(y_train[0])
plt.figure()
plt.imshow(x_train_small[0,:,:,0], vmin=0, vmax=1)
plt.colorbar()
plt.savefig(os.path.join(pathsave,"PlotMnistSavedResizeSmall.png"))

# ### 1.3 Remove contradictory examples

# From section *3.3 Learning to Distinguish Digits* of <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a>, filter the dataset to remove images that are labeled as belonging to both classes.
# 
# This is not a standard machine-learning procedure, but is included in the interest of following the paper.

# In[ ]:


def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass
    
    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num_uniq_3)
    print("Number of unique 6s: ", num_uniq_6)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))
    
    return np.array(new_x), np.array(new_y)


# The resulting counts do not closely match the reported values, but the exact procedure is not specified.
# 
# It is also worth noting here that applying filtering contradictory examples at this point does not totally prevent the model from receiving contradictory training examples: the next step binarizes the data which will cause more collisions. 

# In[ ]:


x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)


# ### 1.4 Encode the data as quantum circuits
# 
# To process images using a quantum computer, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> proposed representing each pixel with a qubit, with the state depending on the value of the pixel. The first step is to convert to a binary encoding.

# In[ ]:



x_train_bin = np.array(x_train_nocon > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)


# If you were to remove contradictory images at this point you would be left with only 193, likely not enough for effective training.

# In[ ]:


_ = remove_contradicting(x_train_bin, y_train_nocon)


# The qubits at pixel indices with values that exceed a threshold, are rotated through an $X$ gate.

# In[ ]:


def convert_to_circuit(image,DI):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(DI[0], DI[1])
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


x_train_circ = [convert_to_circuit(x,DI) for x in x_train_bin]
x_test_circ = [convert_to_circuit(x,DI) for x in x_test_bin]


# Here is the circuit created for the first example (circuit diagrams do not show qubits with zero gates):

# In[ ]:


SVGCircuit(x_train_circ[0])


# Compare this circuit to the indices where the image value exceeds the threshold:

# In[ ]:


bin_img = x_train_bin[0,:,:,0]
indices = np.array(np.where(bin_img)).T
indices


# Convert these `Cirq` circuits to tensors for `tfq`:

# In[ ]:


x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


# ## 2. Quantum neural network
# 
# There is little guidance for a quantum circuit structure that classifies images. Since the classification is based on the expectation of the readout qubit, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> propose using two qubit gates, with the readout qubit always acted upon. This is similar in some ways to running small a <a href="https://arxiv.org/abs/1511.06464" class="external">Unitary RNN</a> across the pixels.

# ### 2.1 Build the model circuit
# 
# This following example shows this layered approach. Each layer uses *n* instances of the same gate, with each of the data qubits acting on the readout qubit.
# 
# Start with a simple class that will add a layer of these gates to a circuit:

# In[ ]:


class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)


# Build an example circuit layer to see how it looks:

# In[ ]:


demo_builder = CircuitLayerBuilder(data_qubits = cirq.GridQubit.rect(DI[0],1),
                                   readout=cirq.GridQubit(-1,-1))

circuit = cirq.Circuit()
demo_builder.add_layer(circuit, gate = cirq.XX, prefix='xx')
SVGCircuit(circuit)


# Now build a two-layered model, matching the data-circuit size, and include the preparation and readout operations.

# In[ ]:


def create_quantum_model(DI):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(DI[0], DI[1])  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)


# In[ ]:

print("Create layer data-circuit size quantum model ........")
#with strategy.scope():
model_circuit, model_readout = create_quantum_model(DI)


# ### 2.2 Wrap the model-circuit in a tfq-keras model
# 
# Build the Keras model with the quantum components. This model is fed the "quantum data", from `x_train_circ`, that encodes the classical data. It uses a *Parametrized Quantum Circuit* layer, `tfq.layers.PQC`, to train the model circuit, on the quantum data.
# 
# To classify these images, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> proposed taking the expectation of a readout qubit in a parameterized circuit. The expectation returns a value between 1 and -1.

# In[ ]:


# Build the Keras model.
#print("Create model tfq quantum model ........")
#with strategy.scope():
#    model = tf.keras.Sequential([
#    # The input is the data-circuit, encoded as a tf.string
#    tf.keras.layers.Input(shape=(), dtype=tf.string),
#    # The PQC layer returns the expected value of the readout gate, range [-1,1].
#    tfq.layers.PQC(model_circuit, model_readout),
#    ])


# Next, describe the training procedure to the model, using the `compile` method.
# 
# Since the the expected readout is in the range `[-1,1]`, optimizing the hinge loss is a somewhat natural fit. 
# 
# Note: Another valid approach would be to shift the output range to `[0,1]`, and treat it as the probability the model assigns to class `3`. This could be used with a standard a `tf.losses.BinaryCrossentropy` loss.
# 
# To use the hinge loss here you need to make two small adjustments. First convert the labels, `y_train_nocon`, from boolean to `[-1,1]`, as expected by the hinge loss.

# In[ ]:


y_train_hinge = 2.0*y_train_nocon-1.0
y_test_hinge = 2.0*y_test-1.0


# Second, use a custiom `hinge_accuracy` metric that correctly handles `[-1, 1]` as the `y_true` labels argument. 
# `tf.losses.BinaryAccuracy(threshold=0.0)` expects `y_true` to be a boolean, and so can't be used with hinge loss).

# In[ ]:


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


# In[ ]:
print("Create model tfq quantum model ........")
with strategy.scope():
    model = tf.keras.Sequential([
                # The input is the data-circuit, encoded as a tf.string
                    tf.keras.layers.Input(shape=(), dtype=tf.string),
                        # The PQC layer returns the expected value of the readout gate, range [-1,1].
                            tfq.layers.PQC(model_circuit, model_readout),
                                ])

    model.compile(
    loss=tf.keras.losses.Hinge(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[hinge_accuracy])


# In[ ]:


print(model.summary())


# ### Train the quantum model
# 
# Now train the model—this takes about 45 min. If you don't want to wait that long, use a small subset of the data (set `NUM_EXAMPLES=500`, below). This doesn't really affect the model's progress during training (it only has 32 parameters, and doesn't need much data to constrain these). Using fewer examples just ends training earlier (5min), but runs long enough to show that it is making progress in the validation logs.

# In[ ]:



NUM_EXAMPLES = len(x_train_tfcirc)


# In[ ]:


x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]


# Training this model to convergence should achieve >85% accuracy on the test set.

# In[ ]:

#dataset_train_tfcirc_sub = tf.data.Dataset.from_tensor((x_train_tfcirc_sub, y_train_hinge_sub))
#dataset_test_tfcirc_sub = tf.data.Dataset.from_tensor((x_test_tfcirc, y_test_hinge))

dataset_train_tfcirc_sub = x_train_tfcirc_sub, y_train_hinge_sub
dataset_test_tfcirc_sub = (x_test_tfcirc, y_test_hinge)
print(y_test.shape)
print(y_test.max())


qnn_history = model.fit(
      x_train_tfcirc_sub, y_train_hinge_sub,
      batch_size=32,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc, y_test_hinge),
      workers=8,
      use_multiprocessing=True)

qnn_results = model.evaluate(x_test_tfcirc, y_test)
predQIA = model.predict(x_test_tfcirc, batch_size=32, workers=8,
      use_multiprocessing=True)

# Note: The training accuracy reports the average over the epoch. The validation accuracy is evaluated at the end of each epoch.

# ## 3. Classical neural network
# 
# While the quantum neural network works for this simplified MNIST problem, a basic classical neural network can easily outperform a QNN on this task. After a single epoch, a classical neural network can achieve >98% accuracy on the holdout set.
# 
# In the following example, a classical neural network is used for for the 3-6 classification problem using the entire 28x28 image instead of subsampling the image. This easily converges to nearly 100% accuracy of the test set.

# In[ ]:


def create_classical_model(input_shape):
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
    return model

input_shape = input_shape_class


if len(stringGPUs) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=stringGPUs)#, cross_device_ops=tf.distribute.NcclAllReduce())
else:
    strategy = tf.distribute.OneDeviceStrategy(device=stringGPUs[0])

print("Number of devices: {}".format(strategy.num_replicas_in_sync))


with strategy.scope():
    model = create_classical_model(input_shape)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(x_train,
          y_train,
          batch_size=128,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test),workers=8,use_multiprocessing=True)

cnn_results = model.evaluate(x_test, y_test)
predCIA28x28 = model.predict(x_test, batch_size=128, workers=8,use_multiprocessing=True)

# The above model has nearly 1.2M parameters. For a more fair comparison, try a 37-parameter model, on the subsampled images:

# In[ ]:


def create_fair_classical_model(DI):
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(DI[0],DI[1],1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


if len(stringGPUs) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=stringGPUs)#, cross_device_ops=tf.distribute.NcclAllReduce())
else:
    strategy = tf.distribute.OneDeviceStrategy(device=stringGPUs[0])

print("Number of devices: {}".format(strategy.num_replicas_in_sync))


with strategy.scope():
    model = create_fair_classical_model(DI)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(x_train_bin,
          y_train_nocon,
          batch_size=128,
          epochs=20,
          verbose=2,
          validation_data=(x_test_bin, y_test),workers=8,use_multiprocessing=True)

fair_nn_results = model.evaluate(x_test_bin, y_test)
predCIA = model.predict(x_test_bin, batch_size=128, workers=8,use_multiprocessing=True)

# ## 4. Comparison
# 
# Higher resolution input and a more powerful model make this problem easy for the CNN. While a classical model of similar power (~32 parameters) trains to a similar accuracy in a fraction of the time. One way or the other, the classical neural network easily outperforms the quantum neural network. For classical data, it is difficult to beat a classical neural network.

# In[ ]:


qnn_accuracy = qnn_results[1]
cnn_accuracy = cnn_results[1]
fair_nn_accuracy = fair_nn_results[1]
plt.figure()
sns.barplot(x=["Quantum " + str(DI[0]) + "x" + str(DI[1]), "Classical 28x28", "Classical " + str(DI[0]) + "x" + str(DI[1])],
            y=[qnn_accuracy, cnn_accuracy, fair_nn_accuracy])
plt.savefig(os.path.join(pathsave,"plotCompareClassicalIAandQIA.png"))


#################### ROCs
ListPred = [predQIA,predCIA28x28,predCIA]
ListY = [y_test_hinge, y_test, y_test]

roc_data = dict()
auc = dict()
for i in range(len(ListPred)):
    roc_data[i] = metrics.roc_curve(ListY[i], ListPred[i])
    auc[i] = metrics.auc(roc_data[i][0], roc_data[i][1])

f, ax = plt.subplots(figsize=[9, 6])
ax.plot(roc_data[0][0], roc_data[0][1], 'k-', label='QIA, AUC = {:4.2f}'.format(auc[0]))
ax.plot(roc_data[1][0], roc_data[1][1], 'b-', label='CIA28x28, AUC = {:4.2f}'.format(auc[1]))
ax.plot(roc_data[2][0], roc_data[2][1], 'r-', label='CIA, AUC = {:4.2f}'.format(auc[2]))
# ax.plot([0, 1], [0, 1], 'g--')
ax.legend(loc='lower right')
f.savefig(os.path.join(pathsave, "RocCompare.png"))


cf_matrix = dict()
for i in range(len(ListPred)):
    cf_matrix[i] = confusion_matrix(ListY[i], ListPred[i])
    print(cf_matrix[i])

kkkkk

f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, 
                    gridspec_kw={'width_ratios':[1,1,1,0.08]})
ax1.get_shared_y_axes().join(ax2,ax3)
g1 = sns.heatmap(cf_matrix[0], annot=True, cbar=True, ax=ax1)
g2 = sns.heatmap(cf_matrix[1], annot=True, cbar=True, ax=ax2)
g3 = sns.heatmap(cf_matrix[2], annot=True, ax=ax3, cbar_ax=axcb)

for ax in [g1,g2,g3]:
    tl = ax.get_xticklabels()
    ax.set_xticklabels(tl, rotation=90)
    tly = ax.get_yticklabels()
    ax.set_yticklabels(tly, rotation=0)

f.savefig(os.path.join(pathsave, "MCCompare.png"))
