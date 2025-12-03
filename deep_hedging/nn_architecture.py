# nn_architecture.py

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform, HeNormal


def build_hedging_networks(N, num_layers, num_neurons, num_outputs):
    layers = []
    for n in range(N):
        for i in range(num_layers+1):
            if i < num_layers:
                layer = Dense(
                    units=num_neurons,
                    activation='tanh',
                    kernel_initializer=GlorotUniform(),
                    name=str(i)+str(n)
                )
            else:
                layer = Dense(
                    units=num_outputs,
                    activation='relu',  # or softplus / custom
                    kernel_initializer=HeNormal(),
                    name=str(i)+str(n)
                )
            layers.append(layer)
    return layers
 
