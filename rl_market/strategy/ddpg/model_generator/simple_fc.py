import rl_market.utils.logging_conf
from keras.layers import Input, Dense, Flatten, merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from .base import ModelGenerator
class SimpleFCCritic(ModelGenerator):
    def __init__(self, h1=300, h2=600):
        self.h1=h1
        self.h2=h2

    def generate_critic(self, state_shape, action_size, optimizer, LEARNING_RATE):
        state =  Input(shape=state_shape)
        #normalized_state = BatchNormalization(epsilon=1e-6, mode = 0, axis = 2)(state) # along value axis
        #flat_state = Flatten()(normalized_state)
        flat_state = Flatten()(state)
        w1 = Dense(self.h1, activation="relu")(flat_state)
        h1 = Dense(self.h2, activation="relu")(w1)

        action =  Input(shape=[action_size], name = "action2")
        a1 = Dense(self.h2, activation="linear")(action)

        h2 = merge([h1,a1],mode="sum")

        h3 = Dense(self.h2, activation="relu")(h2)
        value = Dense(action_size, activation="linear")(h3)

        model = Model(input=[state,action], output=value)
        opt = optimizer(LEARNING_RATE)
        model.compile(loss="mse", optimizer=opt)
        return model, action, state


class SimpleFCAction(ModelGenerator):
    def __init__(self, h1=300, h2=600):
        self.h1=h1
        self.h2=h2

    def generate_actor(self, state_shape, action_size):

        inp =  Input(shape=state_shape)
        #normalized_inp = BatchNormalization(epsilon=1e-6, mode = 0, axis = 2)(inp)
        #flat_inp = Flatten()(normalized_inp)
        flat_inp = Flatten()(inp)
        h1 = Dense(self.h1, activation="relu")(flat_inp)
        h2 = Dense(self.h2, activation="relu")(h1)
        out = Dense(action_size, activation="softmax")(h2)

        model = Model(input=[inp], output=out)
        return model, model.trainable_weights, inp
