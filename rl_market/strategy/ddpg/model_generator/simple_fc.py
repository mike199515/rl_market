import rl_market.utils.logging_conf
from keras.layers import Input, Dense, merge
from keras.models import Model
from .base import ModelGenerator
class SimpleFCCritic(ModelGenerator):
    def __init__(self, h1=300, h2=600):
        self.h1=h1
        self.h2=h2

    def generate_critic(self, state_size, action_size, optimizer, LEARNING_RATE):
        state =  Input(shape=[state_size])
        w1 = Dense(self.h1, activation="relu")(state)
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

    def generate_actor(self, state_size, action_size):
        inp =  Input(shape=[state_size])
        h1 = Dense(self.h1, activation="relu")(inp)
        h2 = Dense(self.h2, activation="relu")(h1)
        out = Dense(action_size, activation="softmax")(h2)

        model = Model(input=inp, output=out)
        return model, model.trainable_weights, inp
