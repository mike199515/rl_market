import rl_market.utils.logging_conf
from keras.layers import Input, GRU, Dense, Activation
from keras.layers import TimeDistributed, Bidirectional, Permute, Reshape, RepeatVector, Flatten
from keras.layers.merge import Concatenate
from keras.models import Model
import tensorflow as tf
from keras.engine.topology import Layer
from .base import ModelGenerator

class ReduceSum(Layer):
    def __init__(self, axis, **kwargs):
        self.name = "ReduceSum"
        self.axis = axis
        super(ReduceSum,self).__init__(**kwargs)

    def build(self, input_shape):
        super(ReduceSum,self).build(input_shape)

    def call(self, x):
        print("call reduce sum")
        return tf.reduce_sum(x, self.axis, keep_dims=True)

    def compute_output_shape(self, input_shape):
        print("input_shape:",input_shape)

        out_shape = input_shape[:self.axis]+(1,) + input_shape[self.axis+1:]
        print("output_shape:",out_shape)
        return out_shape

class GRUModel(ModelGenerator):
    def __init__(self, bh = 128, ih = 256, eh = 512):
        self.bh = bh
        self.ih = ih
        self.eh = eh
        pass

    def generate_individual_model(self, nr_day, nr_feature):
        inp = Input(shape = (nr_day, nr_feature))
        ind_feat = Bidirectional(GRU(self.ih))(inp) #(batch, ih)
        model = Model(inputs=[inp], outputs=[ind_feat])
        return model

    def generate_eval_model(self, nr_feature):
        inp = Input(shape = (nr_feature,))
        dense_feat = Dense(self.eh, activation="relu")(inp) #(batch, ih)
        out = Dense(1, activation="linear")(dense_feat)
        model = Model(inputs=[inp], outputs=[out])
        return model

    def generate_actor(self, state_shape, action_size):
        assert(len(state_shape)==3),"shape mismatch"
        nr_day, nr_feature, nr_seller = state_shape
        assert(action_size == nr_seller)
        inp = Input(shape = (nr_day,nr_feature,nr_seller))

        #Background part, assume seller data ordered
        dsf_inp = Permute((1,3,2))(inp)
        reshape_inp = Reshape((nr_day,nr_seller * nr_feature))(dsf_inp)
        background_feat = Bidirectional(GRU(self.bh))(reshape_inp) # (batch, bh)
        repeated_background_feat = RepeatVector(nr_seller)(background_feat) #(batch, nr_seller, bh)
        #individual part
        sdf_inp = Permute((3,1,2))(inp)
        ind_model = self.generate_individual_model(nr_day, nr_feature)
        individual_feat = TimeDistributed(ind_model)(sdf_inp) #(batch, nr_seller, ih)

        concat_feat = Concatenate(axis=-1)([repeated_background_feat, individual_feat]) #(batch, nr_seller, ih+bh)

        eval_model  = self.generate_eval_model(2*(self.ih+self.bh))
        out = TimeDistributed(eval_model)(concat_feat) #(batch, nr_seller, 1)
        out_flat = Flatten()(out)

        out_softmax = Activation("softmax")(out_flat)

        model = Model(inputs=[inp], outputs=[out_softmax])
        return model, model.trainable_weights, inp

    def generate_critic(self, state_shape, action_size, optimizer, LEARNING_RATE):
        assert(len(state_shape)==3),"shape mismatch"
        nr_day, nr_feature, nr_seller = state_shape
        assert(action_size == nr_seller)
        inp = Input(shape=(nr_day, nr_feature, nr_seller))
        print("inp shape=",inp.shape)
        action =  Input(shape=(nr_seller,))

        #Similarly, first background part, ASSUME seller data ordered
        dsf_inp = Permute((1,3,2))(inp)
        reshape_inp = Reshape((nr_day,nr_seller * nr_feature))(dsf_inp)
        background_feat = Bidirectional(GRU(self.bh))(reshape_inp) # (batch, bh)
        repeated_background_feat = RepeatVector(nr_seller)(background_feat) #(batch, nr_seller, bh)
        #individual part
        sdf_inp = Permute((3,1,2))(inp)
        ind_model = self.generate_individual_model(nr_day, nr_feature)
        individual_feat = TimeDistributed(ind_model)(sdf_inp) #(batch, nr_seller, ih)

        eval_model  = self.generate_eval_model(2*(self.ih+self.bh)+1)
        concat_feat = Concatenate(axis=-1)([repeated_background_feat, individual_feat, Reshape((-1,1))(action)]) #(batch, nr_seller, ih+bh)
        print("concat_feat shape = ",concat_feat.get_shape())
        values = TimeDistributed(eval_model)(concat_feat) #(batch, nr_seller ,1)
        print("values shape = ",values.get_shape())
        flat_values = Reshape((-1,))(values)
        #then reduce to one value by addition
        value = ReduceSum(axis = 1)(flat_values) #(batch,)
        #value  = Dense(1, activation="linear")(flat_values)
        print("value shape = ",value.get_shape())
        model = Model(inputs=[inp,action], outputs=[value])
        opt = optimizer(LEARNING_RATE)
        model.compile(loss="mse", optimizer=opt)
        return model, action, inp
