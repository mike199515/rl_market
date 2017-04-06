import rl_market.utils.logging_conf
from keras.layers import Input, GRU, Dense
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.merge import Concatenate, Merge
from keras.models import Model
import keras.backend as K

from .base import ModelGenerator

class GRUModel(ModelGnerator):
    def __init__(self, bh = 128, ih = 128, eh = 128):
        self.bh = bh
        self.ih = ih
        self.eh = eh
        pass
    
    def generate_individual_model(self, nr_day, nr_feature):
        inp = Input(shape = (nr_day, nr_feature))
        ind_feat = Bidirectional(GRU(self.ih)(inp)) #(batch, ih)
        model = Model(inputs=[inp], outputs=[ind_feat])
        return model

    def generate_eval_model(self, nr_feature):
        inp = Input(shape = (nr_feature,))
        dense_feat = Dense(self.eh)(inp) #(batch, ih)
        out = Dense(1)(dense_feat)
        model = Model(inputs=[inp], outputs=[out])
        return model
        
    def generate_actor(self, nr_seller, nr_day, nr_feature):
        inp = Input(shape = (nr_seller,nr_day,nr_feature))
        
        #Background part, assume seller data ordered 
        dsf_inp = Permute((2,1,3))(inp)
        reshape_inp = Reshape((nr_day,nr_seller * nr_feature))(dsf_inp)
        background_feat = Bidirectional(GRU(self.bh)(reshape_inp)) # (batch, bh)
        repeated_background_feat = RepeatVector(nr_seller)(background_feat) #(batch, nr_seller, bh)
        #individual part
        ind_model = self.generate_individual_model(nr_day, nr_feature)
        individual_feat = Timedistributed(ind_model)(inp) #(batch, nr_seller, ih)
        
        concat_feat = Concatenate([repeated_background_feat, individual_feat],axis=-1) #(batch, nr_seller, ih+bh)
        
        eval_model  = self.generate_eval_model(nr_feature)
        out = Timedistributed(eval_model)(concate_feat) #(batch, nr_seller)
        
        model = Model(input=[inp], output=out)
        return model, model.trainable_weights, inp
        
    def generate_critic(self, nr_seller, nr_day, nr_feature, optimizer, LEARNING_RATE):
        state = Input(shape=(nr_seller, nr_day, nr_feature))
        action =  Input(shape=(nr_seller,))
        
        #Similarly, first background part, ASSUME seller data ordered
        dsf_inp = Permute((2,1,3))(inp)
        reshape_inp = Reshape((nr_day,nr_seller * nr_feature))(dsf_inp)
        background_feat = Bidirectional(GRU(self.bh)(reshape_inp)) # (batch, bh)
        repeated_background_feat = RepeatVector(nr_seller)(background_feat) #(batch, nr_seller, bh)
        #individual part
        ind_model = self.generate_individual_model(nr_day, nr_feature)
        individual_feat = Timedistributed(ind_model)(inp) #(batch, nr_seller, ih)
        
        concat_feat = Concatenate([repeated_background_feat, individual_feat, action],axis=-1) #(batch, nr_seller, ih+bh)
        values = Timedistributed(eval_model)(concate_feat) #(batch, nr_seller)
        #then reduce to one value by addition
        value = K.reduce_sum(values, 1) #(batch,)
        
        model = Model(input=[state,action], output=value)
        opt = optimizer(LEARNING_RATE)
        model.compile(loss="mse", optimizer=opt)
        return model, action, state