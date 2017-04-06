
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_shape, action_size, model_generator, optimizer, BATCH_SIZE, TAU, LEARNING_RATE):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        K.set_session(sess)
        print("state shape={}, action size={}".format(state_shape,action_size))
        #create model
        self.model, self.weights, self.state = model_generator.generate_actor(state_shape,action_size)
        self.target_model, self.target_weights, self.target_state = model_generator.generate_actor(state_shape,action_size)
        self.action_grads = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_grads)
        self.optimize = optimizer(LEARNING_RATE).apply_gradients(zip(self.params_grad, self.weights))
        self.sess.run(tf.global_variables_initializer())

    def train(self,states,action_grads):
        self.sess.run(self.optimize,feed_dict={
            self.state: states,
            self.action_grads: action_grads
        })

    def train_target_network(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i]=self.TAU*actor_weights[i] + (1-self.TAU)*actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

