
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, model_generator, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        #create model
        self.model, self.action, self.state = self.model_generator(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.model_generator(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.initialize_all_variables())

    def get_gradient(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

        def train_target_network(self):
            critic_weights = self.mdoel.get_weights()
            critic_target_weights = self.target_model.get_weights()
            for i in range(len(critic_weights)):
                critic_target_weights[i]=self.TAU*critic_weights[i] + (1-self.TAU)*critic_target_weights[i]
            self.target_model.sest_weights(critic_target_weights)

