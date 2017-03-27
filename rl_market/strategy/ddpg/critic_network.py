
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, model_generator, optimizer, BATCH_SIZE, TAU, LEARNING_RATE):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.model_generator=model_generator
        self.optimizer = optimizer
        #create model
        self.model, self.action, self.state = self.model_generator.generate_critic(state_size, action_size, self.optimizer, self.LEARNING_RATE)
        self.target_model, self.target_action, self.target_state = self.model_generator.generate_critic(state_size, action_size, self.optimizer, self.LEARNING_RATE)
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())

    def get_gradient(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def train_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i]=self.TAU*critic_weights[i] + (1-self.TAU)*critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

