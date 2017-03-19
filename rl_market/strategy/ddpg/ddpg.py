import numpy as np
import tensorflow as tf
from keras import backend as K
from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
#from OU import OU
from ..base import Strategy

class DDPG(Strategy):
    def __init__(self,
            state_dim,
            action_dim,
            sess = None,
            actor_weight_path = None,
            critic_weight_path = None,
            actor_save_path = None,
            critic_save_path = None,
            BUFFER_SIZE = 100000,
            BATCH_SIZE = 32,
            GAMMA = 0.99,           # reward discount factor
            TAU = 0.001,            # target network shift rate
            LRA = 0.0001,           # learning rate for actor
            LRC = 0.001,            # learning rate for critic
            EXPLORE = 100000.,      # explore factor
            random_seed = 42):

        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LRA = LRA
        self.LRC = LRC
        self.EXPLORE = EXPLORE

        np.random.seed(random_seed)

        if sess is not None:
            self.sess = sess
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
        K.set_session(self.sess)

        self.buff = ReplayBuffer(self.BUFFER_SIZE)
        self.actor = ActorNetwork(self.sess, state_dim, action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic =  CriticNetwork(self.sess, state_dim, action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        if actor_weight_path is not None:
            self.actor.model.load_weights(actor_weight_path)
            self.actor.target_model.load_weights(actor_weight_path)
            print("actor model weight loaded from {}".format(actor_weight_path))
        if critic_weight_path is not None:
            self.critic.model.load_weights(critic_weight_path)
            self.critic.target_model.load_weights(critic_weight_path)
            print("critic model weight loaded from {}".format(critic_weight_path))



    def train(self, game, nr_episode = 2000, nr_steps = 100000):
        # NOTE: interact with the stateful game object
        epsilon = 1.
        total_step = 0
        for episode in range(nr_episode):
            game.reset()
            observation = game.get_observation() # should be a numpy array
            state = self._get_state(observation)
            total_reward = 0.
            for step in range(nr_steps):
                epsilon -= 1/self.EXPLORATION
                action, new_state, reward, done = self._perform_action(game, state, epsilon)
                self.buff.add(state, action, reward, new_state, done)
                loss = self._batch_update()

                #update state & show stats
                state = new_state
                print("Episode {} Step {} Action {} Reward {} Loss {}".format(episode,step,action,reward,loss))
                total_reward += reward
                total_step += 1
                if done:
                    break
            if np.mod(episode, 3)==0:
                if self.actor_save_path is not None:
                    self.actor.model.save_weights(self.actor_save_path, overwrite=True)
                if self.critic_save_path is not None:
                    self.critic.model.save_weights(self.critic_save_path, overwrite=True)
            print("total reward @{}-th episode : {}".format(episode, reward))
            print("total step : {}".format(total_step))
        print("train finish")

    def play(self, game):
        #given observation, get action
        observation = game.get_observation()
        state = self._get_state(observation)
        action = self.actor.model.predict([state])[0]
        return action

    def _get_state(self, observation):
        # potential for observation conversion here
        return observation

    def _perform_action(self, game, state, epsilon):
        action = self.actor.model.predict([state])[0]
        #exploration
        for i in range(self.action_dim):
            action[i] += max(epsilon,0) * self._get_noise(action[i])

        reward, done = game.step(action) # NOTE: inside game, it will normalize action if required
        new_observation = game.get_observation()
        new_state = self._get_state(new_observation)
        return action, new_state, reward, done


    def _batch_update(self):
        batch = self.buff.getBatch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        #use critic network to estimate y_t
        y_t = np.zeros_like(actions)
        target_q_values = self.critic.target_model.predict([new_states,self.actor.target_predict(new_states)])
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.GAMMA * target_q_values[k]

        # update actor & critic
        loss = self.critic.model.train_on_batch([states,actions], y_t)
        action_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, action_for_grad)
        self.actor.train(states, grads)
        #update target networks
        self.actor.train_target_network()
        self.critic.train_target_network()
        return loss

    def _get_noise(self, action):
        #TODO: test various noises
        return 0.
