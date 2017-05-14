import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from .replay_buffer import ReplayBuffer
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from ..base import Strategy
import rl_market.utils.logging_conf
import logging as log
from tqdm import tqdm

class DDPG(Strategy):
    def __init__(self,
            state_shape,
            action_dim,
            generator,
            noise_func,
            observation_func = None,
            action_func = None,
            learning_phase = 1,
            sess = None,
            actor_weight_path = None,
            critic_weight_path = None,
            actor_save_path = None,
            critic_save_path = None,
            log_save_path = None,
            BUFFER_SIZE = 1000000,
            BATCH_SIZE = 32,
            GAMMA = 0.99,           # reward discount factor
            TAU = 0.001,            # target network shift rate
            LRA = 0.000001,           # learning rate for actor
            LRC = 0.0000001,            # learning rate for critic
            EXPLORE = 100000.,      # explore factor
            hard_reset = False,
            random_seed = 42):

        super(DDPG, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.noise_func = noise_func
        self.observation_func = observation_func
        self.action_func = action_func

        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LRA = LRA
        self.LRC = LRC
        self.EXPLORE = EXPLORE
        self.hard_reset=hard_reset

        K.set_learning_phase(learning_phase)

        np.random.seed(random_seed)

        if sess is not None:
            self.sess = sess
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config = config)
        K.set_session(self.sess)

        self.buff = ReplayBuffer(self.BUFFER_SIZE)
        self.actor = ActorNetwork(self.sess, state_shape, action_dim, generator, None, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic =  CriticNetwork(self.sess, state_shape, action_dim, generator, None, self.BATCH_SIZE, self.TAU, self.LRC)
        if actor_weight_path is not None:
            self.actor.model.load_weights(actor_weight_path)
            self.actor.target_model.load_weights(actor_weight_path)
            print("actor model weight loaded from {}".format(actor_weight_path))
        if critic_weight_path is not None:
            self.critic.model.load_weights(critic_weight_path)
            self.critic.target_model.load_weights(critic_weight_path)
            print("critic model weight loaded from {}".format(critic_weight_path))

        self.logger = None
        if log_save_path is not None:
            if os.path.isfile(log_save_path):
                answer = input("File {} exists! overwrite?(y/N)".format(log_save_path))
                if answer in ["y","Y"]:
                    self.logger = open(log_save_path,"w")
            else:
                self.logger = open(log_save_path,"w")

    def __repr__(self):
        return "DDPG"

    def pretrain(self, game, strategy, action_encoder = None, nr_episode = 10, nr_steps =1000):
        if action_encoder is None:
            log.warning("Make sure action is the same as action_encoding")
            def identity_func(inp, args):
                return inp
            action_encoder = identity_func

        for episode in range(nr_episode):
            game.reset(hard=self.hard_reset)
            pbar = tqdm(range(nr_steps))
            state, args = self._get_state(game.get_observation())
            for step in pbar:
                action = strategy.play(game)
                action_encoding = action_encoder(action, args)
                reward, done = game.step(action)
                new_state, new_args = self._get_state(game.get_observation())
                self.buff.add((state, action_encoding, reward, new_state, done))
                loss = self._batch_update()
                pbar.set_description("s{} R{:.3} L{:.3}".format(step, reward, loss))
                state, args = new_state, new_args
                if done:
                    break

    def train(self, game, nr_episode = 1000, nr_steps = 100000):
        # NOTE: interact with the stateful game object
        if len(self.buff.buff)>0:
            log.info("{} experience detected in buffer.".format(len(self.buff.buff)))
        epsilon = 1.
        total_step = 0
        for episode in range(nr_episode):
            game.reset(hard = self.hard_reset)
            observation = game.get_observation() # should be a numpy array
            state, args = self._get_state(observation)
            total_reward = 0.
            total_loss= 0.
            pbar = tqdm(range(nr_steps))
            for step in pbar:
                epsilon -= 1/self.EXPLORE
                action_encoding, new_state, reward, done, new_args = self._perform_action(game, state, epsilon, args)
                self.buff.add((state, action_encoding, reward, new_state, done))
                loss = self._batch_update()

                #update state & show stats
                state, args = new_state, new_args
                pbar.set_description("s{} R{:.3} L{:.3}".format(step, reward, loss))
                if self.logger:
                    self.logger.write("{}\n".format((episode*nr_steps+step,reward,loss)))

                total_reward += reward
                total_loss +=loss
                total_step += 1
                if done:
                    break
            if np.mod(episode, 5)==0:
                log.info("save weights to {} & {}".format(self.actor_save_path, self.critic_save_path))
                if self.actor_save_path is not None:
                    self.actor.model.save_weights(self.actor_save_path, overwrite=True)
                if self.critic_save_path is not None:
                    self.critic.model.save_weights(self.critic_save_path, overwrite=True)
            log.info("total reward @{}-th episode : {}".format(episode, total_reward))
            log.info("total loss : {}".format(total_loss))
            log.info("total step : {}".format(total_step))
        log.info("train finish")

    def reset(self):
        pass

    def play(self, game):
        #given observation, get action
        observation = game.get_observation()
        state, args = self._get_state(observation)
        action_encoding = self.actor.model.predict([state[np.newaxis,:]])[0]
        return self._get_action(action_encoding, args)

    def _get_state(self, observation):
        # potential for observation conversion here
        if self.observation_func is None:
            return observation, None
        return self.observation_func(observation)

    def _get_action(self, action_encoding, args):
        if self.action_func is None:
            return action_encoding
        return self.action_func(action_encoding, args)

    def _perform_action(self, game, state, epsilon, args):
        action_encoding = self.actor.model.predict([state[np.newaxis, :]])[0]
        #exploration
        for i in range(self.action_dim):
            action_encoding[i] += max(epsilon,0) * self.noise_func(action_encoding[i])

        action = self._get_action(action_encoding, args)
        reward, done = game.step(action) # NOTE: inside game, it will normalize action if required
        new_observation = game.get_observation()
        new_state, new_args = self._get_state(new_observation)
        return action_encoding, new_state, reward, done, new_args


    def _batch_update(self):
        batch = self.buff.get_batch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        action_encodings = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])

        #use critic network to estimate y_t
        y_t = np.zeros((len(batch),1))
        target_q_values = self.critic.target_model.predict([new_states,self.actor.target_model.predict(new_states)])
        for k in range(len(batch)):
            if dones[k]:
                y_t[k][0] = rewards[k]
            else:
                y_t[k][0] = rewards[k] + self.GAMMA * target_q_values[k]

        # update actor & critic
        loss = self.critic.model.train_on_batch([states, action_encodings], y_t)
        action_for_grad = self.actor.model.predict([states])
        grads = self.critic.get_gradient(states, action_for_grad)
        self.actor.train(states, grads)
        #update target networks
        self.actor.train_target_network()
        self.critic.train_target_network()
        return loss
