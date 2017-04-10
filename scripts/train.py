from rl_market.strategy.ddpg.ddpg import DDPG
from rl_market.strategy.ddpg.model_generator.simple_fc import SimpleFCAction, SimpleFCCritic
#from rl_market.strategy.ddpg.model_generator.simple_fc import GRUModel
from rl_market.game.price_market import PriceMarket
from rl_market.player.buyer.simple_buyer import SimpleBuyer
from rl_market.player.seller.simple_seller import SimpleSeller

import rl_market.utils.logging_conf
import rl_market.utils.sampler as sampler
import logging as log

import numpy as np
#import argparse

THE_MEANING_OF_LIFE_UNIVERSE_AND_EVERYTHING=42

def observation_func(observation):
    return observation

def main():
    np.random.seed(THE_MEANING_OF_LIFE_UNIVERSE_AND_EVERYTHING)
    buyer = SimpleBuyer()

    quality_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    cost_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    price_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    noise_sampler = sampler.GaussianSampler(mu = 0., sigma = 0.05/3)
    sellers = [SimpleSeller(quality_sampler, cost_sampler, price_sampler, noise_sampler) for i in range(10)]

    log.info("initialize game:PriceMarket")
    game = PriceMarket(sellers=sellers,buyer=buyer,max_duration=1000, nr_observation = 1)

    log.info("initialize strategy:ddpg")
    ddpg = DDPG(state_shape = game.state_shape, action_dim = game.action_dim,
            actor_generator = SimpleFCAction(),
            critic_generator = SimpleFCCritic(),
            actor_save_path = "../data/ddpg_actor_model.hdf5",
            critic_save_path = "../data/ddpg_critic_model.hdf5",
            observation_func=observation_func,
            learning_phase = 1
            )
    log.info("start training")
    ddpg.train(game, nr_episode = 1000, nr_steps = 1000)

if __name__=="__main__":
    main()
