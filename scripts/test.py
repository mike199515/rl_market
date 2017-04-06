from rl_market.strategy.ddpg.ddpg import DDPG

from rl_market.strategy.rule_based_distribute.random_distribute import RandomDistribute
from rl_market.strategy.rule_based_distribute.uniform_distribute import UniformDistribute
from rl_market.strategy.rule_based_distribute.direct_optimize import DirectOptimize

from rl_market.strategy.ddpg.model_generator.simple_fc import SimpleFCAction, SimpleFCCritic
from rl_market.game.price_market import PriceMarket
from rl_market.player.buyer.simple_buyer import SimpleBuyer
from rl_market.player.seller.simple_seller import SimpleSeller

import rl_market.utils.logging_conf
import rl_market.utils.sampler as sampler
import logging as log

from tqdm import tqdm
import numpy as np
import argparse

THE_MEANING_OF_LIFE_UNIVERSE_AND_EVERYTHING=42

def observation_func(observation):
    return observation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("s", help= "the strategy to test")

    args = parser.parse_args()


    log.info("=================================")

    np.random.seed(THE_MEANING_OF_LIFE_UNIVERSE_AND_EVERYTHING)
    buyer = SimpleBuyer()

    quality_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    cost_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    #cost_sampler = sampler.UniformSampler()
    price_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    noise_sampler = sampler.GaussianSampler(mu = 0., sigma = 0.05/3)
    sellers = [SimpleSeller(quality_sampler, cost_sampler, price_sampler, noise_sampler) for i in range(10)]

    log.info("initialize game:PriceMarket")
    game = PriceMarket(sellers=sellers,buyer=buyer,max_duration=1000)

    #log.info("initialize strategy:ddpg")
    log.info("start testing {}".format(args.s))
    if args.s == "uniform":
        strategy = UniformDistribute()
    elif args.s == "random":
        strategy = RandomDistribute()
    elif args.s == "direct":
        strategy = DirectOptimize()
    elif args.s == "ddpg":
        strategy = DDPG(state_shape = game.state_shape, action_dim = game.action_dim,
                actor_generator=SimpleFCAction(),
                critic_generator=SimpleFCCritic(),
                observation_func=observation_func,
                actor_weight_path="../data/ddpg_actor_model.hdf5",
                critic_weight_path="../data/ddpg_critic_model.hdf5"
        )

    for epoch in range(1000):
        total_reward = 0
        game.reset(hard=False)
        for step in tqdm(range(1000)):
            #print("=========step {}=========\n{}\n".format(step,game.get_observation_string()))
            action =  strategy.play(game)
            reward, done = game.step(action)
            total_reward += reward
            if done:
                break
        print("total reward@{}-th episode: {}".format(epoch, total_reward))

if __name__=="__main__":
    main()
