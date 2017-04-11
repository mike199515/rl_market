from rl_market.strategy.ddpg.ddpg import DDPG

from rl_market.strategy.rule_based_distribute.random_distribute import RandomDistribute
from rl_market.strategy.rule_based_distribute.uniform_distribute import UniformDistribute
from rl_market.strategy.rule_based_distribute.direct_optimize import DirectOptimize

from rl_market.strategy.ddpg.model_generator.simple_fc import SimpleFC
from rl_market.strategy.ddpg.model_generator.gru import GRUModel
from rl_market.game.price_market import PriceMarket
from rl_market.player.buyer.simple_buyer import SimpleBuyer
from rl_market.player.buyer.history_buyer import HistoryBuyer
from rl_market.player.seller.comparative_seller import ComparativeSeller
from rl_market.player.seller.tricky_seller import TrickySeller
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
    parser.add_argument("--buyer", default="simple")
    parser.add_argument("--seller", default="simple")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", default="fc")
    args = parser.parse_args()
    if args.model == "fc":
        model_class = SimpleFC
    elif args.model == "gru":
        model_class = GRUModel

    if args.buyer == "simple":
        buyer_class = SimpleBuyer
    elif args.buyer == "history":
        buyer_class = HistoryBuyer

    if args.seller == "simple":
        seller_class=SimpleSeller
    elif args.seller == "tricky":
        seller_class=TrickySeller
    elif args.seller == "comp":
        seller_class=ComparativeSeller

    log.info("=================================")

    np.random.seed(THE_MEANING_OF_LIFE_UNIVERSE_AND_EVERYTHING)
    buyer = buyer_class()

    quality_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    cost_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    #cost_sampler = sampler.UniformSampler()
    price_sampler = sampler.BoundGaussianSampler(mu = 0.5, sigma = 0.5/3)
    noise_sampler = sampler.GaussianSampler(mu = 0., sigma = 0.05/3)

    sellers = [seller_class(quality_sampler=quality_sampler, cost_sampler=cost_sampler, price_sampler=price_sampler, noise_sampler=noise_sampler) for i in range(10)]

    log.info("initialize game:PriceMarket")
    log.info("seller:{}".format(seller_class))
    game = PriceMarket(sellers=sellers,buyer=buyer,nr_observation = 2, max_duration=1000)

    print(game.state_shape)

    if args.train:
        log.info("start training")
        strategy = DDPG(state_shape = game.state_shape, action_dim = game.action_dim,
                generator=model_class(),
                observation_func=observation_func,
                actor_save_path = "../data/ddpg_actor_{}_{}_{}.hdf5".format(args.buyer,args.seller,args.model),
                critic_save_path = "../data/ddpg_critic_{}_{}_{}.hdf5".format(args.buyer,args.seller,args.model),
                #actor_weight_path="../data/ddpg_actor_model.hdf5",
                #critic_weight_path="../data/ddpg_critic_model.hdf5"
        )
        strategy.train(game, nr_episode = 1000, nr_steps = 1000)
        return

    #log.info("initialize strategy:ddpg")
    if args.s == "uniform":
        strategy = UniformDistribute()
    elif args.s == "random":
        strategy = RandomDistribute()
    elif args.s == "direct":
        strategy = DirectOptimize()
    elif args.s == "ddpg":
        strategy = DDPG(state_shape = game.state_shape, action_dim = game.action_dim,
                generator=model_class(),
                observation_func=observation_func,
                actor_weight_path="../data/ddpg_actor_{}_{}_{}.hdf5".format(args.buyer,args.seller,args.model),
                critic_weight_path="../data/ddpg_critic_{}_{}_{}.hdf5".format(args.buyer,args.seller,args.model),
        )

    log.info("start testing {}".format(strategy))
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
        log.info("total reward@{}-th episode: {}".format(epoch, total_reward))

if __name__=="__main__":
    main()
