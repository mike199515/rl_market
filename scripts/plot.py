

import argparse
import glob
import rl_market.utils.logging_conf
import logging as log

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def average_value(value):
    gamma = 0.99
    means = []
    sigmas = []
    weighted_average=value[0]
    for i in range(len(value)):
        weighted_average = gamma * weighted_average + (1-gamma) * value[i]
        means.append(weighted_average)
        variance = np.var(value[i-100:i])
        sigmas.append(np.power(variance, 0.5))
    return np.array(means), np.array(sigmas)

def search_possible_files(args):
    possible_files = glob.glob("{}/*.log".format(args.path))
    possible_files.sort()
    log.info("{} files detected.".format(len(possible_files)))
    for i,name in enumerate(possible_files):
        print("{}:{}".format(i,name))
    answer=int(input("Which file to pick? "))
    assert(answer >= 0 and answer < len(possible_files))
    return possible_files[answer]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path" ,default="../data/")
    parser.add_argument("--save_path",default="../plot/")
    parser.add_argument("--draw_baseline",action="store_true")
    parser.add_argument("--step",type=int,default=-1)
    parser.add_argument("type",type=str)
    args = parser.parse_args()
    assert(args.type in ["reward", "loss"])
    chosen_log = search_possible_files(args)
    print("chosen ",chosen_log)
    save_path = "{}/{}_{}.png".format(args.save_path,args.type,chosen_log.split("/")[-1].split(".")[0])
    log.info("prepare to save to {}".format(save_path))
    algorithm_name = chosen_log.split("/")[-1].split(".")[0].split("_")[-1]
    f = open(chosen_log,"r")

    steps = []
    rewards = []

    losses = []
    for line in f.readlines():
        tup = list(map(float,line[1:-2].split(",")))
        step, reward, loss = tup
        steps.append(step)
        rewards.append(reward)
        losses.append(loss)

    log.info("found {} steps.".format(len(steps)))
    if args.step!=-1:
        steps=steps[:args.step]
        rewards=rewards[:args.step]
        losses=losses[:args.step]

    plt.figure()
    if args.type == "loss":
        loss_means, _ = average_value(losses)
        plt.semilogy(steps, loss_means)
        plt.title("training loss/step of {}".format(algorithm_name))
    elif args.type == "reward":
        reward_means, reward_sigmas = average_value(rewards)
        plt.fill_between(steps, reward_means-reward_sigmas, reward_means+reward_sigmas,color="gray",alpha=0.5)
        plt.plot(steps, reward_means, color="red")
        plt.title("training reward/step of {}".format(algorithm_name))
    plt.savefig(save_path)


if __name__=="__main__":
    main()
