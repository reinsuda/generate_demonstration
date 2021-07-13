import torch
import pandas as pd
import cv2
import os
from mpi4py import MPI
import time
import argparse
import gym
import numpy as np
import datetime
import itertools
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory


def get_passer():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--file_name', type=str, default=" ", metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--env_name', default="Walker2d-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    # actor and critic model patch
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true", default=True,
                        help='run on CUDA (default: False)')

    args = parser.parse_args()

    return args


def get_model_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]


np.set_printoptions(precision=3)


def compute_all_file(args, file_name):
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # model_path = "models/{}/{}".format(args.env_name, file_name)
    agent.load_model(file_name)

    total_numsteps = 0
    updates = 0
    return_list = []
    length_list = []
    for _ in range(20):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done:
            action = agent.select_action(state, evaluate=True)  # Sample action from policy
            next_state, reward, done, _ = env.step(action)  # Step
            # env.render()
            episode_reward += reward
            total_numsteps += 1
            episode_steps += 1
            state = next_state
        length_list.append(episode_steps)
        return_list.append(episode_reward)
    dd = []
    for index_i, lens in enumerate(length_list):
        if lens == 1000:
            dd.append(return_list[index_i])
    return np.mean(dd), np.std(dd), len(dd)


if __name__ == "__main__":
    args = get_passer()
    file_name_list = get_model_list("models/" + args.env_name)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()  # 进程id
    comm_size = comm.Get_size()  # 进程个数
    # print(comm_rank, comm_size)

    if comm_rank == 0:
        data = []
        for i in range(len(file_name_list)):
            index = i % comm_size
            if index > len(data) - 1:
                data.append([file_name_list[i]])
            else:
                data[index].append(file_name_list[i])
    else:
        data = None
    local_data = comm.scatter(data, root=0)
    mean_list, std_list, = [], []
    result = []
    for file_name in local_data:
        mean, std, count = compute_all_file(args, file_name)
        result.append([file_name, mean, std, count])
        mean_list.append(mean)
        std_list.append(std)
    gather_result = comm.gather(result, root=0)
    if comm_rank == 0:
        col_names = ["file_name", "mean", "std", "count"]
        gather_result = np.concatenate(gather_result, axis=0)
        df = pd.DataFrame(columns=col_names, data=gather_result)
        path_save = "csv_file" + args.env_name
        if not os.path.isdir(path_save):
            os.makedirs(path_save)
        df.to_csv("csv_file" + args.env_name + "/test_model.csv", encoding="utf-8", index=False)
