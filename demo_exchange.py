import gym
import gym_exchange
from collections import Counter, defaultdict
import time
# from stable_baselines.common.policies import MlpPolicy
from dummy_vec_env import DummyVecEnv
# from stable_baselines import A2C
from model import LEARN_FREQ, MEMORY_SIZE, MEMORY_WARMUP_SIZE, BATCH_SIZE ,LEARNING_RATE, GAMMA, ReplayMemory, Model, DQN, Agent
import numpy as np
from parl.utils import logger as lg
import random
from paddle import fluid

def run_episode(env,agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward , done, _  = env.step([action])
        rpm.append((obs,action, reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and step % LEARN_FREQ == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)  = rpm.sample(BATCH_SIZE)

            batch_obs = np.squeeze(batch_obs,axis=1)
            batch_next_obs = np.squeeze(batch_next_obs,axis=1)
            batch_reward = np.squeeze(batch_reward,axis=-1)
            batch_done = np.squeeze(batch_done,axis=-1)

            # print("batch_obs.shape: ", batch_obs.shape)
            # print("batch_action.shape: ", batch_action.shape)
            # print("batch_reward.shape: ", batch_reward.shape)
            # print("batch_next_obs.shape: ", batch_next_obs.shape)
            # print("batch_done.shape: ", batch_done.shape)


            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward

def evaluate(env,agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            if action == 0:
                action = random.choice([-1,1],)
            else:
                action = 1
            obs, reward, done, _ = env.step([action])
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


if __name__ == '__main__':
    # Create and wrap the environment
    env = gym.make('game-stock-exchange-continuous-v0')
    env = DummyVecEnv([lambda: env])
    action_dim = 2
    obs_shape = env.observation_space.shape
    rpm  = ReplayMemory(MEMORY_SIZE)

    model = Model(act_dim = action_dim)
    algorithm = DQN(model, act_dim = action_dim, gamma = GAMMA, lr = LEARNING_RATE)

    agent = Agent(algorithm, obs_shape[0],obs_shape[1],action_dim)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env,agent,rpm)

    max_episode = 2000
    episode = 0
    while episode < max_episode:
        for i in range(0,50):
            total_reward = run_episode(env,agent,rpm)
            episode += 1

        eval_reward = evaluate(env, agent, render=False)
        lg.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))
