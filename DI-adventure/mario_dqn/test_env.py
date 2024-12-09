# test_env.py

import gym
from wrapper import make_env
from policy import DQNPolicy
import torch

def test_env():
    env_id = "SuperMarioBros-1-1-v0"
    cam_model = None  # 如果不需要CAM，可以设置为None
    video_folder = "./videos_test"
    env = make_env(version=0, action=2, obs=4, cam_model=cam_model, video_folder=video_folder)
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    env.close()
    print(f"Total Reward: {total_reward}")

if __name__ == "__main__":
    test_env()
