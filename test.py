import gym
import time
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
video_dir_path = 'mario_videos'
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
# env = gym.wrappers.RecordVideo(env, video_folder=video_dir_path, episode_trigger=lambda episode_id: True, name_prefix='mario-video-{}'.format(time.ctime())) 
env = gym.wrappers.RecordVideo(env, video_folder=video_dir_path, episode_trigger=lambda episode_id: True, name_prefix=f'mario-video-{timestamp}')

# run 1 episode
env.reset()
while True:
    state, reward, done, info = env.step(env.action_space.sample())
    if done or info['time'] < 250:
        break
print("Your mario video is saved in {}".format(video_dir_path))
try:
    del env
except Exception:
    pass