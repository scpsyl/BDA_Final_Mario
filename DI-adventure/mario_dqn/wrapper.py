"""
wrapper定义文件
"""
from typing import Union, List, Tuple, Callable
from ding.envs.env_wrappers import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv
import gym
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
import torch
from ding.torch_utils import to_ndarray
import os
import warnings
import copy

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from ding.envs import BaseEnvManager


# 粘性动作wrapper
class StickyActionWrapper(gym.ActionWrapper):
    """
    Overview:
       A certain possibility to select the last action
    Interface:
        ``__init__``, ``action``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
        - ``p_sticky``: possibility to select the last action
    """

    def __init__(self, env: gym.Env, p_sticky: float=0.25):
        super().__init__(env)
        self.p_sticky = p_sticky
        self.last_action = 0

    def action(self, action):
        if np.random.random() < self.p_sticky:
            return_action = self.last_action
        else:
            return_action = action
        self.last_action = action
        return return_action


# 稀疏奖励wrapper
class SparseRewardWrapper(gym.Wrapper):
    """
    Overview:
       Only death and pass sparse reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        dead = True if reward == -15 else False
        reward = 0
        if info['flag_get']:
            reward = 15
        if dead:
            reward = -15
        return obs, reward, done, info


# 硬币奖励wrapper
class CoinRewardWrapper(gym.Wrapper):
    """
    Overview:
        add coin reward
    Interface:
        ``__init__``, ``step``
    Properties:
        - env (:obj:`gym.Env`): the environment to wrap.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_coins = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += (info['coins'] - self.num_coins) * 10
        self.num_coins = info['coins']
        return obs, reward, done, info


# CAM相关，不需要了解
def dump_arr2video(arr, video_folder):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 6
    size = (256, 240)
    out = cv2.VideoWriter(video_folder + '/cam_pure.mp4', fourcc, fps, size)
    out1 = cv2.VideoWriter(video_folder + '/obs_pure.mp4', fourcc, fps, size)
    out2 = cv2.VideoWriter(video_folder + '/merged.mp4', fourcc, fps, size)
    for frame, obs in arr:
        frame = (255 * frame).astype('uint8').squeeze(0)
        frame_c = cv2.resize(cv2.applyColorMap(frame, cv2.COLORMAP_JET), size)
        out.write(frame_c)

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        out1.write(obs)

        merged_frame = cv2.addWeighted(obs, 0.6, frame_c, 0.4, 0)
        out2.write(merged_frame)
    # assert False


def get_cam(img, model):
    target_layers = [model.encoder.main[0]]
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    return grayscale_cam


def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordCAM(gym.Wrapper):

    def __init__(
        self,
        env,
        cam_model,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        super(RecordCAM, self).__init__(env)
        self._env = env
        self.cam_model = cam_model

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum([x is not None for x in [episode_trigger, step_trigger]])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_recorder = []

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            warnings.warn(
                f"Overwriting existing videos at {self.video_folder} folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0

    def reset(self, **kwargs):
        observations = super(RecordCAM, self).reset(**kwargs)
        if not self.recording:
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"

        base_path = os.path.join(self.video_folder, video_name)
        self.video_recorder = []

        self.recorded_frames = 0
        self.recording = True

    def _video_enabled(self):
        if self.step_trigger:
            return self.step_trigger(self.step_id)
        else:
            return self.episode_trigger(self.episode_id)

    def step(self, action):
        time_step = super(RecordCAM, self).step(action)
        observations, rewards, dones, infos = time_step

        # increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self.video_recorder.append(
                (get_cam(observations, model=self.cam_model), copy.deepcopy(self.env.render(mode='rgb_array')))
            )
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > 10000:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones or infos['time'] < 250:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return time_step

    def close_video_recorder(self) -> None:
        if self.recorded_frames > 0:
            dump_arr2video(self.video_recorder, self.video_folder)
        self.video_recorder = []
        self.recording = False
        self.recorded_frames = 0

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

# wrapper.py

# wrapper.py

class ForwardRewardWrapper(gym.Wrapper):
    """
    Overview:
        Add a forward progress reward to encourage the agent to move to the right.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_x_position = 0

    def reset(self, **kwargs):
        self.last_x_position = self.env.get_current_x_position()  # 确保环境有此方法
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_x = self.env.get_current_x_position()  # 确保环境有此方法
        forward_progress = current_x - self.last_x_position
        self.last_x_position = current_x
        # 奖励鼓励向右移动
        reward += forward_progress * 1.0  # 可以根据需要调整系数
        return obs, reward, done, info

class TimePenaltyWrapper(gym.Wrapper):
    """
    Overview:
        Add a time penalty to encourage faster completion.
    """
    def __init__(self, env: gym.Env, penalty: float = -0.01):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += self.penalty
        return obs, reward, done, info
    
class CombinedRewardWrapper(gym.Wrapper):
    """
    Overview:
        Combine SparseReward, CoinReward, ForwardReward with TimePenalty
    """
    def __init__(self, env: gym.Env, time_penalty: float = -0.01):
        super().__init__(env)
        self.num_coins = 0
        self.last_x_position = 0
        self.time_penalty = time_penalty

    def reset(self, **kwargs):
        self.last_x_position = self.env.get_current_x_position()
        self.num_coins = self.env.get_current_x_position()  # 根据需要初始化
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 方向性奖励
        current_x = self.env.get_current_x_position()
        forward_progress = current_x - self.last_x_position
        self.last_x_position = current_x
        direction_reward = forward_progress * 1.0  # 可以调整系数

        # 硬币奖励
        coin_reward = (info.get('coins', 0) - self.num_coins) * 10
        self.num_coins = info.get('coins', 0)

        # 稀疏奖励
        if info.get('flag_get', False):
            sparse_reward = 15
        elif reward == -15:
            sparse_reward = -15
        else:
            sparse_reward = 0

        # 时间惩罚
        time_reward = self.time_penalty

        # 综合奖励
        total_reward = direction_reward + coin_reward + sparse_reward + time_reward
        return obs, total_reward, done, info

# CAM相关辅助函数
def dump_arr2video(arr, video_folder):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = 6
    size = (256, 240)
    out = cv2.VideoWriter(os.path.join(video_folder, 'cam_pure.mp4'), fourcc, fps, size)
    out1 = cv2.VideoWriter(os.path.join(video_folder, 'obs_pure.mp4'), fourcc, fps, size)
    out2 = cv2.VideoWriter(os.path.join(video_folder, 'merged.mp4'), fourcc, fps, size)
    for frame, obs in arr:
        frame = (255 * frame).astype('uint8').squeeze(0)
        frame_c = cv2.resize(cv2.applyColorMap(frame, cv2.COLORMAP_JET), size)
        out.write(frame_c)

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        out1.write(obs)

        merged_frame = cv2.addWeighted(obs, 0.6, frame_c, 0.4, 0)
        out2.write(merged_frame)
    out.release()
    out1.release()
    out2.release()


def get_cam(img, model):
    target_layers = [model.encoder.main[0]]
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    return grayscale_cam

def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


def get_cam(img, model):
    target_layers = [model.encoder.main[0]]
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    return grayscale_cam

def capped_cubic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0

class CustomMarioEnv(gym_super_mario_bros.SuperMarioBrosEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_x = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # 假设info字典中有x位置
        self.current_x = info.get('x_position', 0)
        return obs, reward, done, info

    def get_current_x_position(self):
        return self.current_x


def make_env(version=0, action=2, obs=4, cam_model=None, video_folder="./videos"):
    """
    Overview:
        Create and wrap the Gym environment with various wrappers.
    Arguments:
        - version (int): Game version (0,1,2,3).
        - action (int): Number of actions (e.g., 2,7,12).
        - obs (int): Number of stacked frames (1 or 4).
        - cam_model (torch.nn.Module): Model for CAM (optional).
        - video_folder (str): Folder to save CAM videos.
    Returns:
        - env (gym.Env): Wrapped environment.
    """
    env_id = f"SuperMarioBros-1-1-v{version}"
    # 获取简化的动作集合
    actions = SIMPLE_MOVEMENT[:action]
    # 创建自定义环境实例
    env = CustomMarioEnv(env_id, actions=actions)
    # 应用包装器
    env = StickyActionWrapper(env, p_sticky=0.25)
    env = CombinedRewardWrapper(env, time_penalty=-0.01)
    env = FrameStackWrapper(env, n_frames=obs)  # 多帧堆叠
    env = WarpFrameWrapper(env, width=84, height=84)  # 放缩观测图像
    env = MaxAndSkipWrapper(env, skip=4)  # 跳帧
    if cam_model is not None:
        env = RecordCAM(env, cam_model, video_folder)
    return env