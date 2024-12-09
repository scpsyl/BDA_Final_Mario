"""
智能体训练入口，包含训练逻辑
"""
from tensorboardX import SummaryWriter
from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv
from policy import DQNPolicy
from model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from mario_dqn_config import mario_dqn_config
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from functools import partial
import os
import gym_super_mario_bros

import argparse
from copy import deepcopy
import torch
import os

from ding.entry import serial_pipeline
from ding.envs import get_vec_env_setting
from ding.config import compile_config
from wrapper import make_env, CustomMarioEnv  # 确保make_env函数在wrapper.py中正确导出
from policy import DQNPolicy  # 确保policy.py中正确导出DQNPolicy



# 动作相关配置
action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


# mario环境
def wrapped_mario_env(version=0, action=7, obs=1):
    return DingEnvWrapper(
        # 设置mario游戏版本与动作空间
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v"+str(version)), action_dict[int(action)]),
        cfg={
            # 添加各种wrapper
            'env_wrapper': [
                # 默认wrapper：跳帧以降低计算量
                lambda env: MaxAndSkipWrapper(env, skip=4),
                # 默认wrapper：将mario游戏环境图片进行处理，返回大小为84X84的图片observation
                lambda env: WarpFrameWrapper(env, size=84),
                # 默认wrapper：将observation数值进行归一化
                lambda env: ScaledFloatFrameWrapper(env),
                # 默认wrapper：叠帧，将连续n_frames帧叠到一起，返回shape为(n_frames,84,84)的图片observation
                lambda env: FrameStackWrapper(env, n_frames=obs),
                # 默认wrapper：在评估一局游戏结束时返回累计的奖励，方便统计
                lambda env: FinalEvalRewardEnv(env),
                # 以下是你添加的wrapper
            ]
        }
    )

# def main(cfg, args, seed=0, max_env_step=int(3e6)):
#     # 编译配置
#     cfg = compile_config(
#         cfg,
#         SyncSubprocessEnvManager,
#         DQNPolicy,
#         BaseLearner,
#         SampleSerialCollector,
#         InteractionSerialEvaluator,
#         AdvancedReplayBuffer,
#         seed=seed,
#         save_cfg=True
#     )
#     # 获取收集器与评估器环境数
#     collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num

#     # 使用make_env函数，而不是wrapped_mario_env
#     # 请确保你在wrapper.py中已定义make_env函数，并支持version、action、obs等参数
#     # 同时在make_env中整合各种wrapper（如ForwardRewardWrapper、TimePenaltyWrapper、FrameStackWrapper等）
#     collector_env = SyncSubprocessEnvManager(
#         env_fn=[
#             partial(
#                 make_env,
#                 version=args.version,
#                 action=args.action,
#                 obs=args.obs,
#                 cam_model=None,
#                 video_folder="./videos"
#             ) for _ in range(collector_env_num)
#         ],
#         cfg=cfg.env.manager
#     )
#     evaluator_env = SyncSubprocessEnvManager(
#         env_fn=[
#             partial(
#                 make_env,
#                 version=args.version,
#                 action=args.action,
#                 obs=args.obs,
#                 cam_model=None,
#                 video_folder="./videos"
#             ) for _ in range(evaluator_env_num)
#         ],
#         cfg=cfg.env.manager
#     )

#     # 设置随机种子
#     collector_env.seed(seed)
#     evaluator_env.seed(seed, dynamic_seed=False)
#     set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

#     # 创建DQN模型与策略
#     model = DQN(**cfg.policy.model)
#     policy = DQNPolicy(cfg.policy, model=model)

#     # 日志记录器
#     tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))

#     # 初始化学习器、收集器、评估器、回放缓冲区
#     learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
#     collector = SampleSerialCollector(
#         cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
#     )
#     evaluator = InteractionSerialEvaluator(
#         cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
#     )
#     replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

#     # 设置epsilon贪心策略
#     eps_cfg = cfg.policy.other.eps
#     epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

#     # 开始训练与评估循环
#     while True:
#         # 根据训练迭代数决定是否进行评估
#         if evaluator.should_eval(learner.train_iter):
#             stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
#             if stop:
#                 break

#         # 更新ε值
#         eps = epsilon_greedy(collector.envstep)

#         # 收集新数据并推入回放缓冲区
#         new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
#         replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)

#         # 从回放缓冲区中采样并训练模型
#         for i in range(cfg.policy.learn.update_per_collect):
#             train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
#             if train_data is None:
#                 break
#             learner.train(train_data, collector.envstep)

#         # 判断是否达到训练步数上限
#         if collector.envstep >= max_env_step:
#             break


# if __name__ == "__main__":
#     from copy import deepcopy
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", "-s", type=int, default=0)
#     parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
#     parser.add_argument("--action", "-a", type=int, default=7, choices=[2,7,12])
#     parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])
#     args = parser.parse_args()

#     # 更新实验名称与模型输入输出形状
#     mario_dqn_config.exp_name = 'exp/v'+str(args.version)+'_'+str(args.action)+'a_'+str(args.obs)+'f_seed'+str(args.seed)
#     mario_dqn_config.policy.model.obs_shape = [args.obs, 84, 84]
#     mario_dqn_config.policy.model.action_shape = args.action

#     main(deepcopy(mario_dqn_config), args, seed=args.seed)


# mario_dqn_main.py

def main(cfg, args, seed=0, max_env_step=int(3e6)):
    # 编译配置
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        seed=seed,
        save_cfg=True
    )
    # 获取收集器与评估器环境数
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num

    # 使用make_env函数，而不是wrapped_mario_env
    collector_env = SyncSubprocessEnvManager(
        env_fn=[
            partial(
                make_env,
                version=args.version,
                action=args.action,
                obs=args.obs,
                cam_model=None,
                video_folder="./videos"
            ) for _ in range(collector_env_num)
        ],
        cfg=cfg.env.manager
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[
            partial(
                make_env,
                version=args.version,
                action=args.action,
                obs=args.obs,
                cam_model=None,
                video_folder="./videos"
            ) for _ in range(evaluator_env_num)
        ],
        cfg=cfg.env.manager
    )

    # 设置种子
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # 创建DQN模型与策略
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    # 日志记录器
    tb_logger = SummaryWriter(os.path.join(f'./{cfg.exp_name}/log/', 'serial'))

    # 初始化学习器、收集器、评估器、回放缓冲区
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # 设置epsilon贪心策略
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # 训练以及评估
    while True:
        # 根据当前训练迭代数决定是否进行评估
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # 更新epsilon greedy信息
        eps = epsilon_greedy(collector.envstep)
        # 经验收集器从环境中收集经验
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        # 将收集的经验放入replay buffer
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # 采样经验进行训练
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
        if collector.envstep >= max_env_step:
            break

if __name__ == "__main__":
    from copy import deepcopy
    import argparse
    parser = argparse.ArgumentParser()
    # 种子
    parser.add_argument("--seed", "-s", type=int, default=0)
    # 游戏版本，v0 v1 v2 v3 四种选择
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    # 动作集合种类，包含[["right"], ["right", "A"]]、SIMPLE_MOVEMENT、COMPLEX_MOVEMENT，分别对应2、7、12个动作
    parser.add_argument("--action", "-a", type=int, default=2, choices=[2,7,12])  # 默认简化为2个动作
    # 观测空间叠帧数目，不叠帧或叠四帧
    parser.add_argument("--obs", "-o", type=int, default=4, choices=[1,4])  # 默认堆叠4帧
    args = parser.parse_args()
    mario_dqn_config.exp_name = f'exp/v{args.version}_{args.action}a_{args.obs}f_seed{args.seed}'
    mario_dqn_config.policy.model.obs_shape = [args.obs, 84, 84]
    mario_dqn_config.policy.model.action_shape = args.action
    main(deepcopy(mario_dqn_config), args, seed=args.seed)
