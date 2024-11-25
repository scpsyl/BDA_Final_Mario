exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 5,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'reset',
            'retry_waiting_time': 0.1,
            'shared_memory': True,
            'copy_on_get': True,
            'context': 'spawn',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict'
        },
        'stop_value': 3000,
        'collector_env_num': 8,
        'evaluator_env_num': 8,
        'n_evaluator_episode': 8
    },
    'policy': {
        'model': {
            'obs_shape': [1, 84, 84],
            'action_shape': 7,
            'encoder_hidden_size_list': [32, 64, 128],
            'dueling': False
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 10,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'target_update_freq': 500,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict'
            },
            'unroll_len': 1,
            'n_sample': 96
        },
        'eval': {
            'evaluator': {
                'eval_freq': 2000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'n_episode': 8,
                'stop_value': 3000
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 100000,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'eps': {
                'type': 'exp',
                'start': 1.0,
                'end': 0.05,
                'decay': 250000
            }
        },
        'type': 'dqn',
        'cuda': True,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'discount_factor': 0.99,
        'nstep': 3,
        'cfg_type': 'DQNPolicyDict'
    },
    'exp_name': 'exp/v0_7a_1f_seed0_241119_215539',
    'seed': 0
}
