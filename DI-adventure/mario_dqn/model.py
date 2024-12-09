# """
# 神经网络模型定义
# """
# from typing import Union, Optional, Dict, Callable, List
# import torch
# import torch.nn as nn

# from ding.utils import SequenceType, squeeze
# from ding.model.common import FCEncoder, ConvEncoder, DiscreteHead, DuelingHead, MultiHead


# class DQN(nn.Module):

#     mode = ['compute_q', 'compute_q_logit']

#     def __init__(
#             self,
#             obs_shape: Union[int, SequenceType], # 观测空间形状
#             action_shape: Union[int, SequenceType], # 动作空间形状
#             encoder_hidden_size_list: SequenceType = [128, 128, 64], # 编码器隐藏层大小
#             dueling: bool = True, # 是否使用对决网络
#             head_hidden_size: Optional[int] = None, # 头部隐藏层大小
#             head_layer_num: int = 1, # 头部层数
#             activation: Optional[nn.Module] = nn.ReLU(), # 激活函数
#             norm_type: Optional[str] = None # 归一化类型
#     ) -> None:
#         """
#         Overview:
#             Init the DQN (encoder + head) Model according to input arguments.
#         Arguments:
#             - obs_shape (:obj:`Union[int, SequenceType]`): Observation space shape, such as 8 or [4, 84, 84].
#             - action_shape (:obj:`Union[int, SequenceType]`): Action space shape, such as 6 or [2, 3, 3].
#             - encoder_hidden_size_list (:obj:`SequenceType`): Collection of ``hidden_size`` to pass to ``Encoder``, \
#                 the last element must match ``head_hidden_size``.
#             - dueling (:obj:`dueling`): Whether choose ``DuelingHead`` or ``DiscreteHead(default)``.
#             - head_hidden_size (:obj:`Optional[int]`): The ``hidden_size`` of head network.
#             - head_layer_num (:obj:`int`): The number of layers used in the head network to compute Q value output
#             - activation (:obj:`Optional[nn.Module]`): The type of activation function in networks \
#                 if ``None`` then default set it to ``nn.ReLU()``
#             - norm_type (:obj:`Optional[str]`): The type of normalization in networks, see \
#                 ``ding.torch_utils.fc_block`` for more details.
#         """
#         super(DQN, self).__init__()
#         # For compatibility: 1, (1, ), [4, 32, 32]
#         obs_shape, action_shape = squeeze(obs_shape), squeeze(action_shape)
#         if head_hidden_size is None:
#             head_hidden_size = encoder_hidden_size_list[-1]
#         # FC Encoder
#         if isinstance(obs_shape, int) or len(obs_shape) == 1:
#             self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
#         # Conv Encoder
#         elif len(obs_shape) == 3:
#             self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
#         else:
#             raise RuntimeError(
#                 "not support obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
#             )
#         # Head Type
#         if dueling:
#             head_cls = DuelingHead
#         else:
#             head_cls = DiscreteHead
#         multi_head = not isinstance(action_shape, int)
#         if multi_head:
#             self.head = MultiHead(
#                 head_cls,
#                 head_hidden_size,
#                 action_shape,
#                 layer_num=head_layer_num,
#                 activation=activation,
#                 norm_type=norm_type
#             )
#         else:
#             self.head = head_cls(
#                 head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
#             )


#     def forward(self, x: torch.Tensor, mode: str='compute_q_logit') -> Dict:
#         assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
#         return getattr(self, mode)(x)


#     def compute_q(self, x: torch.Tensor) -> Dict:
#         r"""
#         Overview:
#             DQN forward computation graph, input observation tensor to predict q_value.
#         Arguments:
#             - x (:obj:`torch.Tensor`): Observation inputs
#         Returns:
#             - outputs (:obj:`Dict`): DQN forward outputs, such as q_value.
#         ReturnsKeys:
#             - logit (:obj:`torch.Tensor`): Discrete Q-value output of each action dimension.
#         Shapes:
#             - x (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size and N is ``obs_shape``
#             - logit (:obj:`torch.FloatTensor`): :math:`(B, M)`, where B is batch size and M is ``action_shape``
#         Examples:
#             >>> model = DQN(32, 6)  # arguments: 'obs_shape' and 'action_shape'
#             >>> inputs = torch.randn(4, 32)
#             >>> outputs = model(inputs)
#             >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 6])
#         """
#         x = self.encoder(x)
#         x = self.head(x)
#         return x


#     def compute_q_logit(self, x: torch.Tensor) -> Dict:
#         x = self.encoder(x)
#         x = self.head(x)
#         return x['logit']

# policy.py

import torch
import torch.nn as nn
from typing import Union, Sequence, Optional, Dict, List
from ding.model import FCEncoder, ConvEncoder, DuelingHead, DiscreteHead, MultiHead

class DQN(nn.Module):
    mode = ['compute_q', 'compute_q_logit']

    def __init__(
            self,
            obs_shape: Union[int, Sequence[int]],  # 观测空间形状
            action_shape: Union[int, Sequence[int]],  # 动作空间形状
            encoder_hidden_size_list: Sequence[int] = [128, 128, 64],  # 编码器隐藏层大小
            dueling: bool = True,  # 是否使用对决网络
            head_hidden_size: Optional[int] = None,  # 头部隐藏层大小
            head_layer_num: int = 1,  # 头部层数
            activation: Optional[nn.Module] = nn.ReLU(),  # 激活函数
            norm_type: Optional[str] = None  # 归一化类型
    ) -> None:
        """
        Overview:
            Initialize the DQN (encoder + head) model according to input arguments.
        Arguments:
            - obs_shape (Union[int, Sequence[int]]): Observation space shape, e.g., 8 or [4, 84, 84].
            - action_shape (Union[int, Sequence[int]]): Action space shape, e.g., 6 or [2, 3, 3].
            - encoder_hidden_size_list (Sequence[int]): Hidden sizes for the encoder.
            - dueling (bool): Whether to use a dueling network.
            - head_hidden_size (Optional[int]): Hidden size for the head network.
            - head_layer_num (int): Number of layers in the head network.
            - activation (Optional[nn.Module]): Activation function in the network.
            - norm_type (Optional[str]): Type of normalization in the network.
        """
        super(DQN, self).__init__()
        # Ensure obs_shape and action_shape are in list format if they are sequences
        obs_shape, action_shape = self.squeeze_shape(obs_shape), self.squeeze_shape(action_shape)
        
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        
        # Encoder setup
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.encoder = FCEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        elif len(obs_shape) == 3:
            self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        else:
            raise RuntimeError(
                "Unsupported obs_shape for pre-defined encoder: {}, please customize your own DQN".format(obs_shape)
            )
        
        # Head setup
        if dueling:
            head_cls = DuelingHead
        else:
            head_cls = DiscreteHead
        
        multi_head = not isinstance(action_shape, int)
        
        if multi_head:
            self.head = MultiHead(
                head_cls,
                head_hidden_size,
                action_shape,
                layer_num=head_layer_num,
                activation=activation,
                norm_type=norm_type
            )
        else:
            self.head = head_cls(
                head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
            )

    @staticmethod
    def squeeze_shape(shape: Union[int, Sequence[int]]) -> Union[int, List[int]]:
        """
        Squeeze the shape to ensure it is a list if it is a sequence or keep as int if not.
        """
        if isinstance(shape, (list, tuple)):
            return list(shape)
        return shape

    def forward(self, x: torch.Tensor, mode: str='compute_q_logit') -> Dict:
        assert mode in self.mode, "Unsupported forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(x)

    def compute_q(self, x: torch.Tensor) -> Dict:
        """
        Compute Q-values for each action.
        """
        x = self.encoder(x)
        x = self.head(x)
        return x

    def compute_q_logit(self, x: torch.Tensor) -> Dict:
        """
        Compute Q-logits for each action.
        """
        x = self.encoder(x)
        x = self.head(x)
        return {'logit': x['logit'] if isinstance(x, dict) else x}
