import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opencd.registry import MODELS
from typing import List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import caffe2_xavier_init

from mmseg.models.utils import nlc_to_nchw, LayerNorm2d
from mmseg.utils import ConfigType, SampleList
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from opencd.registry import MODELS


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, d, eps=1e-5):
        """
        Args:
            d (int): Dimension of the input features.
            eps (float): Small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        """
        Forward pass for RMSNorm.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """

    def __init__(self, d_model):
        """
        Args:
            d_model (int): Dimension of the input features.
        """
        super().__init__()
        # Intermediate projection layers
        # Typically, SwiGLU splits the computation into two parts
        self.WG = nn.Linear(d_model, d_model * 2)
        self.W1 = nn.Linear(d_model, d_model * 2)
        self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        """
        Forward pass for SwiGLU.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying SwiGLU.
        """
        # Apply the gates
        g = F.silu(self.WG(x))  # Activation part
        z = self.W1(x)  # Linear part
        # Element-wise multiplication and projection
        return self.W2(g * z)


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    Incorporates a causal mask to ensure autoregressive behavior.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head) * 0.1)

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5  # Epsilon for numerical stability

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)

    def forward(self, X1, X2):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X1.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X1)  # Shape: (batch, N, 2 * num_heads * d_head)
        K = self.W_k(X2)  # Shape: (batch, N, 2 * num_heads * d_head)
        V = self.W_v(X2)  # Shape: (batch, N, 2 * num_heads * d_head)

        # Reshape and permute for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * d_head)
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions
        # Shape: (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # ------------------- Causal Mask Implementation ------------------- #
        # Create a causal mask to prevent attention to future tokens
        # mask = torch.tril(torch.ones((N, N), device=X1.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        # mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        # -------------------------------------------------------------------- #

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)

        # Apply the causal mask
        # A1 = A1 + mask  # Mask out future positions
        # A2 = A2 + mask  # Mask out future positions

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, num_heads, N, N)
        attention2 = F.softmax(A2, dim=-1)  # (batch, num_heads, N, N)
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, N, N)

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, num_heads, N, 2 * d_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)  # (batch*num_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*num_heads, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, num_heads * 2 * d_head)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model)

    def forward(self, x1, x2):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        y = self.attn(self.norm1(x1), self.norm1(x2)) + x1
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
        return z


class DiffTransformerBlock(nn.Module):
    def __init__(self, embed_dims=768, num_heads=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            DiffTransformerLayer(
                d_model=embed_dims,
                num_heads=num_heads,
                lambda_init=0.8 - 0.6 * math.exp(-0.3 * (l - 1))  # Decaying lambda_init
            )
            for l in range(num_layers)
        ])

    def forward(self, x1, x2):
        i = 0
        out1 = x1[0][0]
        out2 = x2[0][0]
        for layer in self.layers:
            if i % (len(self.layers) // len(x1)) == 0:
                out1 = layer(out1, x1[i // (len(self.layers) // len(x1))][0])
                out2 = layer(out2, x2[i // (len(self.layers) // len(x1))][0])
            else:
                out1 = layer(out1, out2)
                out2 = layer(out2, out1)
            i = i + 1
        B, hw_shape, C = out1.shape
        hw = int(math.sqrt(hw_shape))
        out1 = out1.reshape(B, hw, hw, C).permute(0, 3, 1, 2).contiguous()
        out2 = out2.reshape(B, hw, hw, C).permute(0, 3, 1, 2).contiguous()

        return out1, out2


@MODELS.register_module()
class DiffTransformer(BaseDecodeHead):
    def __init__(self,
                 difft_cfg: ConfigType,
                 ban_dec_cfg: ConfigType,
                 **kwargs):
        super().__init__(
            in_channels=1,
            channels=1,
            num_classes=ban_dec_cfg.num_classes,
            **kwargs)

        self.diffT = DiffTransformerBlock(**difft_cfg)
        self.mask_decoder = MODELS.build(ban_dec_cfg)

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[List]:
        """Forward function.

        Args:
            inputs (Tuple[Tensor]): A pair including images,
            list of multi-level visual features from image encoder.

        Returns:
            output (List[Tensor]): Mask predicted by BAN.
        """
        img_from, img_to, fm_feat_from, fm_feat_to = inputs

        mask_props_from, mask_props_to = self.diffT(
            fm_feat_from, fm_feat_to)

        output = self.mask_decoder(mask_props_from, mask_props_to)

        return output

    def predict(self, inputs: Tuple[torch.Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> torch.Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): Images, visual features from image encoder
            and class embedding from text encoder.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        mask_props = self.forward(inputs)

        return self.predict_by_feat(mask_props,
                                    batch_img_metas)

    def loss(self, x: Tuple[torch.Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # forward
        seg_logits = self.forward(x)

        # loss
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        return losses

