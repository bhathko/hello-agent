import torch
import torch.nn as nn
import math
import sys
import os

# Base import happens later to allow class definitions first if needed, 
# but here it's fine at top.
try:
    from base_chapter import BaseChapter
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base_chapter import BaseChapter

# --- 占位符模块，将在后续小节中实现 ---
# ... (rest of the classes)

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def forward(self, x):
        pass

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块
    """
    def forward(self, query, key, value, mask):
        pass

class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def forward(self, x):
        pass

# --- 编码器核心层 ---

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# --- 解码器核心层 ---

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.cross_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class TransformerSample(BaseChapter):
    def run(self):
        run_transformer_demo()

def run_transformer_demo():
    """
    Demo function for Chapter 1: Transformer Structure
    """
    print("--- Running Chapter 1: Transformer Structure Demo ---")
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    
    # Initialize an encoder layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
    print(f"Initialized EncoderLayer with d_model={d_model}, num_heads={num_heads}")
    
    # Initialize a decoder layer
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
    print(f"Initialized DecoderLayer with d_model={d_model}, num_heads={num_heads}")
    
    print("\nNote: This is a structural skeleton. MultiHeadAttention and other modules are currently placeholders.")
    print("--- Chapter 1 Demo Completed ---")
