import torch
import torch.nn as nn
import math
from base_example import BaseExample

# ==========================================
# A Simple Transformer Story: Flow and Duty
# ==========================================
# Imagine a Transformer as a factory that translates or understands language.
# The flow of raw materials (words/tokens) goes through several specialized processing stations.
#
# 1. Positional Encoding: Words are just IDs. We need to tell the model *where* each word is located in the sentence.
#    Duty: Stamp each word with a "timestamp" or "position ticket".
#
# 2. Multi-Head Attention: In a factory meeting, workers need to listen to each other to understand the context.
#    Duty: Allow each word to look around at other words in the sentence to understand the full context.
#
# 3. Position-Wise Feed-Forward: After talking (Attention), each worker goes back to their desk to process what they heard.
#    Duty: Independently process the context-rich information for each position.
#
# 4. Encoder Layer: The combination of Attention and Feed-Forward.
#    Duty: Extract fully contextualized features from the input sentence.
#
# 5. Decoder Layer: The worker building the output sentence one word at a time.
#    Duty: Look at the input features (from Encoder) and the words produced so far, to decide what the next word should be.
# ==========================================


class PositionalEncoding(nn.Module):
    """
    第一站：位置編碼 (Positional Encoding)
    Duty: 給每個輸入的詞向量打上位置戳記。
    因為 Transformer 是同時處理所有詞的（不像 RNN 是逐個處理），
    所以它失去了詞的前後順序資訊。位置編碼負責用正弦和餘弦函數，給不同位置加上獨一無二的特徵。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 創建一個足夠長的位置編碼矩陣
        position = torch.arange(max_len).unsqueeze(1)
        # 用對數空間計算 div_term 提升數值穩定性
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小為 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶數維度使用 sin, 奇數維度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 將 pe 註冊為 buffer，這樣它就不會被視為模型參數，但會隨模型移動到 GPU/CPU
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形狀: (batch_size, seq_len, d_model)
        # 將位置編碼加到輸入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    第二站：多頭注意力機制 (Multi-Head Attention)
    Duty: 讓每個詞都能「看到」句子中的其他詞，並決定哪些詞跟自己關係最密切。
    這就像是在分析句法結構，比如「蘋果」這個詞到底是手機品牌還是水果，
    透過注意力機制看周圍的詞就能判斷出來。
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 定義 Q, K, V 和輸出的線性變換層
        # Query (Q): 當前詞正在尋找什麼資訊
        # Key (K): 其他詞能提供什麼資訊
        # Value (V): 實際包含的資訊內容
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 1. 計算注意力得分 (QK^T)
        # 兩個詞的向量點乘越大，說明它們越相關。除以 sqrt(d_k) 是為了防止點乘結果過大導致梯度消失
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 應用遮罩 (如果提供)
        # 在解碼器中，我們不能看未來的詞，所以要把被 Mask 掉的無關位置設置為極小值 (-1e9)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 計算注意力權重 (Softmax)
        # 把得分轉換成機率，使每個詞對其他詞的注意力權重之和等於 1
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加權求和 (權重 * V)
        # 按照得出的關注度權重，把所有相關的詞的資訊彙聚起來，作為當前的輸出
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # 多頭機制：把向量切分到多個「頭」中，讓模型可以同時在不同的表現子空間學習到不同的注意力關聯
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # 把多個頭的計算結果拼接回原來的維度
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    """
    第三站：位置前饋網路 (Position-Wise Feed-Forward Network)
    Duty: 經過注意力機制交流後，每個詞拿到了全局語境，現在需要「自我消化」。
    前饋網路就像是單兵作戰，對每個詞的向量進行非線性變換，提取更深層的特徵，讓模型擁有複雜的表達能力。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # 經典的架構是先經過線性層進行升維 (d_ff 通常是 d_model 的四倍)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # 激活函數進行非線性變換，然後再透過第二層線性層降回到原維度
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形狀: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    第四站：編碼器層 (Encoder Layer)
    Duty: 將上述的模塊組合起來！
    就像工廠流水線上的一個大車間，它將輸入的原材料（句子）徹底分析透徹，提取出最核心的語義資訊。
    在真正的 Transformer 中，往往會將 6 個甚至更多這樣的 EncoderLayer 堆疊起來，層數越深，理解越透徹。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 層歸一化 (LayerNorm)：流水線上的「質檢員」，控制特徵值的變異程度，讓網路更容易訓練收斂
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 多頭自注意力：詞與詞相互關聯，獲取長短距離的全局上下文
        attn_output = self.self_attn(x, x, x, mask)
        # 殘差連接 (x + ...) 保證了即使堆疊多層網路，原始的特徵資訊也不會在複雜的傳遞過程中丟失或導致梯度消失
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前饋網路：每個位置的詞向量獨立進行深層次特徵提取
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    第五站：解碼器層 (Decoder Layer)
    Duty: 根據編碼器提煉的「源語言上下文精華」，以及「目前已經生成出的詞彙」，來預測下一個詞。
    相比編碼器，它的結構多了一個連接編碼器輸出的交叉注意力層 (Cross-Attention)。
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 帶遮罩的多頭自注意力 (Masked Self-Attention)
        # tgt_mask 很關鍵，它的目的是防止「作弊/偷看」。在預測第 t 個詞時，注意力只能關注 1 ~ t-1 的詞，不能看到 t+1 及以後的解。
        attn_output = self.self_attn(Q=x, K=x, V=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉多頭注意力 (Cross-Attention)
        # Query (來自解碼器自身): 是告訴模型「我剛剛生成了這些詞，現在我要準備下個詞了，我需要原文的什麼資訊？」
        # Key 和 Value (來自編碼器提取的最終特徵): 就是整句原文提供的最全面豐富的語境和詞彙表示資訊。
        # 這裡就是在追溯並查找：我目前生成的句子段落在原文中重點對應哪裡。
        cross_attn_output = self.cross_attn(Q=x, K=encoder_output, V=encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前饋網路：類似地，繼續增加非線性維度，消化整合剛剛的資訊交互結果
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class TransformerSample(BaseExample):
    """
    Example 1: Transformer Structure Sample
    """
    def run(self):
        print("=== Transformer 故事：工廠流水線開始運作 ===\n")

        # 1. 準備參數
        batch_size = 2
        seq_len = 5     # 假設句子有 5 個詞
        d_model = 64    # 每個詞被表示為一個 64 維的向量
        num_heads = 8   # 把 64 維拆成 8 個頭，每個頭 8 維
        d_ff = 256      # 前饋網路隱藏層升維到 256
        dropout = 0.1

        print(f"📌 設定參數：Batch Size={batch_size}, 句子長度={seq_len}, 詞向量維度={d_model}")
        
        # 2. 隨機生成輸入（模擬經過 Word Embedding 後的詞向量）
        src_input = torch.randn(batch_size, seq_len, d_model)
        # 假設這是在解碼器逐步生成的輸入
        tgt_input = torch.randn(batch_size, seq_len, d_model) 
        print(f"📦 原始輸入的形狀 (未加位置編碼): {src_input.shape}\n")

        # ==========================================
        # 第一站：位置編碼 (Positional Encoding)
        # ==========================================
        print("📍 [第一站：位置編碼]")
        pe = PositionalEncoding(d_model, dropout)
        src_embedded = pe(src_input)
        tgt_embedded = pe(tgt_input)
        print("   -> 已經給每個詞打上了『位置戳記』！")
        print(f"   -> 加上位置編碼後的形狀: {src_embedded.shape}\n")

        # ==========================================
        # 中間：產生遮罩 (Masks)
        # ==========================================
        src_mask = None
        tgt_mask = None

        # ==========================================
        # 第二到第四站：編碼器層 (Encoder Layer)
        # ==========================================
        print("⚙️  [第二到第四站：編碼器 (Encoder) 處理]")
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        encoder_output = encoder_layer(x=src_embedded, mask=src_mask)
        print("   -> 編碼器已充分吸收輸入句子的所有語境，提取出精華資訊！")
        print(f"   -> 編碼器輸出的形狀: {encoder_output.shape}\n")

        # ==========================================
        # 第五站：解碼器層 (Decoder Layer)
        # ==========================================
        print("🏗  [第五站：解碼器 (Decoder) 處理]")
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_output = decoder_layer(
            x=tgt_embedded, 
            encoder_output=encoder_output, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask
        )
        print("   -> 解碼器已結合原文精華和目前進度，準備好預測下一個詞！")
        print(f"   -> 解碼器輸出的形狀: {decoder_output.shape}\n")

        print("=== Transformer 故事：工廠流水線完成本次作業！ ===")
