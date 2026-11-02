import math
import random
import torch
from torch import nn
import matplotlib.pyplot as plt

random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

# 模型参数
D = 32              # 每个 token 的向量维度
HEADS = 2           # 注意力头数
D_FF = 64           # FFN 中间层的维度
NUM_LAYERS = 2      # Encoder 和 Decoder 堆叠的层数
MAX_LENGTH = 32     # 一句话最长的 token 数量

en_cn_pairs = [
    ("there is a cat under the table", "桌子 下面 有 一只 猫"),
    ("there is a dog on the chair", "椅子 上面 有 一只 狗"),
    ("there is a cat on the chair", "椅子 上面 有 一只 猫"),
    ("there is a dog under the table", "桌子 下面 有 一只 狗"),
    ("there is a small cat on the table", "桌子 上面 有 一只 小 猫"),
    ("there is a small dog under the chair", "椅子 下面 有 一只 小 狗"),
    ("there is a black cat under the chair", "椅子 下面 有 一只 黑色 猫"),
    ("there is a black dog on the table", "桌子 上面 有 一只 黑色 狗"),
]

# 1. 分别收集英文和中文 token，去重、排序，建立词表
words = set()
for english, chinese in en_cn_pairs:
    for word in english.split():
        words.add(word)

source_words = sorted(words)

characters = set()
for english, chinese in en_cn_pairs:
    for character in chinese.split():
        characters.add(character)

target_words = ["<BOS>", "<EOS>"] + sorted(characters)

# 列表用于“ID -> token”，字典用于“token -> ID”
source_ids = dict()
for i, word in enumerate(source_words):
    source_ids[word] = i

target_ids = dict()
for i, character in enumerate(target_words):
    target_ids[character] = i

BOS = target_ids["<BOS>"]
EOS = target_ids["<EOS>"]

# 2. 将每对句子转换为一维整数张量，中文目标两端加入 BOS 和 EOS
examples = list()
for english, chinese in en_cn_pairs:
    source = [source_ids[word] for word in english.split()]
    target = [BOS] + [target_ids[word] for word in chinese.split()] + [EOS]
    examples.append((
        torch.tensor(source, dtype=torch.long),
        torch.tensor(target, dtype=torch.long),
    ))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        for pos in range(max_length):
            for i in range(d_model // 2):
                angle = pos / (10000.0 ** (2 * i / d_model))
                pe[pos, 2 * i] = math.sin(angle)
                pe[pos, 2 * i + 1] = math.cos(angle)
        # 固定位置表，不参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x):
        rows = list()
        for pos in range(len(x)):
            rows.append(x[pos] + self.pe[pos])
        return torch.stack(rows)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 创建形状为 [vocab_size, D] 的词向量表，每个 token ID 对应一行
        self.embedding = nn.Embedding(vocab_size, D)
        self.position = PositionalEncoding(D, MAX_LENGTH)

    def forward(self, token_ids):
        x = self.embedding(token_ids) * math.sqrt(D)
        return self.position(x) # 输出为加入位置编码之后的张量

def attention(q, k, v, causal=False):
    scores = q @ k.T / math.sqrt(q.shape[-1])

    if causal:
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_dim = D // HEADS # 计算单个头的维度
        self.q_proj = nn.Linear(D, head_dim)
        self.k_proj = nn.Linear(D, head_dim)
        self.v_proj = nn.Linear(D, head_dim)

    def forward(self, query, key, value, causal=False):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        return attention(q, k, v, causal)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(HEADS)])
        self.out_proj = nn.Linear(D, D)

    def forward(self, query, key, value, causal=False):
        outputs = list()
        for head in self.heads:
            outputs.append(head(query, key, value, causal))
        joined = torch.cat(outputs, dim=-1)
        return self.out_proj(joined)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(D, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF, D),
        )

    def forward(self, x):
        return self.network(x)

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_mha = MultiHeadAttention()
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x):
        x = self.norm1(x + self.self_mha(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.ffn = FeedForward()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.norm3 = nn.LayerNorm(D)

    def forward(self, x, encoder_output):
        x = self.norm1(x + self.self_attn(x, x, x, causal=True))

        x = self.norm2(x + self.cross_attn(x, encoder_output, encoder_output))

        x = self.norm3(x + self.ffn(x))
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_embedding = TokenEmbedding(len(source_words))
        self.target_embedding = TokenEmbedding(len(target_words))

        self.encoder = Encoder(NUM_LAYERS)
        self.decoder = Decoder(NUM_LAYERS)

        self.output_projection = nn.Linear(D, len(target_words))

    def encode(self, source):
        return self.encoder(self.source_embedding(source))

    def decode(self, target, encoder_output):
        x = self.target_embedding(target)
        x = self.decoder(x, encoder_output)
        return self.output_projection(x)

    def forward(self, source, target):
        encoder_output = self.encode(source)
        return self.decode(target, encoder_output)

model = Transformer()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

losses = list()

model.train()
for epoch in range(100):
    random.shuffle(examples)
    total_loss = 0.0
    for source, target in examples:
        logits = model(source, target[:-1])
        loss = criterion(logits, target[1:])
        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss / len(examples))
    if epoch == 0 or (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1:3d} | loss={losses[-1]:.4f}")

plt.plot(range(1, 101), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean sentence loss")
plt.show()

@torch.no_grad()
def translate(english):
    model.eval()
    source = torch.tensor([source_ids[word] for word in english.split()])
    encoder_output = model.encode(source)
    prefix = [BOS]

    for step in range(min(16, MAX_LENGTH)):
        target = torch.tensor(prefix)
        logits = model.decode(target, encoder_output)
        next_id = logits[-1].argmax().item()
        if next_id == EOS:
            break
        prefix.append(next_id)
    return "".join(target_words[i] for i in prefix[1:])

print(translate("there is a black dog under the chair"))