"""
Transformer Encoder-Decoder — numpy로 처음부터 구현
지금까지 대화에서 다룬 개념을 코드로 그대로 옮긴 버전입니다.

구성 요소:
  1. Positional Encoding      (sin/cos)
  2. Scaled Dot-Product Attention  (Q, K, V, mask 지원)
  3. Multi-Head Attention     (여러 헤드 병렬 계산 -> concat -> Linear)
  4. Add & Norm               (Residual Connection + LayerNorm)
  5. Feed Forward             (Linear -> ReLU -> Linear)
  6. Encoder Layer / Decoder Layer
  7. Encoder / Decoder 스택 (N개 레이어)
  8. Transformer 전체 (Encoder + Decoder + Linear + Softmax)

실행하면 "I study AI hard" 예시로 인코더를 통과시키고,
디코더의 Masked Self-Attention이 실제로 미래를 가리는지도 확인합니다.
"""

import numpy as np

np.random.seed(42)  # 매번 같은 결과가 나오도록 고정


# ----------------------------------------------------------------------
# 1. Positional Encoding
# ----------------------------------------------------------------------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """위치마다 고유한 sin/cos 패턴 벡터를 만든다. shape: (seq_len, d_model)"""
    pos = np.arange(seq_len)[:, None]                      # (seq_len, 1)
    i = np.arange(d_model)[None, :]                         # (1, d_model)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])   # 짝수 인덱스 -> sin
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # 홀수 인덱스 -> cos
    return pe


# ----------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ----------------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)  # 수치 안정화
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (..., seq_len_q, d_k)
    K: (..., seq_len_k, d_k)
    V: (..., seq_len_k, d_v)
    mask: (seq_len_q, seq_len_k), True(또는 1)인 위치를 -inf로 마스킹
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)   # (..., seq_len_q, seq_len_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)         # -inf 대신 -1e9로 (softmax에서 사실상 0)

    weights = softmax(scores, axis=-1)                 # attention score(확률)
    output = weights @ V                                # 가중합
    return output, weights


def causal_mask(seq_len: int) -> np.ndarray:
    """디코더용 마스크: 자기 자신보다 미래 위치를 True(마스킹 대상)로 표시"""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


# ----------------------------------------------------------------------
# 3. Multi-Head Attention
# ----------------------------------------------------------------------
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어 떨어져야 함"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 헤드 하나당 차원

        scale = 0.1
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale   # concat 이후 Linear

    def split_heads(self, x, seq_len):
        # (seq_len, d_model) -> (num_heads, seq_len, d_k)
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)

    def combine_heads(self, x, seq_len):
        # (num_heads, seq_len, d_k) -> (seq_len, d_model)   <- 이게 "Concatenate"
        x = x.transpose(1, 0, 2)
        return x.reshape(seq_len, self.d_model)

    def __call__(self, x_q, x_kv, mask=None):
        """
        x_q : Query의 출처 (self-attention이면 x_kv와 동일, encoder-decoder attention이면 디코더 벡터)
        x_kv: Key/Value의 출처 (self-attention이면 x_q와 동일, encoder-decoder attention이면 인코더 출력)
        """
        seq_len_q = x_q.shape[0]
        seq_len_k = x_kv.shape[0]

        Q = x_q @ self.Wq     # (seq_len_q, d_model)
        K = x_kv @ self.Wk    # (seq_len_k, d_model)
        V = x_kv @ self.Wv    # (seq_len_k, d_model)

        Q = self.split_heads(Q, seq_len_q)   # (num_heads, seq_len_q, d_k)
        K = self.split_heads(K, seq_len_k)
        V = self.split_heads(V, seq_len_k)

        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)  # 헤드별 병렬 계산

        concat = self.combine_heads(attn_out, seq_len_q)   # "Attention Output Concatenate"
        output = concat @ self.Wo                            # "Linear transformation"
        return output, attn_weights


# ----------------------------------------------------------------------
# 4. Add & Norm (Residual Connection + LayerNorm)
# ----------------------------------------------------------------------
class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def add_and_norm(x, sublayer_output, layer_norm: LayerNorm):
    """residual connection( x + f(x) ) 후 LayerNorm 적용"""
    return layer_norm(x + sublayer_output)


# ----------------------------------------------------------------------
# 5. Feed Forward
# ----------------------------------------------------------------------
class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        scale = 0.1
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def __call__(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)   # Linear -> ReLU
        return hidden @ self.W2 + self.b2                  # Linear


# ----------------------------------------------------------------------
# 6. Encoder Layer / Decoder Layer
# ----------------------------------------------------------------------
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x):
        attn_out, attn_w = self.mha(x, x, mask=None)          # self-attention (마스크 없음)
        x = add_and_norm(x, attn_out, self.norm1)
        ffn_out = self.ffn(x)
        x = add_and_norm(x, ffn_out, self.norm2)
        return x, attn_w


class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.masked_mha = MultiHeadAttention(d_model, num_heads)   # 1) Masked self-attention
        self.cross_mha = MultiHeadAttention(d_model, num_heads)    # 2) Encoder-Decoder attention
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def __call__(self, x, enc_output):
        seq_len = x.shape[0]
        mask = causal_mask(seq_len)                                 # 미래 토큰 마스킹

        masked_attn_out, masked_w = self.masked_mha(x, x, mask=mask)
        x = add_and_norm(x, masked_attn_out, self.norm1)

        # Query는 디코더(x), Key/Value는 인코더 출력(enc_output)
        cross_attn_out, cross_w = self.cross_mha(x, enc_output, mask=None)
        x = add_and_norm(x, cross_attn_out, self.norm2)

        ffn_out = self.ffn(x)
        x = add_and_norm(x, ffn_out, self.norm3)
        return x, masked_w, cross_w


# ----------------------------------------------------------------------
# 7. Encoder / Decoder 스택 (N개 레이어)
# ----------------------------------------------------------------------
class Encoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x):
        attn_weights_per_layer = []
        for layer in self.layers:
            x, attn_w = layer(x)
            attn_weights_per_layer.append(attn_w)
        return x, attn_weights_per_layer


class Decoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, enc_output):
        masked_weights_per_layer = []
        cross_weights_per_layer = []
        for layer in self.layers:
            x, masked_w, cross_w = layer(x, enc_output)
            masked_weights_per_layer.append(masked_w)
            cross_weights_per_layer.append(cross_w)
        return x, masked_weights_per_layer, cross_weights_per_layer


# ----------------------------------------------------------------------
# 8. Transformer 전체
# ----------------------------------------------------------------------
class Transformer:
    def __init__(self, vocab_size, d_model=8, num_heads=2, d_ff=32, num_layers=2, max_len=20):
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1   # Input/Output Embedding 테이블

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        self.W_out = np.random.randn(d_model, vocab_size) * 0.1        # 최종 Linear
        self.pe_cache = positional_encoding(max_len, d_model)

    def embed(self, token_ids):
        seq_len = len(token_ids)
        x = self.embedding[token_ids]                 # Embedding lookup
        x = x + self.pe_cache[:seq_len]                # + Positional Encoding
        return x

    def forward(self, src_ids, tgt_ids):
        # ---- 인코더 ----
        enc_input = self.embed(src_ids)
        enc_output, enc_attn = self.encoder(enc_input)

        # ---- 디코더 ----
        dec_input = self.embed(tgt_ids)
        dec_output, masked_attn, cross_attn = self.decoder(dec_input, enc_output)

        # ---- Linear + Softmax ----
        logits = dec_output @ self.W_out                # (seq_len, vocab_size)
        probs = softmax(logits, axis=-1)                 # Output Probabilities

        return {
            "probs": probs,
            "enc_output": enc_output,
            "enc_attn": enc_attn,
            "masked_attn": masked_attn,
            "cross_attn": cross_attn,
        }


# ----------------------------------------------------------------------
# 데모: "I study AI hard" 예시로 실행
# ----------------------------------------------------------------------
if __name__ == "__main__":
    vocab = ["<pad>", "I", "study", "AI", "hard"]
    token2id = {w: i for i, w in enumerate(vocab)}

    src_sentence = ["I", "study", "AI", "hard"]
    tgt_sentence = ["I", "study", "AI", "hard"]   # 간단히 같은 문장으로 데모(실제론 번역 등 다른 문장)

    src_ids = [token2id[w] for w in src_sentence]
    tgt_ids = [token2id[w] for w in tgt_sentence]

    model = Transformer(vocab_size=len(vocab), d_model=8, num_heads=2, d_ff=32, num_layers=2)
    out = model.forward(src_ids, tgt_ids)

    print("=" * 60)
    print("입력 문장:", src_sentence)
    print("=" * 60)

    print("\n[인코더 1번째 레이어 Self-Attention Score] (헤드 0)")
    print(np.round(out["enc_attn"][0][0], 3))

    print("\n[디코더 1번째 레이어 Masked Self-Attention Score] (헤드 0)")
    print("-> 상삼각형(우상단)이 전부 0인지 확인 (미래 토큰 마스킹 검증)")
    print(np.round(out["masked_attn"][0][0], 3))

    print("\n[디코더 1번째 레이어 Encoder-Decoder(Cross) Attention Score] (헤드 0)")
    print(np.round(out["cross_attn"][0][0], 3))

    print("\n[최종 Output Probabilities] (각 위치에서 다음 단어일 확률, shape:", out["probs"].shape, ")")
    for i, word in enumerate(tgt_sentence):
        top_id = np.argmax(out["probs"][i])
        print(f"  위치 {i} ('{word}' 다음 예측) -> 가장 확률 높은 단어: '{vocab[top_id]}' "
              f"(확률 {out['probs'][i][top_id]:.3f})")
