"""
transformer_from_scratch.py 의 모델을 실제로 '학습'시켜보는 데모.

지금까지 대화에서 다룬 역전파(backpropagation) 원리를 그대로 적용합니다:
  1. Forward pass로 예측 확률 계산
  2. 정답(next token)과 비교해서 Cross-Entropy Loss 계산
  3. 그 Loss를 각 가중치에 대해 미분(기울기 계산) -> 가중치 업데이트
  4. 이 과정을 반복하면 Loss가 점점 줄어드는 것을 확인

주의: 매 레이어를 수식으로 손으로 미분(analytic backprop)하는 대신,
여기서는 이해하기 쉬운 '수치 미분(numerical gradient)'을 사용합니다.
    f'(w) ≈ (f(w+h) - f(w-h)) / (2h)
원리는 우리가 손으로 계산했던 chain rule과 동일한 결과를 근사치로 얻는 방법입니다.
(실무에서는 PyTorch/TensorFlow의 autograd가 이걸 analytic하게, 훨씬 빠르게 해줍니다.)
"""

import numpy as np
from transformer_from_scratch import Transformer, softmax

np.random.seed(0)

# ----------------------------------------------------------------------
# 데이터 준비: "I study AI hard" -> 다음 토큰 예측 과제
# ----------------------------------------------------------------------
vocab = ["<pad>", "I", "study", "AI", "hard"]
token2id = {w: i for i, w in enumerate(vocab)}

src_sentence = ["I", "study", "AI", "hard"]
tgt_input    = ["I", "study", "AI", "hard"]     # 디코더 입력
tgt_output   = ["study", "AI", "hard", "hard"]  # 정답(한 칸씩 밀린 다음 단어). 마지막은 그대로 반복.

src_ids = [token2id[w] for w in src_sentence]
tgt_in_ids = [token2id[w] for w in tgt_input]
tgt_out_ids = [token2id[w] for w in tgt_output]


def cross_entropy_loss(probs, target_ids):
    """probs: (seq_len, vocab_size), target_ids: (seq_len,)"""
    seq_len = len(target_ids)
    eps = 1e-9
    picked = probs[np.arange(seq_len), target_ids]   # 정답 위치의 확률만 뽑기
    return -np.mean(np.log(picked + eps))


def loss_fn(model, src_ids, tgt_in_ids, tgt_out_ids):
    out = model.forward(src_ids, tgt_in_ids)
    return cross_entropy_loss(out["probs"], tgt_out_ids)


def numerical_gradient(model, param, loss_fn_args, h=1e-4):
    """주어진 파라미터 행렬 하나에 대한 기울기를 수치미분으로 근사"""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]

        param[idx] = original + h
        loss_plus = loss_fn(model, *loss_fn_args)

        param[idx] = original - h
        loss_minus = loss_fn(model, *loss_fn_args)

        param[idx] = original  # 원상복구
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
        it.iternext()
    return grad


# ----------------------------------------------------------------------
# 학습 대상 파라미터 선정
#   -> 전체 파라미터(임베딩, 모든 레이어의 Wq/Wk/Wv/Wo, FFN 등)를 다 학습시키는 게
#      이론적으로 맞지만, 데모 속도를 위해 핵심 파라미터 몇 개만 업데이트합니다.
#      (원리는 100개든 100만개든 동일합니다)
# ----------------------------------------------------------------------
model = Transformer(vocab_size=len(vocab), d_model=8, num_heads=2, d_ff=32, num_layers=1)

trainable_params = {
    "embedding": model.embedding,
    "W_out": model.W_out,
    "enc_Wq": model.encoder.layers[0].mha.Wq,
    "enc_Wk": model.encoder.layers[0].mha.Wk,
    "enc_Wv": model.encoder.layers[0].mha.Wv,
    "dec_masked_Wq": model.decoder.layers[0].masked_mha.Wq,
    "dec_masked_Wk": model.decoder.layers[0].masked_mha.Wk,
    "dec_cross_Wq": model.decoder.layers[0].cross_mha.Wq,
}

learning_rate = 0.5
num_steps = 30

print("=" * 60)
print("학습 시작 -> 목표: '다음 토큰 예측' Loss를 줄이기")
print("=" * 60)

loss_history = []
for step in range(num_steps):
    current_loss = loss_fn(model, src_ids, tgt_in_ids, tgt_out_ids)
    loss_history.append(current_loss)

    # 각 파라미터마다 기울기 계산 -> 업데이트 (경사하강법, 지금까지 손으로 하던 것과 동일한 원리)
    for name, param in trainable_params.items():
        grad = numerical_gradient(model, param, (src_ids, tgt_in_ids, tgt_out_ids))
        param -= learning_rate * grad          # w_new = w_old - lr * gradient

    if step % 5 == 0 or step == num_steps - 1:
        print(f"step {step:2d}  |  loss = {current_loss:.4f}")

print("\n" + "=" * 60)
print("학습 후 예측 확인")
print("=" * 60)
out = model.forward(src_ids, tgt_in_ids)
for i, word in enumerate(tgt_input):
    pred_id = np.argmax(out["probs"][i])
    true_id = tgt_out_ids[i]
    mark = "O" if pred_id == true_id else "X"
    print(f"  입력까지: {' '.join(tgt_input[:i+1]):15s} -> 예측: '{vocab[pred_id]}' "
          f"(정답: '{vocab[true_id]}')  [{mark}]")

print(f"\n최초 Loss: {loss_history[0]:.4f}  ->  최종 Loss: {loss_history[-1]:.4f}")
print("Loss가 줄었다면, 역전파를 통해 Q/K/V 가중치와 임베딩이")
print("'I->study->AI->hard' 패턴을 반영하는 방향으로 실제로 조정된 것입니다.")
