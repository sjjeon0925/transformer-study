# Transformer from Scratch (NumPy)

논문 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)의 Encoder-Decoder Transformer 구조를
**딥러닝 프레임워크 없이 순수 NumPy만으로** 처음부터 구현한 교육용 프로젝트입니다.

PyTorch/TensorFlow의 `nn.MultiheadAttention` 같은 블랙박스를 쓰지 않고,
Q/K/V 행렬 연산부터 Masking, Positional Encoding, 역전파(gradient)까지
**모든 계산을 직접 눈으로 확인할 수 있도록** 만들었습니다.

---

## 왜 만들었나

Attention이 "그래서 실제로 숫자가 어떻게 움직이는지" 이해하고 싶어서 만든 프로젝트입니다.
아래 두 가지 질문에 코드로 답합니다.

1. Multi-Head Attention의 Q, K, V는 행렬 연산으로 정확히 어떻게 계산되는가?
2. Masked Attention은 "미래를 못 보게 막는다"를 실제로 어떻게 구현하는가?

---

## 파일 구성

| 파일 | 내용 |
|---|---|
| `transformer_from_scratch.py` | Transformer 전체 구조 (Encoder, Decoder, Multi-Head Attention, Masking, Positional Encoding, FFN, Add & Norm) |
| `train_demo.py` | 위 모델을 실제로 학습시켜, Loss가 줄어들고 다음 단어 예측이 맞아가는 과정을 확인하는 데모 |
| `transformer_training.ipynb` | 같은 개념을 PyTorch(autograd, GPU)로 옮긴 GPT-style Decoder-only 모델. Colab에서 tiny-shakespeare 데이터로 실전 학습 |

---

## 아키텍처

```
Input                                    Output (shifted right)
  │                                             │
Input Embedding + Positional Encoding    Output Embedding + Positional Encoding
  │                                             │
┌─────────────────┐                     ┌──────────────────────┐
│  Multi-Head      │                     │  Masked Multi-Head    │
│  Self-Attention   │                    │  Self-Attention        │
│  Add & Norm       │                    │  Add & Norm             │
│                   │                    │  Multi-Head Attention   │  ← Query: Decoder
│  Feed Forward     │  ── K, V ────────▶ │  (Encoder-Decoder)      │  ← Key/Value: Encoder 출력
│  Add & Norm       │                    │  Add & Norm             │
└─────────────────┘   × N                │  Feed Forward           │
    (Encoder)                            │  Add & Norm             │
                                          └──────────────────────┘   × N
                                                    │                (Decoder)
                                              Linear + Softmax
                                                    │
                                            Output Probabilities
```

### 구현된 핵심 컴포넌트

- **Positional Encoding** — `sin`/`cos` 기반 위치 정보 주입 (Attention은 순서를 모르기 때문)
- **Scaled Dot-Product Attention** — `softmax(QKᵀ / √d_k) · V`
- **Multi-Head Attention** — 여러 헤드가 서로 다른 관점(W^Q/W^K/W^V)으로 병렬 계산 후 Concat → Linear
- **Masked Self-Attention (Decoder)** — 상삼각형 마스크로 미래 토큰을 `-∞` 처리 → Softmax 후 확률 0
- **Encoder-Decoder (Cross) Attention** — Query는 Decoder, Key/Value는 Encoder 출력에서 가져옴
- **Add & Norm** — Residual Connection(`x + Sublayer(x)`) + LayerNorm → 기울기 소실 방지, 정보 보존
- **Feed Forward** — `Linear → ReLU → Linear`로 비선형성 추가

---

## 실행 방법

```bash
# 요구 사항: numpy만 있으면 됩니다
pip install numpy

# 1) Forward pass만 실행 (구조 확인용)
python transformer_from_scratch.py

# 2) 실제 학습 데모 (Loss가 줄어드는 것 확인)
python train_demo.py
```

---

## 실행 예시

예제 문장 `"I study AI hard"`로 인코더/디코더를 통과시킨 결과입니다.

```
[디코더 1번째 레이어 Masked Self-Attention Score] (헤드 0)
-> 상삼각형(우상단)이 전부 0인지 확인 (미래 토큰 마스킹 검증)
[[1.    0.    0.    0.   ]
 [0.5   0.5   0.    0.   ]
 [0.333 0.334 0.333 0.   ]
 [0.249 0.25  0.251 0.25 ]]
```

`I` 시점에서는 자기 자신만(1.0), `study` 시점에서는 `I, study`까지만 보고 그 뒤는 정확히 `0`으로
마스킹되는 것을 확인할 수 있습니다.

`train_demo.py`를 실행하면 다음과 같이 학습이 진행됩니다.

```
step  0  |  loss = 1.7507
step  5  |  loss = 0.2361
step 10  |  loss = 0.0887
step 15  |  loss = 0.0505
step 20  |  loss = 0.0346
step 25  |  loss = 0.0262
step 29  |  loss = 0.0219

입력까지: I               -> 예측: 'study' (정답: 'study')  [O]
입력까지: I study         -> 예측: 'AI'    (정답: 'AI')     [O]
입력까지: I study AI      -> 예측: 'hard'  (정답: 'hard')   [O]
입력까지: I study AI hard -> 예측: 'hard'  (정답: 'hard')   [O]
```

---

## 역전파(학습) 방식에 대한 노트

`train_demo.py`는 이해를 돕기 위해 **수치 미분(numerical gradient)**을 사용합니다.

```
f'(w) ≈ (f(w + h) - f(w - h)) / (2h)
```

Chain Rule을 손으로 유도해서 analytic하게 미분하는 대신, 각 가중치를 아주 조금 흔들어보고
Loss가 어떻게 변하는지로 기울기를 근사합니다. 원리는 analytic backprop과 동일하지만
계산 속도가 느려서, 실제 프로덕션에서는 PyTorch/TensorFlow의 **autograd**(analytic 미분을
자동화한 엔진)를 사용합니다. 이 저장소는 학습용으로 원리를 눈으로 보는 데 목적이 있습니다.

---

## 의도적으로 단순화한 부분

이 구현은 **원리 이해**가 목적이라 실제 프로덕션 Transformer와 다음과 같은 차이가 있습니다.

- Byte-Pair Encoding 등의 실제 토크나이저 대신 단어 단위의 작은 vocab 사용
- Dropout, Warmup Learning Rate Schedule 등 학습 안정화 기법 미포함
- Batch 처리 없이 문장 1개 단위로만 동작
- 학습은 analytic backprop이 아닌 수치 미분으로 근사

더 큰 데이터셋이나 실전 학습이 필요하다면 PyTorch로 이식하는 것을 권장합니다.

---

## PyTorch로 실전 학습해보기 (`transformer_training.ipynb`)

위 NumPy 구현의 개념(Q/K/V, Masked Self-Attention, Add & Norm, FeedForward)을 그대로 PyTorch로 옮기고,
Decoder만 사용하는 **GPT-style 모델**로 확장했습니다. tiny-shakespeare 텍스트를 글자 단위로 토큰화해서 학습시키며,
수치 미분 대신 PyTorch의 **autograd**와 GPU 연산을 사용합니다.

```
Colab에서: 런타임 → 런타임 유형 변경 → GPU(T4) 선택 후, 셀을 순서대로 실행
```

---

## 참고

- Vaswani et al., *Attention Is All You Need* (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## License

MIT