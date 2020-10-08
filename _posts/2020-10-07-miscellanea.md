---
title: "miscellanea"
tags:
  - Deep-learning
use_math: true
published : true
---

# RNN

<p align="center"> 
<img src="../images/RNNs.png" alt="drawing" width="800"/> 
<center>http://karpathy.github.io/2015/05/21/rnn-effectiveness/</center>
</p>

For example, the model opens a \begin{proof} environment but then ends it with a \end{lemma}. This is an example of a problem we’d have to fix manually, and is likely due to the fact that the dependency is too `long-term`.

<p align="center"> 
<img src="../images/RNN-seq2seq.png" alt="drawing" width="900"/> 
<center>https://github.com/tensorflow/nmt</center>
</p>

# LSTM

매 프레임 forget, input, output (input: [h,x], sigmoid output: 0~1 값) gate 세개를 생성하고, cell state에 대해 정보를 제거 (forget), 추가 (input), 출력 (output) 한다. 

# Backpropagation through time (BPTT)


# Conditional mask

<p align="center"> 
<img src="../images/maskpredict_process.gif" alt="drawing" width="800"/> 
<center>https://simonjisu.github.io/paper/2020/07/19/maskpredict.html</center>
</p>

<p align="center"> 
<img src="../images/ML.png" alt="drawing" width="800"/> 
<center>https://www.tensorflow.org/about</center>
</p>

# 용어

정제 (cleaning)
정규화 (normalization)
불용어 (stopword)
어간 추출 (stemming)
표제어 추출 (lemmatization)
정규 표현식 (regular expression)
토큰화 (tokenization)

# word2vec

각 문장에서 가까이 함께 자주 나오는 단어들을 임베딩 벡터 차원에서 유사하도록 학습. 즉, 함께 나온 단어끼리 내적하여 값이 크도록 하고, 그렇지 않은 단어들끼리는 내적값이 작도록 학습한다.

<p align="center"> 
<img src="../images/word2vec.png" alt="drawing" width="800"/> 
<center>https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b</center>
</p>

skip-gram (W와 W' 매트릭스) 학습 과정 (negative sampling):
- W 매트릭스에서 input index의 벡터를 가지고 옴 ($W_i$)
- W' 매트릭스에서 `positive` output index (input 주변의 단어들) 의 벡터를 가지고 옴 ($W'_j$)
- W' 매트릭스에서 `negative` output index (negative sampled) 의 벡터를 가지고 옴 ($W'_k$)
- (i, j)는 positive 이므로, $W_i$와 $W'_j$ 의 내적 값이 크도록 학습 ($W_i$와 $W'_j$ 값 조정)
- (i, k)는 negative 이므로, $W_i$와 $W'_k$ 의 내적 값이 작도록 학습 ($W_i$와 $W'_k$ 값 조정))

<p align="center"> 
<img src="../images/lol.png" alt="drawing" width="800"/> 
<center>'ㅋ'와 유사한 것들</center>
</p>

질문: 왜 W와 W' 의 가중치를 공유하여 학습하지 않는가??

# Metrics
perplexity, cross entropy, KL divergence, BLEU, CTC