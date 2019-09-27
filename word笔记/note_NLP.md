

余弦相似度：
$$
cos(\theta)=\frac{a\cdot b}{||a||\times ||b||}
$$


### 统计语言模型

> 把语言（词的序列）看作一个随机事件，并赋予相应的概率来描述其属于某种语言集合的可能性。给定一个词汇集合$V$，对于一个由$V$中的词构成的序列$S=(w_1,\cdots,w_T)\in V_n$，统计语言模型赋予这个序列一个概率$P(S)$，来衡量$S$符合自然语言的语法和语义规则的置信度。

一个句子的打分概率越高，越说明它是更合乎人说出来的自然句子。

常见的有N-gram model。



### Word Embedding

​	

神经网络语言模型（Neural Network Language Model, NNLM）：该模型在学习语言模型的同时也得到了词向量。



#### word2vec





​	假设其中一个长度为T的句子为$w_1,w_2,\cdots,w_T$，假定每个词都跟其相邻的词的关系最密切，换句话说每个词都是由相邻的词决定的（CBOW模型的动机），或者每个词都决定了相邻的词（Skip-gram模型的动机）。

> CBOW的输入是$w_t$周边的词，预测的输出是$w_t$的概率，Skip-gram则反之。

#### skip-gram

> learning high-quality vector representations of words from large amounts unstructured text data.



#### CBOW

> 连续词袋模型（Continuous Bag of Words）



#### item2vec



https://github.com/WillKoehrsen/wikipedia-data-science

DeepNLP的表示学习：https://blog.csdn.net/scotfield_msn/article/details/69075227