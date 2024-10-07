# Cálculo de Perplexidade para Diferentes Modelos N-Gram

<imagem: Um gráfico mostrando curvas de perplexidade para modelos unigram, bigram e trigram em função do tamanho do corpus, com eixos rotulados e legendas claras>

## Introdução

A perplexidade é uma métrica fundamental na avaliação de modelos de linguagem, especialmente para modelos n-gram. Este conceito é crucial para cientistas de dados e pesquisadores em processamento de linguagem natural (NLP) que trabalham com modelos estatísticos de linguagem [1]. A perplexidade oferece uma medida quantitativa da qualidade de um modelo de linguagem, permitindo comparações objetivas entre diferentes abordagens e configurações [2].

Neste resumo, exploraremos em profundidade o cálculo de perplexidade para diferentes modelos n-gram, focando em unigrams, bigrams e modelos de ordem superior. Analisaremos as nuances matemáticas, implicações teóricas e considerações práticas envolvidas nestes cálculos.

## Conceitos Fundamentais

| Conceito         | Explicação                                                   |
| ---------------- | ------------------------------------------------------------ |
| **Perplexidade** | Medida inversa da probabilidade normalizada de um conjunto de teste, utilizada para avaliar modelos de linguagem [3]. Matematicamente, é definida como $\text{Perplexidade}(W) = P(w_1w_2...w_N)^{-\frac{1}{N}}$, onde $W$ é uma sequência de palavras e $N$ é o número de palavras [4]. |
| **N-gram**       | Modelo de linguagem que estima a probabilidade de uma palavra com base em $N-1$ palavras anteriores [5]. Unigrams consideram apenas a palavra atual, bigrams usam a palavra anterior, e assim por diante. |
| **Suavização**   | Técnicas para lidar com n-grams não observados no conjunto de treinamento, evitando probabilidades zero [6]. |

> ⚠️ **Nota Importante**: A perplexidade é inversamente relacionada à probabilidade do conjunto de teste. Quanto menor a perplexidade, melhor o modelo [7].

## Cálculo de Perplexidade para Diferentes Modelos N-gram

### Unigram

<imagem: Diagrama ilustrando o cálculo de perplexidade para um modelo unigram, mostrando a fórmula e um exemplo de cálculo passo a passo>

Para um modelo unigram, a perplexidade é calculada usando a seguinte fórmula [8]:

$$
\text{Perplexidade}(W) = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i)}}
$$

Onde:
- $W$ é a sequência de palavras no conjunto de teste
- $N$ é o número total de palavras
- $P(w_i)$ é a probabilidade unigram da palavra $w_i$

**Exemplo de cálculo:**

Considere um conjunto de teste com as seguintes probabilidades unigram:
$P(\text{the}) = 0.1$, $P(\text{cat}) = 0.05$, $P(\text{sat}) = 0.03$

Para a frase "the cat sat":

$$
\begin{aligned}
\text{Perplexidade} &= \sqrt[3]{\frac{1}{0.1} \cdot \frac{1}{0.05} \cdot \frac{1}{0.03}} \\
&= (666.67)^{\frac{1}{3}} \\
&\approx 8.74
\end{aligned}
$$

### Bigram

<imagem: Gráfico comparando a perplexidade de modelos unigram e bigram em função do tamanho do corpus de treinamento>

Para um modelo bigram, a perplexidade é calculada usando [9]:

$$
\text{Perplexidade}(W) = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i|w_{i-1})}}
$$

Onde $P(w_i|w_{i-1})$ é a probabilidade condicional da palavra $w_i$ dado $w_{i-1}$.

**Exemplo de cálculo:**

Considere as seguintes probabilidades bigram:
$P(\text{cat}|\text{the}) = 0.2$, $P(\text{sat}|\text{cat}) = 0.1$, $P(\text{</s>}|\text{sat}) = 0.5$

Para a frase "the cat sat":

$$
\begin{aligned}
\text{Perplexidade} &= \sqrt[3]{\frac{1}{0.2} \cdot \frac{1}{0.1} \cdot \frac{1}{0.5}} \\
&= (100)^{\frac{1}{3}} \\
&\approx 4.64
\end{aligned}
$$

### Modelos de Ordem Superior

Para n-grams de ordem superior (trigrams, 4-grams, etc.), a fórmula geral é [10]:

$$
\text{Perplexidade}(W) = \sqrt[N]{\prod_{i=1}^N \frac{1}{P(w_i|w_{i-N+1:i-1})}}
$$

Onde $P(w_i|w_{i-N+1:i-1})$ é a probabilidade condicional da palavra $w_i$ dado as $N-1$ palavras anteriores.

> ✔️ **Destaque**: À medida que aumentamos a ordem do n-gram, geralmente observamos uma diminuição na perplexidade, indicando um melhor ajuste aos dados de teste [11].

#### Perguntas Teóricas

1. Derive a fórmula da perplexidade a partir da definição de entropia cruzada para um modelo de linguagem n-gram.

2. Demonstre matematicamente por que a perplexidade tende a diminuir com o aumento da ordem do n-gram, e discuta as limitações teóricas dessa tendência.

3. Analise teoricamente o impacto da suavização add-k na perplexidade de um modelo bigram, comparando-o com um modelo não suavizado.

## Considerações Práticas no Cálculo de Perplexidade

### Tratamento de Palavras Desconhecidas

Um desafio significativo no cálculo de perplexidade é lidar com palavras que não aparecem no conjunto de treinamento, mas estão presentes no conjunto de teste [12]. Existem várias abordagens para lidar com este problema:

1. **Vocabulário Fechado**: Substitua todas as palavras desconhecidas por um token especial <UNK> e trate-o como qualquer outra palavra no vocabulário [13].

2. **Suavização**: Utilize técnicas de suavização como add-k ou interpolação para atribuir probabilidades não nulas a n-grams não observados [14].

3. **Backoff**: Recorra a n-grams de ordem inferior quando um n-gram de ordem superior não é observado [15].

### Impacto do Tamanho do Corpus

O tamanho do corpus de treinamento tem um impacto significativo na perplexidade calculada [16]. Geralmente, observa-se:

- Corpora maiores tendem a resultar em perplexidades mais baixas.
- A relação entre o tamanho do corpus e a perplexidade geralmente segue uma curva logarítmica.

<imagem: Gráfico mostrando a relação entre o tamanho do corpus de treinamento e a perplexidade para diferentes ordens de n-gram>

```python
import numpy as np
import matplotlib.pyplot as plt

def perplexity_vs_corpus_size(corpus_sizes, n_gram_orders):
    perplexities = {n: np.log(corpus_sizes) / np.log(n) for n in n_gram_orders}
    
    plt.figure(figsize=(10, 6))
    for n, perp in perplexities.items():
        plt.plot(corpus_sizes, perp, label=f'{n}-gram')
    
    plt.xscale('log')
    plt.xlabel('Tamanho do Corpus')
    plt.ylabel('Perplexidade')
    plt.title('Perplexidade vs. Tamanho do Corpus para Diferentes N-grams')
    plt.legend()
    plt.grid(True)
    plt.show()

corpus_sizes = np.logspace(3, 8, 100)
n_gram_orders = [1, 2, 3, 4]
perplexity_vs_corpus_size(corpus_sizes, n_gram_orders)
```

Este código gera um gráfico teórico da relação entre o tamanho do corpus e a perplexidade para diferentes ordens de n-gram, ilustrando como a perplexidade geralmente diminui com o aumento do tamanho do corpus e da ordem do n-gram [17].

### Interpolação de Modelos

A interpolação de modelos n-gram de diferentes ordens pode melhorar o desempenho geral e, consequentemente, reduzir a perplexidade [18]. A fórmula geral para interpolação linear é:

$$
\hat{P}(w_n|w_{n-2}w_{n-1}) = \lambda_1P(w_n) + \lambda_2P(w_n|w_{n-1}) + \lambda_3P(w_n|w_{n-2}w_{n-1})
$$

Onde $\lambda_1 + \lambda_2 + \lambda_3 = 1$ e cada $\lambda$ é determinado por validação cruzada [19].

> ❗ **Ponto de Atenção**: A escolha dos valores de $\lambda$ é crucial e pode impactar significativamente a perplexidade resultante [20].

#### Perguntas Teóricas

1. Derive a fórmula para calcular a perplexidade de um modelo interpolado, considerando as perplexidades individuais dos modelos componentes e seus pesos de interpolação.

2. Analise teoricamente como a perplexidade de um modelo interpolado se comporta em relação às perplexidades dos modelos individuais que o compõem.

3. Demonstre matematicamente por que a interpolação tende a produzir modelos com menor perplexidade, e discuta as condições sob as quais isso pode não ser verdade.

## Conclusão

O cálculo de perplexidade para diferentes modelos n-gram é uma ferramenta essencial na avaliação e comparação de modelos de linguagem. Compreender as nuances matemáticas e considerações práticas envolvidas neste processo é crucial para desenvolver e selecionar modelos eficazes [21].

Observamos que:
1. A perplexidade diminui com o aumento da ordem do n-gram, refletindo uma melhor modelagem das dependências linguísticas [22].
2. O tratamento adequado de palavras desconhecidas e a escolha de técnicas de suavização são cruciais para cálculos precisos de perplexidade [23].
3. A interpolação de modelos oferece um meio de combinar as forças de diferentes ordens de n-gram, potencialmente resultando em modelos com menor perplexidade [24].

À medida que avançamos para modelos de linguagem mais complexos, como os baseados em redes neurais, a perplexidade continua sendo uma métrica valiosa, embora outras métricas possam se tornar igualmente importantes [25].

## Perguntas Teóricas Avançadas

1. Derive a relação teórica entre a perplexidade e a entropia cruzada para um modelo de linguagem n-gram, e discuta como essa relação se mantém ou se modifica para modelos de linguagem neurais.

2. Desenvolva uma prova matemática para demonstrar que, sob certas condições, a perplexidade de um modelo n-gram interpolado é sempre menor ou igual à menor perplexidade dos modelos individuais que o compõem.

3. Analise teoricamente o impacto da esparsidade dos dados na perplexidade de modelos n-gram de ordem superior. Derive uma expressão que relacione a ordem do n-gram, o tamanho do vocabulário e o tamanho do corpus com a esparsidade esperada e seu efeito na perplexidade.

4. Proponha e justifique matematicamente uma nova métrica que combine perplexidade com outras medidas de desempenho do modelo de linguagem, como cobertura do vocabulário ou sensibilidade contextual.

5. Desenvolva um framework teórico para analisar o trade-off entre a complexidade computacional do cálculo de perplexidade e a ordem do n-gram. Inclua considerações sobre espaço de armazenamento e tempo de processamento.

## Referências

[1] "Perplexity is one of the most important metrics in NLP, used for evaluating large language models as well as n-gram models." *(Trecho de n-gram language models.pdf.md)*

[2] "The perplexity measure actually arises from the information-theoretic concept of cross-entropy, which explains otherwise mysterious properties of perplexity (why the inverse probability, for example?) and its relationship to entropy." *(Trecho de n-gram language models.pdf.md)*

[3] "The perplexity (sometimes abbreviated as PP or PPL) of a language model on a test set is the inverse probability of the test set (one over the probability of the test set), normalized by the number of words (or tokens)." *(Trecho de n-gram language models.pdf.md)*

[4] "For a test set W = w1w2...wN: perplexity(W) = P(w1w2...wN)^(-1/N)" *(Trecho de n-gram language models.pdf.md)*

[5] "N-grams are perhaps the simplest kind of language model. They are Markov models that estimate words from a fixed window of previous words." *(Trecho de n-gram language models.pdf.md)*

[6] "Smoothing algorithms provide a way to estimate probabilities for events that were unseen in training." *(Trecho de n-gram language models.pdf.md)*

[7] "Note that because of the inverse in Eq. 3.15, the higher the probability of the word sequence, the lower the perplexity. Thus the the lower the perplexity of a model on the data, the better the model." *(Trecho de n-gram language models.pdf.md)*

[8] "The perplexity of W computed with a unigram language model is still a geometric mean, but now of the inverse of the unigram probabilities: perplexity(W) = (Π(i=1 to N) 1/P(wi))^(1/N)" *(Trecho de n-gram language models.pdf.md)*

[9] "The perplexity of W computed with a bigram language model is still a geometric mean, but now of the inverse of the bigram probabilities: perplexity(W) = (Π(i=1 to N) 1/P(wi|wi-1))^(1/N)" *(Trecho de n-gram language models.pdf.md)*

[10] "For the general case of MLE n-gram parameter estimation: P(wn|wn-N+1:n-1) = C(wn-N+1:n-1wn) / C(wn-N+1:n-1)" *(Trecho de n-gram language models.pdf.md)*

[11] "As we see above, the more information the n-gram gives us about the word sequence, the higher the probability the n-gram will assign to the string. A trigram model is less surprised than a unigram model because it has a better idea of what words might come next, and so it assigns them a higher probability." *(Trecho de n-gram language models.pdf.md)*

[12] "There is a problem with using maximum likelihood estimates for probabilities: any finite training corpus will be missing some perfectly acceptable English word sequences." *(Trecho de