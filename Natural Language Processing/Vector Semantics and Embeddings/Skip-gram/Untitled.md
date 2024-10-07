# Skip-gram com Amostragem Negativa (SGNS): Fundamentos, Demonstração Passo a Passo e Exemplo Numérico

## Introdução

O **Skip-gram com Amostragem Negativa (SGNS)** é um modelo essencial no campo do Processamento de Linguagem Natural (PLN) para a geração de *word embeddings*—representações vetoriais densas de palavras. Desenvolvido como parte do *Word2Vec*, o SGNS revolucionou a forma como capturamos relações semânticas e sintáticas entre palavras em um espaço vetorial contínuo, permitindo que máquinas compreendam e processem a linguagem humana de maneira mais eficaz.

Este documento fornece uma explicação detalhada e aprofundada do SGNS, demonstrando passo a passo sua fundamentação teórica, o processo de treinamento e como utilizá-lo. Além disso, um exemplo numérico intuitivo é fornecido para ilustrar os conceitos e facilitar a compreensão.

## Conceitos Fundamentais

Antes de mergulharmos no SGNS, é importante compreender alguns conceitos fundamentais:

- **Word Embedding**: Representação vetorial densa de palavras em um espaço multidimensional. Permite que palavras com significados semelhantes tenham representações vetoriais próximas.
- **Modelo Skip-gram**: Um modelo que, dada uma palavra-alvo, prevê as palavras de contexto. Diferentemente dos modelos de linguagem tradicionais que preveem a próxima palavra, o Skip-gram inverte essa abordagem.
- **Amostragem Negativa**: ==Técnica que melhora a eficiência do treinamento selecionando "palavras de ruído" (negativas) para contrastar com as palavras de contexto reais (positivas)==, evitando a necessidade de computar atualizações para todo o vocabulário.

## Fundamentos Teóricos do SGNS

### Objetivo do Modelo

O objetivo do SGNS é aprender vetores de palavras que são bons em prever o contexto em que as palavras aparecem. Isto é feito através da maximização da probabilidade de palavras de contexto reais e da minimização da probabilidade de palavras de ruído.

### Função de Probabilidade

Para ==uma palavra-alvo $w$ e uma palavra de contexto $c$, a probabilidade de $c$ ser um contexto verdadeiro de $w$ é modelada usando a função sigmoide==:

$$
P(D = 1 | w, c) = \sigma(\mathbf{v}_c \cdot \mathbf{v}_w) = \frac{1}{1 + e^{-\mathbf{v}_c \cdot \mathbf{v}_w}}
$$

Onde:

- $\sigma(x)$ é a função sigmoide.
- $\mathbf{v}_w$ é o vetor de embedding da palavra-alvo $w$.
- $\mathbf{v}_c$ é o vetor de embedding da palavra de contexto $c$.
- $D$ é uma variável binária indicando se $c$ é um contexto verdadeiro de $w$ ($D = 1$) ou uma palavra de ruído ($D = 0$).

### Função de Perda

A função de perda para um par $(w, c)$ com $K$ amostras negativas $\{n_1, n_2, \dots, n_K\}$ é dada por:

$$
L = -\log \sigma(\mathbf{v}_c \cdot \mathbf{v}_w) - \sum_{i=1}^{K} \log \sigma(-\mathbf{v}_{n_i} \cdot \mathbf{v}_w)
$$

Esta função penaliza o modelo quando ele atribui alta probabilidade a palavras de ruído e baixa probabilidade a palavras de contexto reais.

### Processo de Treinamento

1. **Seleção da Palavra-Alvo e Contexto**: Para cada palavra-alvo $w$ no corpus, selecione palavras de contexto $c$ dentro de uma janela de contexto definida (por exemplo, duas palavras antes e depois de $w$).

2. **Geração de Amostras Negativas**: Para cada par $(w, c)$, amostre $K$ palavras de ruído $\{n_1, n_2, \dots, n_K\}$ a partir de uma distribuição de probabilidade específica.

3. **Atualização dos Vetores**: Use métodos de otimização, como a Descida de Gradiente Estocástica (SGD), para atualizar os vetores de embedding $\mathbf{v}_w$, $\mathbf{v}_c$ e $\mathbf{v}_{n_i}$ de modo a minimizar a função de perda $L$.

### Atualização dos Vetores

As atualizações dos vetores de embedding são realizadas conforme as derivadas parciais da função de perda em relação a cada vetor:

1. **Para a Palavra de Contexto Positiva $c$**:

$$
\mathbf{v}_c \leftarrow \mathbf{v}_c + \eta (1 - \sigma(\mathbf{v}_c \cdot \mathbf{v}_w)) \mathbf{v}_w
$$

2. **Para Cada Palavra de Ruído $n_i$**:

$$
\mathbf{v}_{n_i} \leftarrow \mathbf{v}_{n_i} + \eta (0 - \sigma(-\mathbf{v}_{n_i} \cdot \mathbf{v}_w)) \mathbf{v}_w
$$

3. **Para a Palavra-Alvo $w$**:

$$
\mathbf{v}_w \leftarrow \mathbf{v}_w + \eta \left[ (1 - \sigma(\mathbf{v}_c \cdot \mathbf{v}_w)) \mathbf{v}_c + \sum_{i=1}^{K} (0 - \sigma(-\mathbf{v}_{n_i} \cdot \mathbf{v}_w)) \mathbf{v}_{n_i} \right]
$$

Onde $\eta$ é a taxa de aprendizado.

## Exemplo Numérico Intuitivo

Vamos ilustrar o funcionamento do SGNS com um exemplo simplificado.

### Cenário

- **Corpus**: "gato senta no tapete"
- **Vocabulário**: {gato, senta, no, tapete}
- **Dimensão dos Embeddings**: 2 (para simplificar)
- **Janela de Contexto**: 1 (uma palavra antes e depois)
- **Amostras Negativas**: 1 por par positivo

### Passo a Passo

1. **Inicialização dos Vetores**

   - Inicialize vetores de embedding aleatórios para cada palavra. Por exemplo:

     - $\mathbf{v}_{gato} = [0.5, -0.2]$
     - $\mathbf{v}_{senta} = [-0.3, 0.8]$
     - $\mathbf{v}_{no} = [0.1, -0.5]$
     - $\mathbf{v}_{tapete} = [0.7, 0.4]$

2. **Seleção da Palavra-Alvo e Contexto**

   - Para a palavra-alvo "senta", as palavras de contexto dentro da janela são "gato" e "no".

3. **Geração de Amostras Negativas**

   - Suponha que a palavra de ruído selecionada seja "tapete".

4. **Cálculo das Probabilidades**

   - **Probabilidade para a Palavra de Contexto Positiva ("gato")**:

     $$
     \sigma(\mathbf{v}_{gato} \cdot \mathbf{v}_{senta}) = \sigma([0.5, -0.2] \cdot [-0.3, 0.8]) = \sigma((0.5 \times -0.3) + (-0.2 \times 0.8)) = \sigma(-0.15 - 0.16) = \sigma(-0.31) \approx 0.423
     $$

   - **Probabilidade para a Palavra de Ruído ("tapete")**:

     $$
     \sigma(-\mathbf{v}_{tapete} \cdot \mathbf{v}_{senta}) = \sigma(-([0.7, 0.4] \cdot [-0.3, 0.8])) = \sigma(-(-0.21 + 0.32)) = \sigma(-0.11) \approx 0.473
     $$

5. **Cálculo da Função de Perda**

   - **Perda para a Palavra de Contexto Positiva**:

     $$
     L_{pos} = -\log \sigma(\mathbf{v}_{gato} \cdot \mathbf{v}_{senta}) = -\log(0.423) \approx 0.860
     $$

   - **Perda para a Palavra de Ruído**:

     $$
     L_{neg} = -\log \sigma(-\mathbf{v}_{tapete} \cdot \mathbf{v}_{senta}) = -\log(0.473) \approx 0.749
     $$

   - **Perda Total**:

     $$
     L = L_{pos} + L_{neg} \approx 0.860 + 0.749 = 1.609
     $$

6. **Atualização dos Vetores**

   - **Gradiente em Relação a $\mathbf{v}_{senta}$**:

     $$
     \Delta \mathbf{v}_{senta} = \eta \left[ (1 - \sigma(\mathbf{v}_{gato} \cdot \mathbf{v}_{senta})) \mathbf{v}_{gato} + (0 - \sigma(-\mathbf{v}_{tapete} \cdot \mathbf{v}_{senta})) \mathbf{v}_{tapete} \right]
     $$

     - Calculando os termos:

       - $(1 - 0.423) = 0.577$
       - $(0 - 0.473) = -0.473$

     - Atualização:

       $$
       \Delta \mathbf{v}_{senta} = \eta \left[ 0.577 \times [0.5, -0.2] - 0.473 \times [0.7, 0.4] \right]
       $$

   - **Supondo $\eta = 0.1$**:

     - Calculando:

       $$
       \Delta \mathbf{v}_{senta} = 0.1 \left[ [0.2885, -0.1154] - [0.3311, 0.1892] \right] = 0.1 \times [-0.0426, -0.3046] = [-0.00426, -0.03046]
       $$

   - **Atualizando $\mathbf{v}_{senta}$**:

     $$
     \mathbf{v}_{senta} \leftarrow \mathbf{v}_{senta} + \Delta \mathbf{v}_{senta} = [-0.3, 0.8] + [-0.00426, -0.03046] = [-0.30426, 0.76954]
     $$

   - **Atualizações similares são feitas para $\mathbf{v}_{gato}$ e $\mathbf{v}_{tapete}$**.

Este exemplo simplificado demonstra como o SGNS atualiza os vetores de embedding com base nas palavras de contexto positivas e nas amostras negativas.

## Propriedades Matemáticas do SGNS

### Distribuição de Amostragem Negativa

A probabilidade de selecionar uma palavra como amostra negativa é proporcional à sua frequência elevada a uma potência $\alpha$:

$$
P_{neg}(w) = \frac{f(w)^\alpha}{\sum_{w'} f(w')^\alpha}
$$

Onde:

- $f(w)$ é a frequência da palavra $w$ no corpus.
- $\alpha$ é tipicamente definido como 0.75.

Esta escolha de $\alpha$ equilibra a influência de palavras frequentes e raras, garantindo que as amostras negativas sejam informativas.

### Relação com a Matriz PMI

O SGNS pode ser visto como uma fatoração implícita de uma matriz de **Informação Mútua Pontual (PMI)** deslocada entre palavras e contextos. A relação é dada por:

$$
\mathbf{v}_w^\top \mathbf{v}_c = PMI(w, c) - \log K
$$

Onde:

- $PMI(w, c) = \log \frac{P(w, c)}{P(w)P(c)}$
- $K$ é o número de amostras negativas.

Esta relação explica por que o SGNS é eficaz em capturar associações semânticas entre palavras.

## Implementação Simplificada em Python

A seguir, uma implementação simplificada do SGNS usando NumPy:

```python
import numpy as np

# Dados de exemplo
corpus = ["gato", "senta", "no", "tapete"]
vocab = {word: i for i, word in enumerate(corpus)}
data = [(vocab["senta"], vocab["gato"]), (vocab["senta"], vocab["no"])]

# Parâmetros
embedding_dim = 2
eta = 0.1
K = 1  # Número de amostras negativas
alpha = 0.75

# Inicialização dos vetores
V = len(vocab)
W = np.random.randn(V, embedding_dim)
C = np.random.randn(V, embedding_dim)

# Frequências para amostragem negativa
word_counts = np.array([1 for _ in corpus])
neg_sampling_dist = word_counts ** alpha
neg_sampling_dist /= neg_sampling_dist.sum()

# Função sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Treinamento
for target, context in data:
    # Amostras negativas
    negatives = np.random.choice(V, size=K, p=neg_sampling_dist)
    
    # Vetores
    v_w = W[target]
    v_c = C[context]
    v_ns = C[negatives]
    
    # Cálculo dos scores
    score_pos = sigmoid(np.dot(v_w, v_c))
    score_negs = sigmoid(-np.dot(v_w, v_ns.T))
    
    # Cálculo dos gradientes
    grad_w = (1 - score_pos) * v_c - np.sum((score_negs[:, None] * v_ns), axis=0)
    grad_c = (1 - score_pos) * v_w
    grad_ns = -np.outer(score_negs, v_w)
    
    # Atualização dos vetores
    W[target] += eta * grad_w
    C[context] += eta * grad_c
    for i, neg in enumerate(negatives):
        C[neg] += eta * grad_ns[i]
```

Esta implementação demonstra o processo básico de treinamento do SGNS, incluindo a seleção de amostras negativas e a atualização dos vetores de embedding.

## Conclusão

O **Skip-gram com Amostragem Negativa** é uma técnica poderosa para a geração de *word embeddings*, permitindo capturar relações semânticas complexas entre palavras de forma eficiente. Compreender sua fundamentação teórica e seu funcionamento é essencial para aplicá-lo efetivamente em tarefas de PLN.

Este documento apresentou uma explicação detalhada do SGNS, incluindo uma demonstração passo a passo e um exemplo numérico intuitivo. Espera-se que este material auxilie na compreensão profunda do modelo e de suas aplicações práticas.

## Referências

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. *arXiv preprint arXiv:1301.3781*.
2. Goldberg, Y., & Levy, O. (2014). **Word2Vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method**. *arXiv preprint arXiv:1402.3722*.
3. Jurafsky, D., & Martin, J. H. (2019). **Speech and Language Processing** (3rd ed.). Draft chapters available at [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3).
4. Levy, O., & Goldberg, Y. (2014). **Neural Word Embedding as Implicit Matrix Factorization**. *Advances in Neural Information Processing Systems*, 27.