# Skip-gram with Negative Sampling (SGNS): Fundamentos e Aplicações Avançadas em Word Embeddings

<imagem: Uma rede neural representando o modelo Skip-gram, com uma camada de entrada para a palavra-alvo, uma camada oculta, e múltiplas saídas para palavras de contexto e ruído, ilustrando o processo de amostragem negativa.>

## Introdução

O Skip-gram with Negative Sampling (SGNS) é um algoritmo fundamental no campo do processamento de linguagem natural (NLP) e representa uma evolução significativa na criação de word embeddings. Este método, parte central do modelo Word2Vec, revolucionou a forma como representamos palavras em espaços vetoriais densos, permitindo capturar relações semânticas complexas de maneira eficiente [1]. 

O SGNS baseia-se na ideia de que palavras que ocorrem em contextos similares tendem a ter significados semelhantes, um princípio conhecido como hipótese distribucional. Esta abordagem não apenas superou métodos anteriores em termos de eficiência computacional, mas também demonstrou uma notável capacidade de capturar nuances semânticas e sintáticas da linguagem [2].

## Conceitos Fundamentais

| Conceito              | Explicação                                                   |
| --------------------- | ------------------------------------------------------------ |
| **Word Embedding**    | Representação vetorial densa de palavras em um espaço multidimensional. No contexto do SGNS, estas são aprendidas através de um processo de classificação binária [3]. |
| **Skip-gram**         | Modelo que prevê palavras de contexto dada uma palavra-alvo, invertendo a abordagem tradicional dos modelos de linguagem [4]. |
| **Negative Sampling** | Técnica de otimização que seleciona aleatoriamente "palavras de ruído" para contrastar com as palavras de contexto reais, melhorando a eficiência do treinamento [5]. |

> ⚠️ **Nota Importante**: O SGNS difere de modelos anteriores por não requerer a normalização sobre todo o vocabulário, tornando o treinamento significativamente mais eficiente para vocabulários grandes [6].

> ❗ **Ponto de Atenção**: A qualidade dos embeddings gerados pelo SGNS é altamente dependente da escolha dos hiperparâmetros, especialmente o tamanho da janela de contexto e o número de amostras negativas [7].

> ✔️ **Destaque**: O SGNS demonstrou capacidade de capturar relações analógicas complexas entre palavras, como "rei - homem + mulher ≈ rainha", um marco na representação semântica vetorial [8].

### Fundamentos Teóricos do SGNS

<imagem: Diagrama ilustrando a arquitetura do modelo Skip-gram, destacando a entrada (palavra-alvo), a camada oculta (embedding), e as múltiplas saídas (palavras de contexto e ruído), com setas indicando o fluxo de informação e o processo de otimização.>

O SGNS fundamenta-se na maximização da probabilidade de palavras de contexto verdadeiras enquanto minimiza a probabilidade de palavras de ruído. Matematicamente, para uma palavra-alvo $w$ e uma palavra de contexto $c$, o modelo define:

$$
P(+|w,c) = \sigma(c \cdot w) = \frac{1}{1 + e^{-c \cdot w}}
$$

Onde $\sigma$ é a função sigmoide, $c$ e $w$ são vetores de embedding para as palavras de contexto e alvo, respectivamente [9].

A função objetivo do SGNS para um par $(w,c)$ e $k$ palavras de ruído $c_{neg}$ é:

$$
L = -\log\sigma(c \cdot w) - \sum_{i=1}^{k} \log\sigma(-c_{neg_i} \cdot w)
$$

Esta função é otimizada usando descida de gradiente estocástica (SGD) [10].

#### Processo de Treinamento

1. Para cada palavra-alvo $w$, selecione palavras de contexto $c$ dentro de uma janela.
2. Para cada par $(w,c)$, gere $k$ palavras de ruído $c_{neg}$.
3. Atualize os embeddings para maximizar $\sigma(c \cdot w)$ e minimizar $\sigma(c_{neg} \cdot w)$.

A atualização dos vetores segue as equações:

$$
c_{pos}^{t+1} = c_{pos}^t - \eta[\sigma(c_{pos}^t \cdot w^t) - 1]w^t
$$

$$
c_{neg}^{t+1} = c_{neg}^t - \eta[\sigma(c_{neg}^t \cdot w^t)]w^t
$$

$$
w^{t+1} = w^t - \eta([\sigma(c_{pos}^t \cdot w^t) - 1]c_{pos}^t + \sum_{i=1}^k [\sigma(c_{neg_i}^t \cdot w^t)]c_{neg_i}^t)
$$

Onde $\eta$ é a taxa de aprendizado [11].

#### Perguntas Teóricas

1. Derive a expressão para o gradiente da função de perda $L$ em relação ao vetor de embedding $w$ para uma única palavra de contexto e uma palavra de ruído.

2. Demonstre matematicamente por que o SGNS é computacionalmente mais eficiente que modelos que requerem normalização sobre todo o vocabulário.

3. Analise teoricamente como o tamanho da janela de contexto afeta a captura de relações semânticas vs. sintáticas nos embeddings resultantes.

### Amostragem Negativa: Teoria e Implementação

A amostragem negativa é crucial para a eficiência do SGNS. Palavras de ruído são amostradas de acordo com uma distribuição de probabilidade modificada:

$$
P_\alpha(w) = \frac{\sum_{w'} \text{count}(w')^\alpha \text{count}(w)^\alpha}{\sum_{w'} \text{count}(w')^\alpha}
$$

Onde $\alpha = 0.75$ é comumente usado para dar maior probabilidade a palavras raras [12].

> 💡 **Insight**: A escolha de $\alpha = 0.75$ equilibra a frequência das palavras, permitindo que palavras menos comuns tenham maior chance de serem selecionadas como amostras negativas, o que melhora a qualidade dos embeddings [13].

#### Implementação em Python

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, target_word, context_word, negative_samples):
        target_emb = self.target_embeddings(target_word)
        context_emb = self.context_embeddings(context_word)
        neg_embs = self.context_embeddings(negative_samples)
        
        pos_score = torch.sum(target_emb * context_emb, dim=1)
        neg_scores = torch.sum(target_emb.unsqueeze(1) * neg_embs, dim=2)
        
        pos_loss = -F.logsigmoid(pos_score)
        neg_loss = -torch.sum(F.logsigmoid(-neg_scores), dim=1)
        
        return pos_loss + neg_loss

# Implementação da amostragem negativa
def negative_sampling(word_freqs, alpha=0.75, k=5):
    adjusted_freqs = word_freqs ** alpha
    probs = adjusted_freqs / np.sum(adjusted_freqs)
    return np.random.choice(len(word_freqs), size=k, p=probs)
```

Este código implementa o modelo SGNS usando PyTorch, incluindo a lógica de amostragem negativa [14].

#### Perguntas Teóricas

1. Derive a expressão para a complexidade computacional do SGNS em função do tamanho do vocabulário e compare com a complexidade de um modelo softmax completo.

2. Analise matematicamente como diferentes valores de $\alpha$ na distribuição de amostragem negativa afetam a convergência do modelo e a qualidade dos embeddings resultantes.

3. Proponha e justifique teoricamente uma modificação na função de perda que poderia melhorar a captura de relações semânticas específicas (por exemplo, hiperonímia).

### Análise Avançada de Propriedades dos Embeddings SGNS

Os embeddings gerados pelo SGNS exibem propriedades matemáticas interessantes que refletem relações semânticas e sintáticas entre palavras. Uma propriedade notável é a capacidade de resolver analogias através de operações vetoriais [15].

Para uma analogia $a:b::c:d$, onde $d$ é desconhecido, podemos aproximar:

$$
\vec{d} \approx \vec{c} - \vec{a} + \vec{b}
$$

Esta propriedade emerge da estrutura linear capturada pelo espaço de embeddings [16].

> ⚠️ **Nota Importante**: Apesar de sua eficácia em muitos casos, o método do paralelogramo para analogias tem limitações, especialmente para relações semânticas complexas ou palavras raras [17].

#### Análise de Componentes Principais (PCA) dos Embeddings

A aplicação de PCA aos embeddings SGNS revela estruturas semânticas latentes:

1. Os primeiros componentes principais frequentemente capturam dimensões semânticas gerais (por exemplo, polaridade, formalidade).
2. Componentes subsequentes podem representar campos semânticos específicos ou relações gramaticais [18].

#### Visualização t-SNE

A técnica t-SNE (t-Distributed Stochastic Neighbor Embedding) é frequentemente usada para visualizar embeddings SGNS em duas ou três dimensões, revelando clusters semânticos [19].

<imagem: Gráfico de dispersão 2D mostrando clusters de palavras após aplicação de t-SNE em embeddings SGNS, com diferentes cores representando campos semânticos distintos.>

#### Perguntas Teóricas

1. Derive matematicamente a relação entre a similaridade do cosseno de dois vetores de embedding e a probabilidade condicional usada no treinamento do SGNS.

2. Proponha e justifique um método teórico para identificar e quantificar vieses (por exemplo, de gênero ou raça) nos embeddings SGNS usando análise de componentes principais.

3. Desenvolva uma prova formal demonstrando por que o método do paralelogramo para analogias funciona no espaço de embeddings SGNS, e identifique condições sob as quais ele pode falhar.

### Extensões e Variantes do SGNS

O sucesso do SGNS inspirou várias extensões e variantes:

1. **FastText**: Incorpora informações de subpalavras, melhorando representações para línguas morfologicamente ricas e lidando com palavras fora do vocabulário [20].

2. **GloVe**: Combina as vantagens de factorização de matrizes globais com a aprendizagem de contexto local do SGNS [21].

3. **CBOW (Continuous Bag of Words)**: Uma variante do Word2Vec que prediz a palavra-alvo dado o contexto, em vez do contrário [22].

#### Comparação de Modelos

| Modelo   | Vantagens                                                    | Desvantagens                               |
| -------- | ------------------------------------------------------------ | ------------------------------------------ |
| SGNS     | Eficiente, bom para analogias                                | Limitado a palavras no vocabulário         |
| FastText | Lida com palavras OOV, bom para línguas morfologicamente ricas | Maior complexidade computacional           |
| GloVe    | Captura estatísticas globais e locais                        | Requer mais dados para treinamento efetivo |
| CBOW     | Mais rápido que SGNS para treinar                            | Geralmente inferior em tarefas semânticas  |

### Conclusão

O Skip-gram with Negative Sampling representa um marco fundamental no desenvolvimento de técnicas de word embedding. Sua eficiência computacional, combinada com a capacidade de capturar relações semânticas e sintáticas complexas, tornou-o uma base para inúmeras aplicações em NLP [23].

A compreensão profunda do SGNS não apenas fornece insights sobre como as máquinas podem aprender representações significativas de linguagem, mas também abre caminhos para o desenvolvimento de modelos ainda mais sofisticados. À medida que o campo do NLP continua a evoluir, os princípios fundamentais estabelecidos pelo SGNS permanecem cruciais para o avanço de tecnologias de processamento de linguagem [24].

### Perguntas Teóricas Avançadas

1. Desenvolva uma prova matemática formal demonstrando que, sob certas condições, o SGNS implicitamente fatoriza uma matriz de Informação Mútua Pontual (PMI) entre palavras e contextos.

2. Proponha e justifique teoricamente uma extensão do SGNS que incorpore informações de grafos de conhecimento externos durante o treinamento. Como isso afetaria a função objetivo e o processo de otimização?

3. Analise matematicamente o impacto da dimensionalidade dos embeddings na capacidade do modelo de capturar relações semânticas. Existe um "ponto ótimo" teórico para a dimensionalidade, e como ele se relaciona com o tamanho do vocabulário e a complexidade do corpus?

4. Derive uma expressão para a complexidade amostral do SGNS em termos de propriedades estatísticas do corpus de treinamento (por exemplo, distribuição de frequência de palavras, entropia do texto).

5. Proponha um framework teórico para quantificar a "interpretabilidade" dos embeddings SGNS. Como isso se compara com outras técnicas de word embedding em termos de trade-off entre interpretabilidade e performance em tarefas downstream?

### Referências

[1] "O Skip-gram with Negative Sampling (SGNS) é um algoritmo fundamental no campo do processamento de linguagem natural (NLP) e representa uma evolução significativa na criação de word embeddings." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[2] "O SGNS baseia-se na ideia de que palavras que ocorrem em contextos similares tendem a ter significados semelhantes, um princípio conhecido como hipótese distribucional." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[3] "No contexto do SGNS, estas são aprendidas através de um processo de classificação binária" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[4] "Modelo que prevê palavras de contexto dada uma palavra-alvo, invertendo a abordagem tradicional dos modelos de linguagem" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[5] "Técnica de otimização que seleciona aleatoriamente "palavras de ruído" para contrastar com as palavras de contexto reais, melhorando a eficiência do treinamento" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[6] "O SGNS difere de modelos anteriores por não requerer a normalização sobre todo o vocabulário, tornando o treinamento significativamente mais eficiente para vocabulários grandes" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[7] "A qualidade dos embeddings gerados pelo SGNS é altamente dependente da escolha dos hiperparâmetros,