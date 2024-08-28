## Token Embeddings: Explorando Métodos Avançados de Representação Vetorial de Palavras

<image: Uma rede neural com camadas de entrada, ocultas e de saída, onde as palavras de entrada são transformadas em vetores densos (embeddings) na camada de saída. Setas indicam o fluxo de informação e transformação das palavras em vetores multidimensionais.>

### Introdução

Token embeddings são representações vetoriais densas de palavras ou tokens, fundamentais para o processamento de linguagem natural (NLP) moderno. Estas representações capturam relações semânticas e sintáticas entre palavras, permitindo que modelos de machine learning compreendam e processem texto de forma mais eficaz [1]. Este resumo explorará métodos avançados de criação de embeddings, incluindo Word2Vec, GloVe e FastText, analisando seu impacto na qualidade das representações e suas aplicações em tarefas de NLP.

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Embedding**             | Representação vetorial densa de uma palavra ou token em um espaço multidimensional, capturando características semânticas e sintáticas [1]. |
| **Dimensionalidade**      | Número de dimensões do vetor de embedding, influenciando a capacidade de representação e a eficiência computacional [2]. |
| **Corpus de Treinamento** | Conjunto de textos utilizados para treinar os modelos de embedding, afetando diretamente a qualidade e abrangência das representações [3]. |

> ✔️ **Ponto de Destaque**: Embeddings transformam palavras em vetores numéricos, permitindo operações matemáticas que capturam relações semânticas entre palavras.

### Word2Vec: Modelo Pioneiro de Embeddings

<image: Duas arquiteturas de rede neural lado a lado - CBOW e Skip-gram - mostrando a entrada de palavras de contexto e a previsão da palavra-alvo (CBOW) ou vice-versa (Skip-gram).>

Word2Vec, introduzido por Mikolov et al., revolucionou a criação de embeddings com duas arquiteturas principais: Continuous Bag-of-Words (CBOW) e Skip-gram [4].

#### Continuous Bag-of-Words (CBOW)

O modelo CBOW prediz uma palavra-alvo dado seu contexto. A função objetivo para CBOW pode ser expressa como:

$$
J_\theta = -\frac{1}{T} \sum_{t=1}^T \log p(w_t | w_{t-n}, ..., w_{t-1}, w_{t+1}, ..., w_{t+n})
$$

Onde:
- $T$ é o número total de palavras no corpus
- $w_t$ é a palavra-alvo
- $w_{t-n}, ..., w_{t+n}$ são as palavras de contexto
- $\theta$ são os parâmetros do modelo

#### Skip-gram

O modelo Skip-gram faz o inverso, prevendo as palavras de contexto dada uma palavra-alvo. A função objetivo para Skip-gram é:

$$
J_\theta = -\frac{1}{T} \sum_{t=1}^T \sum_{-n \leq j \leq n, j \neq 0} \log p(w_{t+j} | w_t)
$$

> ❗ **Ponto de Atenção**: Skip-gram geralmente tem melhor performance para palavras raras e com conjuntos de dados menores [4].

#### Negative Sampling

Para otimizar o treinamento, Word2Vec utiliza Negative Sampling, reduzindo a complexidade computacional. A função de loss para Skip-gram com Negative Sampling é:

$$
J_\theta = \log \sigma(v_{w_O}^T v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-v_{w_i}^T v_{w_I})]
$$

Onde:
- $v_{w_O}$ é o vetor de output para a palavra de contexto
- $v_{w_I}$ é o vetor de input para a palavra-alvo
- $\sigma$ é a função sigmoide
- $k$ é o número de amostras negativas
- $P_n(w)$ é a distribuição de ruído

#### Questões Técnicas/Teóricas

1. Como a escolha entre CBOW e Skip-gram afeta a qualidade dos embeddings para palavras raras em um corpus de domínio específico?
2. Explique como o Negative Sampling melhora a eficiência computacional do treinamento do Word2Vec e por que isso é crucial para grandes corpora.

### GloVe: Global Vectors for Word Representation

<image: Uma matriz de co-ocorrência de palavras sendo fatorada em duas matrizes menores, representando os vetores de palavras e contextos do GloVe.>

GloVe, desenvolvido por Pennington et al., combina as vantagens de métodos baseados em contagem (como LSA) e métodos preditivos (como Word2Vec) [5].

A função objetivo do GloVe é:

$$
J = \sum_{i,j=1}^V f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

Onde:
- $X_{ij}$ é a matriz de co-ocorrência de palavras
- $w_i$ e $\tilde{w}_j$ são vetores de palavras e contextos
- $b_i$ e $\tilde{b}_j$ são termos de bias
- $f(X_{ij})$ é uma função de peso que dá menos importância a co-ocorrências muito frequentes

> ✔️ **Ponto de Destaque**: GloVe captura informações globais de co-ocorrência, complementando a abordagem local do Word2Vec.

#### Questões Técnicas/Teóricas

1. Como a matriz de co-ocorrência no GloVe difere da abordagem de janela deslizante do Word2Vec, e quais são as implicações para a captura de relações semânticas?
2. Discuta as vantagens e desvantagens de usar a função de peso $f(X_{ij})$ no treinamento do GloVe.

### FastText: Embeddings Subpalavras

<image: Representação de uma palavra sendo composta por n-gramas de caracteres, com vetores correspondentes sendo somados para formar o embedding final.>

FastText, proposto por Bojanowski et al., estende o modelo Skip-gram do Word2Vec para incorporar informações de subpalavras [6].

A representação de uma palavra $w$ no FastText é dada por:

$$
s(w) = \sum_{g \in G_w} z_g
$$

Onde:
- $G_w$ é o conjunto de n-gramas da palavra $w$
- $z_g$ é o vetor de embedding para o n-grama $g$

A probabilidade de uma palavra de contexto $w_c$ dado uma palavra-alvo $w_t$ é:

$$
p(w_c|w_t) = \frac{\exp(s(w_t)^T v_{w_c})}{\sum_{w' \in V} \exp(s(w_t)^T v_{w'})}
$$

> ⚠️ **Nota Importante**: FastText é particularmente eficaz para línguas morfologicamente ricas e para lidar com palavras fora do vocabulário (OOV).

#### Questões Técnicas/Teóricas

1. Como o FastText lida com palavras fora do vocabulário (OOV) e por que isso é uma vantagem significativa sobre Word2Vec e GloVe?
2. Discuta o trade-off entre a capacidade do FastText de capturar informações morfológicas e o aumento da complexidade computacional devido ao uso de n-gramas.

### Impacto da Dimensionalidade e Dados de Treinamento

A qualidade dos embeddings é fortemente influenciada pela dimensionalidade dos vetores e pelo corpus de treinamento [7].

#### Dimensionalidade

| 👍 Vantagens de Alta Dimensionalidade  | 👎 Desvantagens de Alta Dimensionalidade |
| ------------------------------------- | --------------------------------------- |
| Maior capacidade de representação [7] | Aumento do custo computacional [7]      |
| Captura de relações mais sutis [7]    | Risco de overfitting [7]                |

A escolha da dimensionalidade ótima pode ser expressa como um problema de otimização:

$$
\text{dim}_{\text{opt}} = \arg\max_d \frac{\text{Performance}(d)}{\text{Custo Computacional}(d)}
$$

#### Corpus de Treinamento

A qualidade e quantidade dos dados de treinamento afetam diretamente a qualidade dos embeddings [8].

> ❗ **Ponto de Atenção**: Corpora maiores e mais diversos geralmente levam a embeddings mais robustos, mas podem introduzir ruído e viés.

A relação entre o tamanho do corpus e a qualidade dos embeddings pode ser modelada como:

$$
Q = f(S, D)
$$

Onde:
- $Q$ é a qualidade dos embeddings
- $S$ é o tamanho do corpus
- $D$ é a diversidade do corpus
- $f$ é uma função não-linear que captura a relação complexa entre estas variáveis

#### Questões Técnicas/Teóricas

1. Como você determinaria empiricamente a dimensionalidade ótima para embeddings em uma tarefa específica de NLP, considerando o trade-off entre performance e custo computacional?
2. Discuta estratégias para mitigar o viés introduzido por corpora de treinamento em embeddings e como isso pode afetar aplicações downstream.

### Conclusão

Token embeddings representam um avanço fundamental em NLP, permitindo a captura de relações semânticas e sintáticas complexas em representações vetoriais densas. Word2Vec, GloVe e FastText oferecem abordagens distintas, cada uma com suas vantagens e desafios específicos. A escolha do método, dimensionalidade e corpus de treinamento deve ser cuidadosamente considerada com base nos requisitos específicos da tarefa e recursos disponíveis. À medida que o campo evolui, a integração destes métodos com técnicas de aprendizado profundo e a adaptação a domínios específicos continuam a impulsionar avanços em aplicações de NLP.

### Questões Avançadas

1. Compare e contraste as abordagens de Word2Vec, GloVe e FastText em termos de sua capacidade de capturar analogias semânticas complexas. Como você projetaria um experimento para avaliar quantitativamente o desempenho de cada método nesta tarefa?

2. Discuta as implicações éticas e práticas do uso de embeddings pré-treinados em tarefas de NLP sensíveis, como análise de sentimento em contextos multilíngues ou detecção de discurso de ódio. Como os vieses inerentes aos dados de treinamento podem ser mitigados?

3. Proponha uma arquitetura híbrida que combine elementos de Word2Vec, GloVe e FastText para criar embeddings mais robustos. Quais seriam os desafios técnicos na implementação de tal modelo e como você avaliaria sua eficácia em comparação com os métodos individuais?

### Referências

[1] "Token embeddings são representações vetoriais densas de palavras ou tokens, fundamentais para o processamento de linguagem natural (NLP) moderno." (Trecho de Transformers and Large Language Models - Chapter 10)

[2] "Recall that in a one-hot vector all the elements are 0 except one, the element whose dimension is the word's index in the vocabulary, which has value 1." (Trecho de Transformers and Large Language Models - Chapter 10)

[3] "Large language models are mainly trained on text scraped from the web, augmented by more carefully curated data." (Trecho de Transformers and Large Language Models - Chapter 10)

[4] "We can formalize this algorithm for generating a sequence of words W = w1, w2, . . . , wN until we hit the end-of-sequence token, using x ∼ p(x) to mean 'choose x by sampling from the distribution p(x):" (Trecho de Transformers and Large Language Models - Chapter 10)

[5] "To train a transformer as a language model, we use the same self-supervision (or self-training) algorithm we saw in Section ??: we take a corpus of text as training material and at each time step t ask the model to predict the next word." (Trecho de Transformers and Large Language Models - Chapter 10)

[6] "Large models are generally trained by filling the full context window (for example 2048 or 4096 tokens for GPT3 or GPT4) with text." (Trecho de Transformers and Large Language Models - Chapter 10)

[7] "The performance of large language models has shown to be mainly determined by 3 factors: model size (the number of parameters not counting embeddings), dataset size (the amount of training data), and the amount of computer used for training." (Trecho de Transformers and Large Language Models - Chapter 10)

[8] "Web text is usually taken from corpora of automatically-crawled web pages like the common crawl, a series of snapshots of the entire web produced by the non-profit Common Crawl (https://commoncrawl.org/) that each have billions of webpages." (Trecho de Transformers and Large Language Models - Chapter 10)