# N-Gram Models: Aproximando Probabilidades de Palavras em Contexto

<imagem: Um diagrama mostrando uma sequência de palavras com janelas deslizantes representando bigrams, trigrams e n-grams de ordem superior, destacando como eles capturam diferentes níveis de contexto.>

## Introdução

Os **modelos n-gram** são uma abordagem fundamental em processamento de linguagem natural (NLP) para modelar sequências de palavras. Eles se baseiam na ideia de que podemos aproximar a probabilidade de uma palavra dado seu contexto usando apenas as n-1 palavras anteriores [1]. Este conceito, originado dos trabalhos de Markov no início do século XX, tornou-se uma pedra angular no desenvolvimento de modelos de linguagem estatísticos [2].

N-gram models são particularmente importantes porque oferecem uma maneira simples e eficaz de capturar dependências locais em texto, permitindo:

1. Atribuir probabilidades a sentenças
2. Prever a próxima palavra dado um contexto
3. Gerar texto de forma probabilística

Neste resumo, exploraremos em profundidade os conceitos por trás dos modelos n-gram, focando principalmente em bigrams (2 palavras) e trigrams (3 palavras), mas também discutindo n-grams de ordem superior [3].

## Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **N-gram**                    | Um n-gram é uma sequência contígua de n itens (geralmente palavras) de um dado texto ou corpus. Por exemplo, um bigram (n=2) é uma sequência de duas palavras, como "quero comer" [4]. |
| **Probabilidade Condicional** | A base dos modelos n-gram é a probabilidade condicional, que estima a chance de uma palavra ocorrer dado seu contexto anterior. Para um bigram, isso é representado como P(w_n\|w_{n-1}) [5]. |
| **Cadeia de Markov**          | Os n-grams são essencialmente modelos de Markov de ordem n-1, onde a probabilidade de um estado (palavra) depende apenas dos n-1 estados anteriores [6]. |

> ⚠️ **Nota Importante**: A suposição fundamental dos modelos n-gram é que a probabilidade de uma palavra depende apenas das n-1 palavras anteriores. Esta é uma simplificação que, embora útil, não captura dependências de longo alcance na linguagem [7].

### Formulação Matemática

A probabilidade de uma sequência de palavras W = w_1, w_2, ..., w_n pode ser decomposta usando a regra da cadeia de probabilidade:

$$
P(W) = P(w_1, w_2, ..., w_n) = \prod_{k=1}^n P(w_k|w_{1:k-1})
$$

Para um modelo n-gram, fazemos a aproximação:

$$
P(w_k|w_{1:k-1}) \approx P(w_k|w_{k-N+1:k-1})
$$

onde N é a ordem do n-gram [8].

## Tipos de N-Grams

### Bigrams

<imagem: Uma visualização de uma janela deslizante de tamanho 2 sobre uma sequência de palavras, destacando como os bigrams são formados.>

Bigrams são o tipo mais simples de n-gram após unigrams. Eles consideram pares de palavras adjacentes para estimar probabilidades [9].

A probabilidade de um bigram é calculada como:

$$
P(w_n|w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}
$$

onde C(x) representa a contagem de ocorrências de x no corpus de treinamento [10].

### Trigrams

Trigrams estendem o contexto para três palavras, oferecendo uma captura mais rica de dependências locais [11].

A probabilidade de um trigram é:

$$
P(w_n|w_{n-2}, w_{n-1}) = \frac{C(w_{n-2}, w_{n-1}, w_n)}{C(w_{n-2}, w_{n-1})}
$$

### N-Grams de Ordem Superior

Modelos de ordem superior (n > 3) podem capturar contextos mais amplos, mas sofrem com o problema de esparsidade de dados e aumento exponencial de parâmetros [12].

> ❗ **Ponto de Atenção**: À medida que n aumenta, a precisão do modelo pode melhorar, mas o custo computacional e a necessidade de dados de treinamento crescem significativamente [13].

#### Perguntas Teóricas

1. Derive a fórmula geral para a probabilidade de um n-gram de ordem k, generalizando as equações para bigrams e trigrams apresentadas.

2. Considerando um corpus de tamanho N e um vocabulário de tamanho V, demonstre matematicamente como o número de parâmetros cresce em função de n para um modelo n-gram.

3. Prove que, para qualquer sequência de palavras, a probabilidade atribuída por um modelo n-gram de ordem k é sempre maior ou igual à probabilidade atribuída por um modelo de ordem k+1. Explique as implicações teóricas deste resultado.

## Estimação de Probabilidades

A estimação de probabilidades em modelos n-gram é tipicamente baseada em contagens de ocorrências no corpus de treinamento. ==O método mais simples é a Estimativa de Máxima Verossimilhança (MLE) [14].==

Para um n-gram genérico:

$$
P_{MLE}(w_n|w_{n-N+1:n-1}) = \frac{C(w_{n-N+1:n})}{C(w_{n-N+1:n-1})}
$$

Onde:
- $C(w_{n-N+1:n})$ é a contagem do n-gram completo
- $C(w_{n-N+1:n-1})$ é a contagem do contexto (n-1 palavras anteriores)

> ✔️ **Destaque**: A MLE fornece estimativas não-viesadas, mas sofre com o problema de probabilidade zero para n-grams não observados no corpus de treinamento [15].

### Suavização (Smoothing)

Para lidar com o problema de n-grams não observados e melhorar as estimativas, várias técnicas de suavização foram desenvolvidas [16]:

1. **Suavização de Laplace (Add-One)**:
   Adiciona 1 a todas as contagens antes da normalização.

   $$P_{Laplace}(w_n|w_{n-1}) = \frac{C(w_{n-1}, w_n) + 1}{C(w_{n-1}) + V}$$

   Onde V é o tamanho do vocabulário.

2. **Interpolação Linear**:
   Combina probabilidades de n-grams de diferentes ordens.

   $$\hat{P}(w_n|w_{n-2}, w_{n-1}) = \lambda_1P(w_n) + \lambda_2P(w_n|w_{n-1}) + \lambda_3P(w_n|w_{n-2}, w_{n-1})$$

   Onde $\sum_i \lambda_i = 1$.

3. **Backoff**:
   Usa n-grams de ordem inferior quando não há dados suficientes para n-grams de ordem superior.

> ⚠️ **Nota Importante**: A escolha do método de suavização pode ter um impacto significativo no desempenho do modelo, especialmente para n-grams de ordem superior e vocabulários grandes [17].

#### Perguntas Teóricas

1. Demonstre matematicamente por que a suavização de Laplace (Add-One) tende a superestimar as probabilidades de eventos raros em comparação com eventos frequentes.

2. Derive a fórmula para o erro quadrático médio (MSE) das estimativas de probabilidade usando interpolação linear em função dos coeficientes $\lambda_i$. Como você otimizaria esses coeficientes?

3. Proponha e analise teoricamente uma nova técnica de suavização que combine aspectos de backoff e interpolação, discutindo suas vantagens e desvantagens potenciais.

## Avaliação de Modelos N-Gram

A avaliação de modelos n-gram é crucial para entender seu desempenho e compará-los. A métrica principal usada é a **perplexidade** [18].

### Perplexidade

A perplexidade é definida como a inversa da probabilidade média por palavra, normalizada pelo número de palavras:

$$
\text{Perplexidade}(W) = P(w_1w_2...w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1w_2...w_N)}}
$$

Onde W é a sequência de palavras do conjunto de teste [19].

> ✔️ **Destaque**: Uma perplexidade menor indica um modelo melhor, pois significa que o modelo está menos "surpreso" pelo texto de teste [20].

A perplexidade está intimamente relacionada com a entropia cruzada:

$$
\text{Perplexidade}(W) = 2^{H(W)}
$$

Onde H(W) é a entropia cruzada [21].

### Relação com Teoria da Informação

A perplexidade pode ser interpretada como o fator de ramificação médio ponderado de uma linguagem. Para uma linguagem com vocabulário V e distribuição uniforme, a perplexidade seria V. Na prática, devido à não uniformidade da distribuição de palavras, a perplexidade é geralmente muito menor que V [22].

## Aplicações e Limitações

### Aplicações

1. **Correção ortográfica e gramatical**: N-grams podem identificar sequências improváveis de palavras [23].
2. **Reconhecimento de fala**: Ajudam a distinguir entre palavras foneticamente similares com base no contexto [24].
3. **Tradução automática**: Usados para avaliar a fluência das traduções geradas [25].
4. **Geração de texto**: Podem gerar texto sintetizado, embora com limitações [26].

### Limitações

1. **Dependências de longo alcance**: N-grams capturam apenas contextos locais, perdendo dependências distantes [27].
2. **Esparsidade de dados**: Para n grande, muitos n-grams possíveis nunca são observados no treinamento [28].
3. **Falta de semântica**: Baseiam-se apenas em estatísticas de co-ocorrência, sem compreensão do significado [29].

> ❗ **Ponto de Atenção**: Modelos n-gram são uma baseline importante em NLP, mas foram largamente superados por modelos neurais em muitas tarefas modernas [30].

## Implementação Avançada

Aqui está um exemplo avançado de implementação de um modelo n-gram usando Python e NumPy, incorporando suavização e cálculo de perplexidade:

```python
import numpy as np
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n, smoothing='laplace', alpha=1):
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        
    def fit(self, corpus):
        for sentence in corpus:
            padded = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            self.vocab.update(padded)
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i+self.n])
                self.ngram_counts[ngram[:-1]][ngram[-1]] += 1
                self.context_counts[ngram[:-1]] += 1
        
    def _smooth_probability(self, word, context):
        if self.smoothing == 'laplace':
            return (self.ngram_counts[context][word] + self.alpha) / \
                   (self.context_counts[context] + self.alpha * len(self.vocab))
        # Implementar outros métodos de suavização aqui
        
    def probability(self, word, context):
        return self._smooth_probability(word, tuple(context))
    
    def perplexity(self, test_corpus):
        log_prob_sum = 0
        token_count = 0
        for sentence in test_corpus:
            padded = ['<s>'] * (self.n - 1) + sentence + ['</s>']
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i-self.n+1:i])
                word = padded[i]
                log_prob_sum += np.log2(self.probability(word, context))
                token_count += 1
        return 2 ** (-log_prob_sum / token_count)

# Uso
corpus = [
    ['o', 'gato', 'come', 'peixe'],
    ['o', 'cão', 'come', 'carne'],
    ['gatos', 'gostam', 'de', 'peixe']
]

model = NGramModel(n=3, smoothing='laplace', alpha=0.1)
model.fit(corpus)

test_sentence = ['o', 'gato', 'come', 'carne']
perplexity = model.perplexity([test_sentence])
print(f"Perplexidade: {perplexity}")
```

Este código implementa um modelo n-gram com suavização de Laplace e cálculo de perplexidade, demonstrando conceitos avançados discutidos anteriormente [31].

## Conclusão

Os modelos n-gram representam uma abordagem fundamental e historicamente significativa para modelagem de linguagem estatística. Embora simples em conceito, eles capturam eficientemente dependências locais em texto e fornecem uma base sólida para entender desafios mais complexos em NLP [32].

Apesar de suas limitações, principalmente em capturar dependências de longo alcance e semântica profunda, os n-grams continuam sendo relevantes como baselines, em aplicações específicas e como componentes de sistemas mais complexos [33].

À medida que o campo de NLP avança, é crucial entender os fundamentos fornecidos pelos modelos n-gram, pois muitos conceitos, como suavização e avaliação baseada em perplexidade, permanecem relevantes mesmo em arquiteturas mais avançadas como modelos de linguagem neurais [34].

## Perguntas Teóricas Avançadas

1. Derive a relação matemática entre a perplexidade de um modelo n-gram e a entropia da distribuição verdadeira da linguagem. Sob quais condições a perplexidade do modelo se iguala à verdadeira entropia da linguagem?

2. Considerando um modelo n-gram com suavização de interpolação, prove que existe um conjunto ótimo de pesos de interpolação que minimiza a perplexidade no conjunto de validação. Como você formularia um algoritmo para encontrar esses pesos ótimos?

3. Desenvolva uma prova formal mostrando que, para qualquer distribuição de linguagem estacionária e ergódica, a estimativa de máxima verossimilhança de um modelo n-gram converge para a verdadeira distribuição à medida que o tamanho do corpus de treinamento tende ao infinito.

4. Analise teoricamente o impacto da ordem do n-gram na capacidade do modelo de capturar diferentes fenômenos linguísticos (por exemplo, concordância sujeito-verbo, dependências de longa distância). Quais