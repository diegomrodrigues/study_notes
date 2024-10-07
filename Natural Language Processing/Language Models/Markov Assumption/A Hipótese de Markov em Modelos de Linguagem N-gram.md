# A Hipótese de Markov em Modelos de Linguagem N-gram

<imagem: Uma representação visual de uma cadeia de Markov de ordem $n$, mostrando como a probabilidade de uma palavra depende apenas das $n$ palavras anteriores, com setas indicando as dependências limitadas.>

## Introdução

A **Hipótese de Markov** é um conceito fundamental na modelagem estatística de linguagem, especialmente em modelos n-gram. Esta hipótese simplifica significativamente o cálculo de probabilidades em sequências de palavras, permitindo aproximações eficientes e computacionalmente viáveis [1]. Neste resumo, exploraremos em profundidade a Hipótese de Markov, sua aplicação em modelos n-gram e suas implicações no processamento de linguagem natural.

==A essência da Hipótese de Markov em modelos de linguagem é a suposição de que a probabilidade de ocorrência de uma palavra depende apenas de um número limitado de palavras anteriores==, ao invés de toda a história precedente. Isso possibilita a simplificação de modelos complexos de linguagem em cálculos mais manejáveis, mantendo um nível aceitável de precisão [2].

> ⚠️ **Nota Importante**: Embora poderosa, a Hipótese de Markov implica em limitações na capacidade do modelo de capturar dependências de longo alcance na linguagem [3].

## Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Hipótese de Markov**        | Supõe que a probabilidade de uma palavra depende apenas de um número fixo de palavras anteriores, não de toda a sequência anterior [4]. |
| **N-gram**                    | Modelo de linguagem baseado na Hipótese de Markov, onde $n$ representa o número de palavras consideradas no contexto [5]. |
| **Probabilidade Condicional** | Base matemática para calcular a probabilidade de uma palavra dado seu contexto em modelos n-gram [6]. |

### Formulação Matemática da Hipótese de Markov

A Hipótese de Markov pode ser expressa matematicamente como [7]:

$$
P(w_n \mid w_1^{n-1}) \approx P(w_n \mid w_{n-N+1}^{n-1})
$$

Onde:

- $w_n$ é a palavra atual.
- $w_1^{n-1}$ representa todas as palavras anteriores.
- $w_{n-N+1}^{n-1}$ representa as $N-1$ palavras anteriores em um modelo n-gram.

Esta aproximação é o núcleo da simplificação que permite a implementação eficiente de modelos n-gram [8].

> 💡 **Insight**: A Hipótese de Markov reduz drasticamente o espaço de parâmetros a serem estimados, tornando o treinamento de modelos de linguagem computacionalmente viável [9].

### Implicações da Hipótese de Markov

1. **Redução de Complexidade**: Ao limitar o contexto considerado, reduz-se exponencialmente o número de probabilidades que precisam ser estimadas [10]. Em vez de considerar todas as possíveis sequências de palavras anteriores, consideram-se apenas as últimas $N-1$ palavras.
   
2. **Perda de Informação**: Ignora dependências de longo alcance, o que pode levar à perda de informações contextuais importantes [11]. Isso afeta a capacidade do modelo de capturar nuances e relações semânticas distantes na linguagem.

3. **Eficiência Computacional**: Permite cálculos rápidos e eficientes de probabilidades de sequências, viabilizando a utilização prática em aplicações de processamento de linguagem natural [12].

## Modelos N-gram e a Hipótese de Markov

Os modelos n-gram são uma aplicação direta da Hipótese de Markov na modelagem de linguagem [13]. Eles aproximam a probabilidade de uma palavra com base nas $N-1$ palavras anteriores:

1. **Unigram ($n=1$)**: Assume independência completa entre palavras.

   $$
   P(w_n) = \frac{\text{count}(w_n)}{N}
   $$

2. **Bigram ($n=2$)**: Considera a palavra imediatamente anterior.

   $$
   P(w_n \mid w_{n-1}) = \frac{\text{count}(w_{n-1}, w_n)}{\text{count}(w_{n-1})}
   $$

3. **Trigram ($n=3$)**: Considera as duas palavras anteriores.

   $$
   P(w_n \mid w_{n-2}, w_{n-1}) = \frac{\text{count}(w_{n-2}, w_{n-1}, w_n)}{\text{count}(w_{n-2}, w_{n-1})}
   $$

> ✔️ **Destaque**: A escolha do valor de $n$ representa um trade-off entre a capacidade de capturar contexto e a esparsidade dos dados [14]. Valores maiores de $n$ capturam mais contexto, mas aumentam a esparsidade.

### Exemplo Prático

Considere a sequência "I am Sam":

1. **Probabilidade Unigram**:

   $$
   P(\text{I}) \times P(\text{am}) \times P(\text{Sam})
   $$

2. **Probabilidade Bigram**:

   $$
   P(\text{I} \mid \langle s \rangle) \times P(\text{am} \mid \text{I}) \times P(\text{Sam} \mid \text{am})
   $$

3. **Probabilidade Trigram**:

   $$
   P(\text{I} \mid \langle s \rangle \langle s \rangle) \times P(\text{am} \mid \langle s \rangle \text{I}) \times P(\text{Sam} \mid \text{I am})
   $$

==Onde $\langle s \rangle$ representa o início da sentença [15].==

### 1. **Perplexidade e Entropia Cruzada**

A **perplexidade** é uma medida que avalia o quão bem um modelo de linguagem prevê uma sequência de palavras. Ela está diretamente relacionada à **entropia cruzada** do modelo.

**Derivação da Perplexidade:**

A entropia cruzada $H(p)$ de um modelo de linguagem é definida como:

$$
H(p) = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_{i-n+1}^{i-1})
$$

Onde:

- $N$ é o número total de palavras na sequência de teste.
- $P(w_i \mid w_{i-n+1}^{i-1})$ é a probabilidade predita pelo modelo para a palavra $w_i$ dado o contexto das $n-1$ palavras anteriores.

A perplexidade é então definida como:

$$
\text{Perplexidade} = 2^{H(p)} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_{i-n+1}^{i-1})}
$$

Simplificando, temos:

$$
\text{Perplexidade} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i \mid w_{i-n+1}^{i-1})\right)
$$

(Usando a base natural de logaritmos, onde $\ln 2$ é a constante de conversão entre logaritmos de base 2 e logaritmos naturais.)

**Relação entre Perplexidade e Entropia Cruzada:**

- ==A entropia cruzada $H(p)$ mede a média da incerteza ou surpresa associada às previsões do modelo em relação à distribuição real dos dados.==
- A perplexidade é a ==exponenciação da entropia cruzada==, servindo como uma medida mais intuitiva: ==representa o número efetivo de escolhas possíveis que o modelo considera ao prever a próxima palavra.==
- ==Uma perplexidade menor indica que o modelo é mais certo em suas previsões==, ou seja, atribui probabilidades maiores às palavras corretas.

**Interpretação:**

- **Perplexidade como Métrica de Desempenho**: Avalia a capacidade do modelo de prever sequências de teste não vistas. Um modelo perfeito teria perplexidade igual a 1.
- **Entropia Cruzada como Base Teórica**: Fornece a fundamentação matemática para o cálculo da perplexidade, relacionando a distribuição predita pelo modelo com a distribuição verdadeira dos dados.

### 2. **Limitações da Hipótese de Markov**

A **Hipótese de Markov** assume que a probabilidade de uma palavra depende apenas das $n-1$ palavras anteriores, isto é:

$$
P(w_n \mid w_1^{n-1}) \approx P(w_n \mid w_{n-N+1}^{n-1})
$$

Onde $N$ é a ordem do modelo n-gram.

**Demonstração das Limitações:**

- **Dependências de Longo Alcance**: Em linguagens naturais, há dependências que podem abranger mais do que $n-1$ palavras. Por exemplo, em frases complexas, o sujeito e o verbo podem estar distantes, e elementos como pronomes podem referir-se a substantivos mencionados muito antes.
  
- **Propriedades de Cadeias de Markov**: ==Uma cadeia de Markov de ordem $N-1$ não captura transições de estados (palavras) que estão além dos $N-1$ estados anteriores.== Assim, informações relevantes fora deste contexto imediato são ignoradas.

**Matematicamente:**

- A probabilidade real deveria considerar toda a história:

  $$
  P(w_n \mid w_1^{n-1}) = P(w_n \mid w_{n-N+1}^{n-1}, w_1^{n-N})
  $$

- A Hipótese de Markov simplifica para:

  $$
  P(w_n \mid w_1^{n-1}) \approx P(w_n \mid w_{n-N+1}^{n-1})
  $$

- **Consequência**: Se houver uma forte dependência entre $w_n$ e alguma palavra em $w_1^{n-N}$, o modelo n-gram não capturará essa dependência, levando a uma estimativa imprecisa de $P(w_n \mid w_1^{n-1})$.

**Exemplo Prático:**

- Considere a frase: "Se chover amanhã, o jogo será cancelado, então traga um guarda-chuva."
- A palavra "guarda-chuva" está relacionada à possibilidade de "chover", mencionada no início. Um modelo n-gram com $n$ pequeno não capturará essa relação, subestimando a probabilidade de "guarda-chuva" neste contexto.

**Conclusão:**

- A Hipótese de Markov, ao limitar o contexto, não consegue modelar dependências de longo alcance.
- Isso resulta em estimativas de probabilidade imprecisas para sequências onde tais dependências são significativas.
- Portanto, a precisão do modelo é comprometida em contextos onde a linguagem natural exige considerações além das $n-1$ palavras anteriores.

### 3. **Esparsidade de Dados**

**Análise Teórica do Impacto do Aumento de $n$:**

- **Crescimento Combinatorial**: O número total de possíveis n-grams é proporcional a $V^n$, onde $V$ é o tamanho do vocabulário.
  
- **Esparsidade**: À medida que $n$ aumenta, o espaço de possíveis n-grams cresce exponencialmente, mas o tamanho do corpus de treinamento permanece fixo. Isso leva a:

  - **Muitos n-grams Inobservados**: A maioria dos n-grams possíveis não aparece no corpus.
  - **Contagens Baixas**: Mesmo os n-grams observados têm contagens muito pequenas, frequentemente apenas uma ocorrência.

**Impacto na Qualidade das Estimativas de Probabilidade:**

- **Estimativas Não Confiáveis**: Probabilidades baseadas em poucas observações são estatisticamente instáveis.
  
- **Probabilidades Zero**: N-grams não observados recebem probabilidade zero, o que é problemático para modelagem, especialmente em aplicações como reconhecimento de fala ou tradução automática.

- **Overfitting**: O modelo pode se ajustar excessivamente aos dados de treinamento, capturando ruído ao invés de padrões reais.

**Consequências Práticas:**

- **Necessidade de Suavização**: Técnicas como suavização de Laplace, Good-Turing e Kneser-Ney são empregadas para ajustar as probabilidades, atribuindo valores não zero a n-grams não observados.

- **Limitação no Valor de $n$**: Na prática, valores de $n$ maiores que 3 ou 4 são raramente usados sem técnicas avançadas devido à esparsidade extrema.

**Conclusão:**

- **Trade-off entre Contexto e Esparsidade**: Aumentar $n$ captura mais contexto, mas agrava a esparsidade dos dados.
  
- **Equilíbrio Necessário**: É crucial encontrar um equilíbrio que permita capturar dependências relevantes sem comprometer a confiabilidade das estimativas.

- **Importância de Dados Abundantes**: Para modelos com $n$ elevado, corpora muito grandes são necessários para mitigar a esparsidade, o que nem sempre é viável.

---

**Referências Adicionais:**

- **Lei de Zipf**: Em linguística computacional, a distribuição de frequências das palavras segue a Lei de Zipf, indicando que poucas palavras ocorrem com alta frequência, enquanto a maioria ocorre raramente. Isso exacerba a esparsidade em modelos n-gram.

- **Modelos Alternativos**: Para lidar com as limitações dos modelos n-gram de alto valor de $n$, modelos baseados em redes neurais e técnicas de aprendizagem profunda são utilizados, pois podem capturar dependências de longo alcance sem sofrer tanto com a esparsidade.

## Limitações e Extensões da Hipótese de Markov

### Limitações

1. **Dependências de Longo Alcance**: ==A Hipótese de Markov falha em capturar relações entre palavras distantes==, o que é crucial para a compreensão de contextos complexos [16].

2. **Esparsidade de Dados**: Com o aumento de $n$, o número de possíveis n-grams cresce exponencialmente, resultando em muitos n-grams não observados no corpus de treinamento [17].

3. **Necessidade de Suavização**: Atribui probabilidade zero a sequências não observadas, exigindo técnicas de suavização para ajustar as probabilidades [18].

### Extensões e Melhorias

1. **Interpolação**: ==Combina modelos de diferentes ordens para melhorar as estimativas [19]. A probabilidade interpolada é dada por:==
   $$
   P_{\text{interp}}(w_n \mid w_{n-2}, w_{n-1}) = \lambda_1 P(w_n) + \lambda_2 P(w_n \mid w_{n-1}) + \lambda_3 P(w_n \mid w_{n-2}, w_{n-1})
   $$
   
   Onde $\lambda_i$ são pesos que somam 1.
   
2. **Backoff**: Recorre a modelos de ordem inferior quando não há dados suficientes para estimar um n-gram de ordem superior [20]. Usa-se a probabilidade de um (n-1)-gram quando o n-gram é desconhecido.

3. **Técnicas de Suavização**: Métodos como Add-one, Good-Turing e Kneser-Ney são utilizados para ajustar as probabilidades de n-grams não observados [21]. Por exemplo, a suavização de Kneser-Ney é especialmente eficaz para modelos de linguagem.

> ❗ **Ponto de Atenção**: Mesmo com essas extensões, modelos baseados na Hipótese de Markov têm limitações fundamentais em capturar a complexidade total da linguagem natural [22]. Dependências semânticas e contextuais de longo alcance muitas vezes ficam fora do alcance desses modelos.

## Implementação e Considerações Práticas

A implementação de modelos n-gram baseados na Hipótese de Markov envolve diversas considerações práticas:

1. **Contagem de N-grams**: Eficiência na contagem e armazenamento de n-grams é crucial, especialmente para valores altos de $n$ [23].

2. **Tratamento de Vocabulário**: Decisões sobre tokens desconhecidos (UNK) e o tamanho do vocabulário impactam diretamente na qualidade do modelo [24].

3. **Suavização e Backoff**: Implementação de técnicas adequadas para lidar com n-grams não observados é essencial para evitar probabilidades zero [25].

Exemplo de implementação simplificada de um modelo bigram em Python:

```python
import numpy as np
from collections import defaultdict, Counter

class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()

    def train(self, corpus):
        for sentence in corpus:
            tokens = ['<s>'] + sentence.split() + ['</s>']
            self.vocab.update(tokens)
            for i in range(len(tokens) - 1):
                self.bigram_counts[tokens[i]][tokens[i + 1]] += 1
                self.unigram_counts[tokens[i]] += 1
        self.vocab_size = len(self.vocab)

    def probability(self, word, context):
        bigram_count = self.bigram_counts[context][word]
        context_count = self.unigram_counts[context]
        # Suavização Add-one
        return (bigram_count + 1) / (context_count + self.vocab_size)

    def generate(self, num_words=10):
        current_word = '<s>'
        generated = []
        for _ in range(num_words):
            words = list(self.bigram_counts[current_word].keys())
            probs = [self.probability(w, current_word) for w in words]
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]
            next_word = np.random.choice(words, p=probs)
            if next_word == '</s>':
                break
            generated.append(next_word)
            current_word = next_word
        return ' '.join(generated)

# Uso do modelo
corpus = ["I am Sam", "Sam I am", "I do not like green eggs and ham"]
model = BigramModel()
model.train(corpus)
print(model.generate())
```

Este código implementa um modelo bigram com suavização Add-one, demonstrando os princípios básicos da Hipótese de Markov em ação [26].

#### Perguntas Teóricas

1. **Complexidade Computacional**: Derive a complexidade computacional e espacial para o treinamento e inferência de um modelo n-gram genérico. Considere o número de possíveis n-grams e o custo de armazenamento e busca.

2. **Impacto da Suavização**: Analise teoricamente o impacto da suavização Add-one na distribuição de probabilidades de um modelo bigram. Discuta como ela afeta a probabilidade de n-grams observados e não observados.

3. **Interpolação Linear**: Demonstre matematicamente como a interpolação linear de diferentes ordens de n-grams afeta a estimativa de probabilidade final. Mostre como os pesos $\lambda_i$ influenciam o equilíbrio entre modelos de diferentes ordens.

## Conclusão

A Hipótese de Markov, fundamental para os modelos n-gram, representa uma simplificação poderosa que permitiu avanços significativos na modelagem estatística de linguagem. Ao aproximar a história de uma sequência por um contexto limitado, torna-se tratável o problema de estimar probabilidades de sequências de palavras [27].

Embora os modelos baseados na Hipótese de Markov apresentem limitações, especialmente na captura de dependências de longo alcance, eles permanecem uma ferramenta valiosa no processamento de linguagem natural. Sua simplicidade e eficiência computacional os tornam adequados para diversas aplicações práticas [28].

À medida que o campo avança, modelos mais sofisticados, como redes neurais recorrentes e Transformers, têm superado algumas das limitações dos modelos n-gram tradicionais. No entanto, o entendimento dos princípios fundamentais da Hipótese de Markov continua sendo crucial para o desenvolvimento e compreensão de modelos de linguagem mais avançados [29].

## Perguntas Teóricas Avançadas

1. **Entropia de Processos Estocásticos**: Derive a expressão para a entropia de um processo estocástico estacionário e ergódico baseado na Hipótese de Markov e relacione-a com a perplexidade de um modelo n-gram.

2. **Comportamento Assintótico**: Analise teoricamente o comportamento assintótico da qualidade de um modelo n-gram à medida que $n$ tende ao infinito, considerando um corpus de treinamento finito. Discuta a convergência das estimativas de probabilidade.

3. **Convergência de Modelos de Markov**: Desenvolva uma prova formal demonstrando que, para qualquer distribuição de probabilidade real sobre sequências infinitas, existe uma sequência de modelos de Markov de ordem crescente que converge para essa distribuição.

4. **Comparação com Modelos de Atenção**: Compare teoricamente a capacidade de modelos baseados na Hipótese de Markov e modelos de atenção (como Transformers) em capturar dependências de longo alcance. Forneça uma análise matemática das limitações fundamentais de cada abordagem.

5. **Extensão da Hipótese de Markov**: Proponha e analise matematicamente uma extensão da Hipótese de Markov que permita capturar dependências de alcance variável de forma eficiente, mantendo a tratabilidade computacional.

## Referências

[1] "The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words." *(Trecho de n-gram language models.pdf.md)*

[2] "The bigram model, for example, approximates the probability of a word given all the previous words $P(w_n \mid w_1^{n-1})$ by using only the conditional probability of the preceding word $P(w_n \mid w_{n-1})$." *(Trecho de n-gram language models.pdf.md)*

[3] "The assumption that the probability of a word depends only on the previous word is called a Markov assumption. Markov models are the class of probabilistic models that assume we can predict the probability of some future unit without looking too far into the past." *(Trecho de n-gram language models.pdf.md)*

[4] "We can generalize the bigram (which looks one word into the past) to the trigram (which looks two words into the past) and thus to the n-gram (which looks $n-1$ words into the past)." *(Trecho de n-gram language models.pdf.md)*

[5] "Let's see a general equation for this n-gram approximation to the conditional probability of the next word in a sequence. We'll use N here to mean the n-gram size, so $N = 2$ means bigrams and $N = 3$ means trigrams." *(Trecho de n-gram language models.pdf.md)*

[6] "P(w_n \mid w_1^{n-1}) \approx P(w_n \mid w_{n-N+1}^{n-1})" *(Trecho de n-gram language models.pdf.md)*

[7] "Given the bigram assumption for the probability of an individual word, we can compute the probability of a complete word sequence by substituting Eq. 3.7 into Eq. 3.4: $P(w_1^n) \approx \prod_{k=1}^n P(w_k \mid w_{k-1})$" *(Trecho de n-gram language models.pdf.md)*

[8] "How do we estimate these bigram or n-gram probabilities? An intuitive way to estimate probabilities is called maximum likelihood estimation or MLE. We get the MLE estimate for the parameters of an n-gram model by getting counts from a corpus, and normalizing the counts so that they lie between 0 and 1." *(Trecho de n-gram language models.pdf.md)*

[9] "For example, to compute a particular bigram probability of a word $w_n$ given a previous word $w_{n-1}$, we'll compute the count of the bigram $C(w_{n-1} w_n)$ and normalize by the sum of all the bigrams that share the same first word $w_{n-1}$:" *(Trecho de n-gram language models.pdf.md)*

[10] "P(w_n \mid w_{n-1}) = \frac{C(w_{n-1} w_n)}{\sum_w C(w_{n-1} w)}" *(Trecho de n-gram language models.pdf.md)*

[11] "We can simplify this equation, since the sum of all bigram counts that start with a given word $w_{n-1}$ must be equal to the unigram count for that word $w_{n-1}$ (the reader should take a moment to be convinced of this): $P(w_n \mid w_{n-1}) = \frac{C(w_{n-1} w_n)}{C(w_{n-1})}$" *(Trecho de n-gram language models.pdf.md)*

[12] "For the general case of MLE n-gram parameter estimation: $P(w_n \mid w_{n-N+1}^{n-1}) = \frac{C(w_{n-N+1}^{n-1} w_n)}{C(w_{n-N+1}^{n-1})}$" *(Trecho de n-gram language models.pdf.md)*

[13] "Equation 3.12 (like Eq. 3.11) estimates the n-gram probability by dividing the observed frequency of a particular sequence by the observed frequency of a prefix. This ratio is called a relative frequency." *(Trecho de n-gram language models.pdf.md)*

[14] "We said above that this use of relative frequencies is an estimate, called the maximum likelihood estimate or MLE, of the probability $P(w_n \mid w_{n-N+1}^{n-1})$." *(Trecho de n-gram language models.pdf.md)*

[15] "To compute the probability of a word sequence, we need to prefix it with $n-1$ start symbols $\langle s \rangle$." *(Trecho de n-gram language models.pdf.md)*

[16] "One of the biggest problems with the MLE estimates is the sparseness of the data. Even with very large corpora, we will have zero counts for a vast number of possible n-grams." *(Trecho de n-gram language models.pdf.md)*

[17] "This data sparsity causes problems when we try to compute probabilities, as we cannot have probabilities of zero for events that might occur." *(Trecho de n-gram language models.pdf.md)*

[18] "To address this issue, we use smoothing techniques to adjust the maximum likelihood estimates to produce more reliable probabilities." *(Trecho de n-gram language models.pdf.md)*

[19] "Interpolation is a method where we combine different order n-gram models by assigning weights to them." *(Trecho de n-gram language models.pdf.md)*

[20] "Backoff models back off to lower-order n-gram models when the higher-order n-gram has zero counts." *(Trecho de n-gram language models.pdf.md)*

[21] "Common smoothing techniques include Laplace (add-one), Good-Turing, and Kneser-Ney smoothing." *(Trecho de n-gram language models.pdf.md)*

[22] "Despite these smoothing techniques, n-gram models have limitations in capturing the complexities of natural language, particularly long-distance dependencies." *(Trecho de n-gram language models.pdf.md)*

[23] "Efficient data structures and algorithms are required to store and retrieve n-gram counts, especially for large corpora and higher-order n-grams." *(Trecho de n-gram language models.pdf.md)*

[24] "Handling unknown words is critical, often involving the use of an unknown word token and careful consideration of the vocabulary size." *(Trecho de n-gram language models.pdf.md)*

[25] "Implementation of smoothing and backoff requires careful programming to ensure that probabilities remain valid and the model performs well." *(Trecho de n-gram language models.pdf.md)*

[26] "This simple bigram model demonstrates how Markov assumptions enable us to build probabilistic models of language." *(Trecho de n-gram language models.pdf.md)*

[27] "The Markov assumption reduces the complexity of modeling language by limiting the dependencies between words." *(Trecho de n-gram language models.pdf.md)*

[28] "While more advanced models have surpassed n-gram models in performance, understanding n-grams is fundamental to the field of computational linguistics." *(Trecho de n-gram language models.pdf.md)*

[29] "Modern models like neural networks and Transformers build upon the foundations established by n-gram models and the Markov assumption." *(Trecho de n-gram language models.pdf.md)*