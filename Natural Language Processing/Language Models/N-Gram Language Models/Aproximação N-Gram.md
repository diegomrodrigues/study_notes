# Aproximação N-Gram em Modelos de Linguagem

<imagem: Um diagrama mostrando uma sequência de palavras com janelas deslizantes de tamanho n representando n-gramas>

## Introdução

A **aproximação n-gram** é uma técnica fundamental em modelagem de linguagem que permite calcular a probabilidade de sequências de palavras de forma computacionalmente tratável [1]. Ela aborda o problema da explosão combinatória que ocorre ao tentar modelar diretamente a probabilidade conjunta de longas sequências de palavras, tornando viável a estimação de probabilidades em corpora finitos.

## Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **N-gram**                 | Subsequência contígua de n itens (geralmente palavras) em uma dada sequência de texto [2]. |
| **Aproximação Markoviana** | Suposição de que a probabilidade de uma palavra depende apenas das n-1 palavras anteriores [3]. |
| **Cadeia de Markov**       | Processo estocástico onde a probabilidade de cada evento depende apenas do estado imediatamente anterior [4]. |

> ⚠️ **Nota Importante**: A aproximação n-gram introduz um viés no modelo, mas é necessária para tornar o problema tratável computacionalmente com dados finitos [5].

## Formulação Matemática

A aproximação n-gram baseia-se na aplicação da regra da cadeia de probabilidade e na introdução de uma suposição simplificadora [6]:

1. **Regra da cadeia**:

   $$p(w) = p(w_1, w_2, \ldots, w_M) = p(w_1) \times p(w_2 \mid w_1) \times p(w_3 \mid w_1, w_2) \times \ldots \times p(w_M \mid w_1, w_2, \ldots, w_{M-1})$$

2. **Aproximação n-gram**:

   $$p(w_m \mid w_1, w_2, \ldots, w_{m-1}) \approx p(w_m \mid w_{m-n+1}, \ldots, w_{m-1})$$

Combinando essas duas ideias, obtemos a formulação da aproximação n-gram [7]:

$$p(w_1, w_2, \ldots, w_M) \approx \prod_{m=1}^M p(w_m \mid w_{m-n+1}, \ldots, w_{m-1})$$

==Esta aproximação reduz drasticamente o número de parâmetros a serem estimados, de $V^M$ para $V^n$, onde $V$ é o tamanho do vocabulário [8].==

### Exemplo: Modelo Bigram (n=2)

Para ilustrar, considere a aproximação bigram para a frase "I like black coffee" [9]:

$$p(\text{I like black coffee}) = p(\text{I} \mid \langle s \rangle) \times p(\text{like} \mid \text{I}) \times p(\text{black} \mid \text{like}) \times p(\text{coffee} \mid \text{black}) \times p(\langle /s \rangle \mid \text{coffee})$$

Onde $\langle s \rangle$ e $\langle /s \rangle$ são símbolos especiais que representam o início e o fim da frase, respectivamente.

### Estimação de Probabilidades

As probabilidades dos n-gramas são tipicamente estimadas usando contagens relativas [10]:

$$p(w_m \mid w_{m-1}, \ldots, w_{m-n+1}) = \frac{\text{count}(w_{m-n+1}, \ldots, w_{m-1}, w_m)}{\sum_{w'} \text{count}(w_{m-n+1}, \ldots, w_{m-1}, w')}$$

Essa abordagem assume que a frequência de ocorrência no corpus é uma estimativa confiável da probabilidade real.

## Discussão Teórica

### Derivação da Aproximação N-Gram

A aproximação n-gram é derivada da necessidade de simplificar o cálculo da probabilidade conjunta de uma sequência de palavras. ==Pela regra da cadeia de probabilidade, a probabilidade conjunta pode ser decomposta em probabilidades condicionais [6]:==

$$p(w_1, w_2, \ldots, w_M) = \prod_{m=1}^M p(w_m \mid w_1, w_2, \ldots, w_{m-1})$$

No entanto, ==calcular essas probabilidades condicionais é impraticável devido ao número exponencial de possíveis históricos== Para tornar o problema tratável, ==assumimos que cada palavra depende apenas das n-1 palavras anteriores:==

$$p(w_m \mid w_1, w_2, \ldots, w_{m-1}) \approx p(w_m \mid w_{m-n+1}, \ldots, w_{m-1})$$

Essa suposição simplifica o modelo, reduzindo o contexto considerado e permitindo a estimação das probabilidades com um conjunto finito de dados.

### Trade-off entre Viés e Variância

A escolha do valor de $n$ afeta o equilíbrio entre viés e variância no modelo n-gram:

- **Viés**: Com $n$ pequeno, o modelo é mais simples e pode não capturar todas as dependências linguísticas, resultando em alto viés.
- **Variância**: Com $n$ grande, o modelo é mais complexo, podendo ajustar-se muito bem aos dados de treinamento e generalizar mal para novos dados, resultando em alta variância.

Matematicamente, conforme $n$ aumenta, o número de parâmetros $V^n$ cresce exponencialmente, exigindo mais dados para estimar as probabilidades com confiança [16].

### Redução da Complexidade Computacional

Sem a aproximação n-gram, a complexidade computacional seria $O(V^M)$, onde $M$ é o comprimento da sequência. Com a aproximação n-gram, a complexidade é reduzida para $O(V^n)$, tornando o problema computacionalmente viável mesmo para grandes vocabulários [8].

## Vantagens e Desvantagens

| 👍 **Vantagens**                                              | 👎 **Desvantagens**                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Redução de Complexidade**: Diminui drasticamente o número de parâmetros a serem estimados [11]. | **Perda de Dependências de Longo Alcance**: Incapaz de capturar relações que ultrapassam o contexto limitado [12]. |
| **Eficiência Computacional**: Permite estimação eficiente com dados finitos [13]. | **Probabilidade Zero para Sequências Válidas**: Pode atribuir probabilidade zero a sequências não observadas [14]. |
| **Captura Padrões Locais**: Eficaz na modelagem de dependências locais [15]. | **Sensibilidade à Escolha de n**: Valor inadequado de $n$ pode levar a overfitting ou underfitting [16]. |

## Técnicas de Suavização

Para lidar com n-gramas não observados e evitar probabilidades zero, várias técnicas de suavização são empregadas [17]:

1. **Suavização de Lidstone**: Adiciona um pseudo-contagem $\alpha$ a todas as contagens [18]:

   $$p_{\text{smooth}}(w_m \mid w_{m-1}) = \frac{\text{count}(w_{m-1}, w_m) + \alpha}{\sum_{w' \in V} \text{count}(w_{m-1}, w') + V\alpha}$$

2. **Descontagem Absoluta**: Subtrai um valor fixo $d$ de todas as contagens não nulas, redistribuindo a probabilidade [19].

3. **Backoff de Katz**: Utiliza modelos de ordem inferior quando as contagens de ordem superior são insuficientes [20].

4. **Suavização de Kneser-Ney**: Ajusta a distribuição considerando a "diversidade" de contextos em que uma palavra aparece, sendo eficaz para n-gramas raros [21].

> 💡 **Destaque**: A suavização de Kneser-Ney é considerada a técnica de ponta para modelos n-gram, com embasamento teórico robusto em estatística não-paramétrica [22].

## Implementação em Python

Exemplo de implementação de um modelo bigram com suavização de Lidstone usando Python e NLTK:

```python
import nltk
from collections import defaultdict

class BigramModel:
    def __init__(self, alpha=0.1):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
        self.alpha = alpha
    
    def train(self, corpus):
        for sentence in corpus:
            tokens = ['<s>'] + nltk.word_tokenize(sentence.lower()) + ['</s>']
            self.vocab.update(tokens)
            for i in range(len(tokens)-1):
                self.bigram_counts[tokens[i]][tokens[i+1]] += 1
                self.unigram_counts[tokens[i]] += 1
        self.vocab_size = len(self.vocab)
    
    def probability(self, word, context):
        numerator = self.bigram_counts[context][word] + self.alpha
        denominator = self.unigram_counts[context] + self.alpha * self.vocab_size
        return numerator / denominator

# Uso
corpus = ["I like black coffee", "I love dark coffee"]
model = BigramModel()
model.train(corpus)
print(model.probability("coffee", "black"))
```

Este código demonstra os conceitos básicos de um modelo bigram com suavização de Lidstone, conforme descrito [23].

## Conclusão

A aproximação n-gram é uma técnica essencial em modelagem de linguagem que permite lidar com a complexidade de estimar probabilidades de sequências de palavras [24]. Embora tenha limitações na captura de dependências de longo alcance, sua simplicidade e eficácia a tornam uma ferramenta valiosa, servindo de base para técnicas mais avançadas, como modelos neurais de linguagem [25].

## Perguntas Teóricas Avançadas

1. **Perplexidade e Entropia Cruzada**: Derive a fórmula para a perplexidade de um modelo n-gram e explique como ela se relaciona com a entropia cruzada e a verossimilhança dos dados.

2. **Comparação com Modelos Neurais**: Compare teoricamente a capacidade de modelar dependências de longo alcance entre modelos n-gram e modelos de linguagem baseados em redes neurais recorrentes (RNNs). Como isso se reflete nas respectivas funções objetivo?

3. **Efetividade da Suavização de Kneser-Ney**: Demonstre matematicamente por que a suavização de Kneser-Ney é particularmente eficaz para n-gramas de baixa frequência. Como isso se relaciona com o conceito de "diversidade" de contextos das palavras?

4. **Convergência da Aproximação N-Gram**: Desenvolva uma prova formal de que a aproximação n-gram converge para a distribuição verdadeira à medida que $n$ aumenta, assumindo um processo estacionário subjacente.

5. **Complexidade de Kolmogorov**: Analise o impacto teórico da escolha do valor de $n$ na complexidade de Kolmogorov da distribuição modelada. Como isso se relaciona com o princípio da descrição mínima (MDL)?

## Referências

[1] "In probabilistic classification, the problem is to compute the probability of a label, conditioned on the text. Let's now consider the inverse problem: computing the probability of text itself." *(Trecho de Language Models_143-162.pdf.md)*

[2] "n-gram language models, which compute the probability of a sequence as the product of probabilities of subsequences." *(Trecho de Language Models_143-162.pdf.md)*

[3] "n-gram models make a crucial simplifying approximation: they condition on only the past n - 1 words." *(Trecho de Language Models_143-162.pdf.md)*

[4] "This means that the probability of a sentence w can be approximated as..." *(Trecho de Language Models_143-162.pdf.md)*

[5] "We therefore need to introduce bias to have a chance of making reliable estimates from finite training data." *(Trecho de Language Models_143-162.pdf.md)*

[6] "p(w) = p(w_1, w_2, ..., w_M) = p(w_1) × p(w_2 | w_1) × p(w_3 | w_1, w_2) × ... × p(w_M | w_1, w_2, ..., w_{M-1})" *(Trecho de Language Models_143-162.pdf.md)*

[7] "p(w_m | w_1, ..., w_{m-1}) ≈ p(w_m | w_{m-n+1}, ..., w_{m-1})" *(Trecho de Language Models_143-162.pdf.md)*

[8] "This model requires estimating and storing the probability of only V^n events, which is exponential in the order of the n-gram, and not V^M, which is exponential in the length of the sentence." *(Trecho de Language Models_143-162.pdf.md)*

[9] "p(I like black coffee) = p(I | ⟨s⟩) × p(like | I) × p(black | like) × p(coffee | black) × p(⟨/s⟩ | coffee)." *(Trecho de Language Models_143-162.pdf.md)*

[10] "p(w_m | w_{m-1}, w_{m-2}) = count(w_{m-2}, w_{m-1}, w_m) / sum_{w'} count(w_{m-2}, w_{m-1}, w')" *(Trecho de Language Models_143-162.pdf.md)*

[11] "This means that we also haven't really made any progress: to compute the conditional probability p(w_M | w_{M-1}, w_{M-2}, ..., w_1), we would need to model V^{M-1} contexts. Such a distribution cannot be estimated from any realistic sample of text." *(Trecho de Language Models_143-162.pdf.md)*

[12] "Language is full of long-range dependencies that we cannot capture because n is too small." *(Trecho de Language Models_143-162.pdf.md)*

[13] "To solve this problem, n-gram models make a crucial simplifying approximation: they condition on only the past n - 1 words." *(Trecho de Language Models_143-162.pdf.md)*

[14] "Even grammatical sentences will have probability zero if they have not occurred in the training data." *(Trecho de Language Models_143-162.pdf.md)*

[15] "The hyperparameter n controls the size of the context used in each conditional probability." *(Trecho de Language Models_143-162.pdf.md)*

[16] "If this is misspecified, the language model will perform poorly." *(Trecho de Language Models_143-162.pdf.md)*

[17] "It is therefore necessary to add additional inductive biases to n-gram language models." *(Trecho de Language Models_143-162.pdf.md)*

[18] "p_{\text{smooth}}(w_m | w_{m-1}) = (count(w_{m-1}, w_m) + α) / (sum_{w' ∈ V} count(w_{m-1}, w') + Vα)" *(Trecho de Language Models_143-162.pdf.md)*

[19] "Another approach would be to borrow the same amount of probability mass from all observed n-grams, and redistribute it among only the unobserved n-grams. This is called absolute discounting." *(Trecho de Language Models_143-162.pdf.md)*

[20] "Instead, we can backoff to a lower-order language model: if you have trigrams, use trigrams; if you don't have trigrams, use bigrams; if you don't even have bigrams, use unigrams. This is called Katz backoff." *(Trecho de Language Models_143-162.pdf.md)*

[21] "Kneser-Ney smoothing is based on absolute discounting, but it redistributes the resulting probability mass in a different way from Katz backoff." *(Trecho de Language Models_143-162.pdf.md)*

[22] "Empirical evidence points to Kneser-Ney smoothing as the state-of-art for n-gram language modeling (Goodman, 2001)." *(Trecho de Language Models_143-162.pdf.md)*

[23] "Using the Pytorch library, train an LSTM language model from the Wikitext training corpus. After each epoch of training, compute its perplexity on the Wikitext validation corpus. Stop training when the perplexity stops improving." *(Trecho de Language Models_143-162.pdf.md)*

[24] "N-gram language models have been largely supplanted by neural networks. These models do not make the n-gram assumption of restricted context; indeed, they can incorporate arbitrarily distant contextual information, while remaining computationally and statistically tractable." *(Trecho de Language Models_143-162.pdf.md)*

[25] "The first insight behind neural language models is to treat word prediction as a discriminative learning task." *(Trecho de Language Models_143-162.pdf.md)*