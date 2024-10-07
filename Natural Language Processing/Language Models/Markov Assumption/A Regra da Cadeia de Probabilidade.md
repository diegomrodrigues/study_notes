## A Regra da Cadeia de Probabilidade em Modelos de Linguagem

<imagem: Um diagrama mostrando uma sequência de palavras conectadas por setas, representando as dependências probabilísticas da regra da cadeia>

### Introdução

A **regra da cadeia de probabilidade** é um conceito fundamental na modelagem de linguagem probabilística, ==permitindo decompor a probabilidade conjunta de uma sequência de palavras em um produto de probabilidades condicionais mais simples [1][2]==. Esta abordagem é crucial para calcular e estimar as probabilidades de sequências de tokens em linguagem natural, ==formando a base teórica para muitos modelos de linguagem modernos.==

Em processamento de linguagem natural, ==é essencial modelar a probabilidade de sequências de palavras para tarefas como previsão de palavras, geração de texto e reconhecimento de fala==. A regra da cadeia fornece um método sistemático para este propósito, permitindo que modelos computem a probabilidade de uma sequência inteira a partir das probabilidades condicionais das palavras individuais, dado seu contexto precedente.

### Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Regra da Cadeia**           | Método para fatorar a probabilidade conjunta de uma sequência em um produto de probabilidades condicionais [2]. |
| **Probabilidade Condicional** | A probabilidade de uma palavra dado o contexto das palavras anteriores [3]. |
| **Modelo N-gram**             | Aproximação que considera apenas as N-1 palavras anteriores como contexto, simplificando o cálculo das probabilidades [4]. |

> ⚠️ **Nota Importante**: Embora a regra da cadeia permita modelar sequências de comprimento arbitrário, na prática, limitações computacionais e de disponibilidade de dados levam a aproximações como modelos N-gram [5].

### Formulação Matemática da Regra da Cadeia

A regra da cadeia para uma sequência de palavras $w = (w_1, w_2, \dots, w_M)$ é expressa matematicamente como [6]:

$$
p(w) = p(w_1, w_2, \dots, w_M) = p(w_1) \times p(w_2 \mid w_1) \times p(w_3 \mid w_1, w_2) \times \dots \times p(w_M \mid w_1, w_2, \dots, w_{M-1})
$$

Onde:

- $p(w)$ é a probabilidade conjunta da sequência completa.
- $p(w_m \mid w_1, w_2, \dots, w_{m-1})$ ==é a probabilidade condicional da m-ésima palavra dado todo o contexto anterior.==

Esta formulação é derivada diretamente das definições básicas de probabilidade condicional e permite decompor o problema complexo de estimar a probabilidade de uma sequência inteira em subproblemas mais gerenciáveis de estimar probabilidades condicionais [7].

No entanto, calcular estas probabilidades condicionais para sequências longas é impraticável devido ao número exponencial de possíveis contextos que precisam ser considerados, especialmente considerando o tamanho do vocabulário em linguagens naturais.

### Aplicação em Modelos de Linguagem N-gram

Para tornar o problema tratável, os modelos N-gram fazem uma aproximação crucial da regra da cadeia, ==limitando o contexto considerado às N-1 palavras anteriores [8]:==

$$
p(w_m \mid w_1, w_2, \dots, w_{m-1}) \approx p(w_m \mid w_{m-n+1}, w_{m-n+2}, \dots, w_{m-1})
$$

Esta aproximação resulta na seguinte fórmula para a probabilidade aproximada de uma sequência [9]:

$$
p(w_1, w_2, \dots, w_M) \approx \prod_{m=1}^M p(w_m \mid w_{m-n+1}, \dots, w_{m-1})
$$

> 💡 **Insight**: Esta aproximação reduz drasticamente o número de parâmetros a serem estimados, de $V^M$ para $V^n$, onde $V$ é o tamanho do vocabulário, tornando o modelo computacionalmente viável. No entanto, isto introduz um viés ao ignorar dependências de longo alcance [10].

### Estimação de Parâmetros

Para estimar as probabilidades condicionais em modelos N-gram, geralmente se utiliza a estimativa de frequência relativa, calculada a partir de um corpus de treinamento [11]:

$$
p(w_m \mid w_{m-2}, w_{m-1}) = \frac{\text{count}(w_{m-2}, w_{m-1}, w_m)}{\sum_{w'} \text{count}(w_{m-2}, w_{m-1}, w')}
$$

Onde:

- $\text{count}(w_{m-2}, w_{m-1}, w_m)$ é a contagem de ocorrência do trigram $(w_{m-2}, w_{m-1}, w_m)$ no corpus.
- O denominador soma as contagens de todos os trigrams que compartilham o mesmo contexto $(w_{m-2}, w_{m-1})$.

No entanto, esta abordagem pode levar a problemas de esparsidade de dados, especialmente para N-grams de ordem superior. Muitas sequências possíveis nunca aparecem no corpus de treinamento, resultando em probabilidades estimadas como zero [12].

Para mitigar este problema, técnicas de suavização (smoothing) são empregadas, como Laplace smoothing, Good-Turing discounting e métodos de interpolação, que ajustam as estimativas de probabilidade para dar algum peso a eventos não observados.

### Desafios e Limitações

#### 👎 Desvantagens da Aproximação N-gram

- **Esparsidade de Dados**: À medida que o valor de $n$ aumenta, o número de possíveis N-grams cresce exponencialmente, e muitos deles não são observados no conjunto de treinamento [13].
- **Perda de Dependências de Longo Alcance**: Os modelos N-gram não capturam dependências além das N-1 palavras anteriores, ignorando informações que podem ser relevantes para a previsão da próxima palavra [14].

Estas limitações refletem um trade-off entre a capacidade do modelo de capturar dependências contextuais e a viabilidade computacional. Modelos com maior valor de $n$ podem teoricamente capturar mais contexto, mas exigem mais dados para estimativas confiáveis e maior capacidade computacional.

Para abordar estas limitações, técnicas avançadas, como modelos baseados em árvores e modelos de linguagem neurais, foram desenvolvidas [15].

### Modelos de Linguagem Neurais e a Regra da Cadeia

Modelos de linguagem baseados em redes neurais, como Redes Neurais Recorrentes (RNNs) e Transformers, estendem a aplicação da regra da cadeia ao incorporar mecanismos que podem teoricamente capturar dependências de longo alcance sem a necessidade de limitar o contexto a N-1 palavras [16].

#### Redes Neurais Recorrentes (RNNs)

Em um modelo RNN, o estado oculto é atualizado iterativamente, permitindo que informações de todo o contexto anterior influenciem a previsão da próxima palavra. A probabilidade condicional é calculada como:

$$
p(w_{m+1} \mid w_1, w_2, \dots, w_m) = \text{Softmax}( \mathbf{W} \cdot \mathbf{h}_m + \mathbf{b} )
$$

Onde:

- $\mathbf{h}_m$ é o estado oculto no passo $m$, que depende de $\mathbf{h}_{m-1}$ e da entrada atual $\mathbf{x}_m$.
- $\mathbf{W}$ e $\mathbf{b}$ são parâmetros do modelo.
- $\text{Softmax}$ produz uma distribuição de probabilidade sobre o vocabulário.

Este mecanismo permite que o modelo aprenda representações contextuais ricas, potencialmente capturando dependências de longo alcance [17].

#### Transformers

Modelos baseados em Transformers utilizam mecanismos de atenção que permitem acesso direto a todos os estados anteriores, superando limitações das RNNs em capturar dependências de longo prazo devido a problemas de gradientes. A probabilidade condicional é calculada considerando todas as posições anteriores na sequência através de mecanismos de autoatenção.

Embora os modelos neurais sejam mais poderosos em capturar dependências complexas, eles ainda se baseiam na regra da cadeia para decompor a probabilidade conjunta em probabilidades condicionais.

### Conclusão

A regra da cadeia de probabilidade é um princípio central na modelagem de linguagem, permitindo a decomposição da probabilidade conjunta de uma sequência em probabilidades condicionais. Esta abordagem fundamenta desde modelos N-gram clássicos até arquiteturas neurais avançadas, como RNNs e Transformers [19].

Apesar das limitações práticas que levaram ao desenvolvimento de modelos mais sofisticados, a regra da cadeia continua sendo fundamental para nossa compreensão e abordagem da modelagem probabilística de sequências de texto. Compreender suas vantagens e trade-offs é essencial para desenvolver modelos eficazes em processamento de linguagem natural.

### Perguntas Teóricas

1. **Perplexidade e Log-Verossimilhança**: Derive a fórmula da perplexidade de um modelo de linguagem baseado na regra da cadeia e explique como ela se relaciona com a log-verossimilhança.

   *A perplexidade é uma medida de quão bem um modelo de linguagem prevê uma amostra. É definida como a exponencial da entropia cruzada média entre a distribuição real e a do modelo. Para uma sequência de teste $w_1, w_2, \dots, w_N$, a perplexidade PP é dada por:*

   $$
   PP = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(w_i \mid w_1, \dots, w_{i-1}) \right)
   $$

   *Isto está diretamente relacionado à log-verossimilhança, que é a soma dos logaritmos das probabilidades condicionais preditas pelo modelo.*

2. **Subestimação de Dependências de Longo Alcance**: Demonstre matematicamente por que a aproximação N-gram da regra da cadeia pode levar a uma subestimação de dependências de longo alcance em sequências de texto.

   *Como os modelos N-gram consideram apenas as N-1 palavras anteriores, eles assumem independência condicional das palavras além deste contexto. Isto significa que qualquer dependência estatística além deste escopo é ignorada, potencialmente subestimando a probabilidade de sequências que dependem de contexto distante.*

3. **Incorporação de Contexto Bidirecional**: Como a regra da cadeia poderia ser modificada para incorporar informações bidirecionais do contexto, e quais seriam as implicações teóricas dessa modificação?

   *Uma modificação possível é considerar modelos que condicionam não apenas no passado, mas também no futuro. No entanto, isto viola a causalidade e não é adequado para tarefas de geração de texto. Modelos como Conditional Random Fields (CRFs) e modelos baseados em redes neurais bidirecionais podem incorporar contexto bidirecional para tarefas de etiquetagem e classificação, mas não para previsão causal.*

### Perguntas Teóricas Avançadas

1. **Complexidade Computacional**: Derive a complexidade computacional e espacial de um modelo de linguagem baseado na regra da cadeia sem aproximações, e compare com a complexidade de um modelo N-gram e um modelo RNN.

   *Sem aproximações, o número de parâmetros necessários para modelar as probabilidades condicionais é da ordem de $V^M$, onde $V$ é o tamanho do vocabulário e $M$ é o comprimento máximo da sequência, o que é computacionalmente intratável. Em contraste, modelos N-gram requerem $V^n$ parâmetros, e modelos RNN utilizam um número fixo de parâmetros independentes do tamanho do contexto.*

2. **Incorporação de Lookahead**: Proponha e analise matematicamente uma modificação da regra da cadeia que incorpore informações futuras (lookahead) de forma eficiente, mantendo a causalidade do modelo.

   *Incorporar informações futuras enquanto mantém a causalidade é um desafio. Uma abordagem é utilizar modelos de encoders-decoder com atenção, onde o encoder processa a sequência inteira e o decoder gera a sequência passo a passo, mas isso não é causal no sentido estrito. Alternativamente, pode-se usar modelos que preveem múltiplos passos à frente, mas isto muda a natureza do problema.*

3. **Modelagem de Múltiplos Passos Futuros**: Demonstre como a regra da cadeia pode ser estendida para modelar não apenas a próxima palavra, mas distribuições sobre múltiplos passos futuros, e discuta as implicações teóricas e práticas dessa extensão.

   *A regra da cadeia pode ser estendida para considerar distribuições conjuntas de múltiplas palavras futuras. No entanto, isto aumenta a complexidade computacional exponencialmente. Em prática, modelos como Transformers podem gerar distribuições para múltiplos tokens simultaneamente durante o treinamento, mas a geração ainda é feita token por token para manter a consistência.*

4. **Ambiguidades Lexicais e Sintáticas**: Analise teoricamente como a regra da cadeia poderia ser adaptada para lidar com ambiguidades lexicais e sintáticas em linguagens naturais, propondo uma formulação matemática que incorpore essas incertezas.

   *Uma abordagem é introduzir variáveis latentes que modelam estados ocultos, como categorias sintáticas ou significados lexicais, resultando em modelos como Modelos Ocultos de Markov (HMMs) ou Gramáticas Livres de Contexto Probabilísticas (PCFGs). A regra da cadeia é então aplicada às sequências observáveis e às variáveis latentes, incorporando as incertezas inerentes.*

5. **Capacidade de Aproximação de Transformers**: Desenvolva uma prova formal mostrando que, sob certas condições, um modelo de linguagem baseado em atenção (como os Transformers) pode aproximar arbitrariamente bem a verdadeira distribuição de probabilidade conjunta definida pela regra da cadeia.

   *Esta demonstração envolveria mostrar que, dado poder computacional e dados suficientes, os mecanismos de atenção nos Transformers podem modelar qualquer função mensurável das sequências de entrada, incluindo as dependências definidas pela distribuição conjunta original. Isto se baseia na capacidade universal de aproximação das redes neurais profundas.*

### Referências

[1] "In probabilistic classification, the problem is to compute the probability of a label, conditioned on the text. Let's now consider the inverse problem: computing the probability of text itself." *(Trecho de Language Models_143-162.pdf.md)*

[2] "The chain rule (see § A.2): $p(w) = p(w_1, w_2, \dots, w_M) = p(w_1) \times p(w_2 \mid w_1) \times p(w_3 \mid w_1, w_2) \times \dots \times p(w_M \mid w_1, w_2, \dots, w_{M-1})$" *(Trecho de Language Models_143-162.pdf.md)*

[3] "Each element in the product is the probability of a word given all its predecessors." *(Trecho de Language Models_143-162.pdf.md)*

[4] "n-gram models make a crucial simplifying approximation: they condition on only the past n − 1 words." *(Trecho de Language Models_143-162.pdf.md)*

[5] "To solve this problem, n-gram models make a crucial simplifying approximation: they condition on only the past n − 1 words." *(Trecho de Language Models_143-162.pdf.md)*

[6] "p(w) = p(w_1, w_2, ..., w_M) = p(w_1) × p(w_2 | w_1) × p(w_3 | w_1, w_2) × ... × p(w_M | w_1, w_2, ..., w_{M−1})" *(Trecho de Language Models_143-162.pdf.md)*

[7] "Each element in the product is the probability of a word given all its predecessors. We can think of this as a word prediction task: given the context Computers are, we want to compute a probability over the next token." *(Trecho de Language Models_143-162.pdf.md)*

[8] "n-gram models make a crucial simplifying approximation: they condition on only the past n − 1 words. $p(w_m \mid w_{m-1}, ..., w_1) \approx p(w_m \mid w_{m-n+1}, ..., w_{m-1})$" *(Trecho de Language Models_143-162.pdf.md)*

[9] "This means that the probability of a sentence w can be approximated as $p(w_1, ..., w_M) \approx \prod_{m=1}^M p(w_m \mid w_{m-n+1}, ..., w_{m-1})$" *(Trecho de Language Models_143-162.pdf.md)*

[10] "This model requires estimating and storing the probability of only $V^n$ events, which is exponential in the order of the n-gram, and not $V^M$, which is exponential in the length of the sentence." *(Trecho de Language Models_143-162.pdf.md)*

[11] "The n-gram probabilities can be computed by relative frequency estimation, $p(w_m \mid w_{m-1}, w_{m-2}) = \frac{\text{count}(w_{m-2}, w_{m-1}, w_m)}{\sum_{w'} \text{count}(w_{m-2}, w_{m-1}, w')}$" *(Trecho de Language Models_143-162.pdf.md)*

[12] "Language is full of long-range dependencies that we cannot capture because n is too small; at the same time, language datasets are full of rare phenomena, whose probabilities we fail to estimate accurately because n is too large." *(Trecho de Language Models_143-162.pdf.md)*

[13] "Limited data is a persistent problem in estimating language models." *(Trecho de Language Models_143-162.pdf.md)*

[14] "In each example, the words written in bold depend on each other: the likelihood of their depends on knowing that gorillas is plural, and the likelihood of crashed depends on knowing that the subject is a computer. If the n-grams are not big enough to capture this context, then the resulting language model would offer probabilities that are too low for these sentences" *(Trecho de Language Models_143-162.pdf.md)*

[15] "It is therefore necessary to add additional inductive biases to n-gram language models. This section covers some of the most intuitive and common approaches, but there are many more (see Chen and Goodman, 1999)." *(Trecho de Language Models_143-162.pdf.md)*

[16] "N-gram language models have been largely supplanted by neural networks. These models do not make the n-gram assumption of restricted context; indeed, they can incorporate arbitrarily distant contextual information, while remaining computationally and statistically tractable." *(Trecho de Language Models_143-162.pdf.md)*

[17] "$p(w_{m+1} | w_1, w_2, ..., w_m) = \text{SoftMax}([\beta_1 \cdot v_u, \beta_2 \cdot v_u, ..., \beta_V \cdot v_u])$" *(Trecho de Language Models_143-162.pdf.md)*

[18] "Using the Pytorch library, train an LSTM language model from the Wikitext training corpus. After each epoch of training, compute its perplexity on the Wikitext validation corpus. Stop training when the perplexity stops improving." *(Trecho de Language Models_143-162.pdf.md)*

[19] "The first insight behind neural language models is to treat word prediction as a discriminative learning task. The goal is to compute the probability $p(w \mid u)$, where $w \in V$ is a word, and $u$ is the context, which depends on the previous words." *(Trecho de Language Models_143-162.pdf.md)*