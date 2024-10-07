## Estimativa de Frequência Relativa em Modelos de Linguagem: Desafios e Limitações

<imagem: Um gráfico mostrando a curva de frequência relativa de palavras em um corpus, com uma longa cauda de palavras raras>

### Introdução

A **estimativa de frequência relativa** é uma abordagem fundamental na modelagem probabilística da linguagem natural, servindo como alicerce para modelos de linguagem (Language Models, LMs). ==Este método intuitivo busca calcular a probabilidade de sequências de tokens com base em suas ocorrências observadas em um corpus de treinamento [1]==. Embora conceitualmente simples e estatisticamente ==consistente como um estimador de máxima verossimilhança==, essa técnica enfrenta desafios práticos significativos. ==Problemas relacionados à **esparsidade de dados** e à **alta variância** das estimativas comprometem a capacidade dos modelos de generalizar para sequências não observadas==, especialmente ao lidar com sequências longas ou vocabulários extensos [2].

### Conceitos Fundamentais

| Conceito                 | Explicação                                                   |
| ------------------------ | ------------------------------------------------------------ |
| **Frequência Relativa**  | A proporção de vezes que uma sequência específica ocorre em relação ao total de sequências observadas [3]. É utilizada como estimativa da probabilidade dessa sequência no modelo de linguagem. |
| **Esparsidade de Dados** | ==O fenômeno em que, devido ao vasto número de possíveis sequências em linguagem natural, muitas sequências gramaticalmente válidas não aparecem no corpus de treinamento, resultando em probabilidades estimadas iguais a zero para essas sequências [4].== Isso dificulta a capacidade do modelo de atribuir probabilidades significativas a eventos não observados. |
| **Variância**            | ==A medida da instabilidade nas estimativas de probabilidade devido ao tamanho limitado do corpus de treinamento, especialmente para eventos raros [5]==. Alta variância nas estimativas pode levar a desempenho inconsistente do modelo em dados não vistos. |

> ⚠️ **Nota Importante**: ==A estimativa de frequência relativa é um estimador não enviesado; no limite teórico de dados infinitos, as estimativas convergem para as probabilidades verdadeiras==. Entretanto, na prática, com dados finitos, ==precisamos introduzir **viés** nas estimativas para reduzir a variância e obter estimativas mais confiáveis [6].==

### Formulação Matemática

A estimativa de frequência relativa para uma sequência de tokens $w = (w_1, w_2, ..., w_m)$ é dada por [7]:

$$
p(w) = \frac{\text{count}(w)}{\text{count}(\text{todas as sequências})}
$$

Para um exemplo específico, considere a frase atribuída a Picasso [8]:

$$
p(\text{"Computers are useless, they can only give you answers"}) = \frac{\text{count}(\text{"Computers are useless, they can only give you answers"})}{\text{count}(\text{todas as frases já ditas})}
$$

Embora esta formulação seja matematicamente correta, apresenta desafios práticos significativos:

1. **Infinidade de Possíveis Sequências**: O denominador $\text{count}(\text{todas as sequências})$ teoricamente inclui todas as sequências possíveis, que são infinitas na linguagem natural devido à possibilidade de sequências de comprimento arbitrário [9].

2. **Esparsidade Extrema**: ==Mesmo ao limitar o comprimento das sequências a um valor máximo $M$, o número de possíveis sequências é exponencial em relação ao tamanho do vocabulário $V$, dado por $V^M$ [10].== Com um vocabulário de tamanho $V = 10^5$ e $M = 20$, temos $10^{100}$ sequências possíveis, tornando impraticável a observação de todas elas.

3. **Probabilidade Zero para Frases Não Observadas**: Sequências gramaticalmente corretas, mas não presentes no corpus de treinamento, recebem probabilidade zero, o que é problemático para a generalização do modelo [11].

### Limitações e Desafios

#### 👎 Desvantagens

1. **Alta Variância**: As estimativas são altamente instáveis para eventos raros devido às baixas contagens, levando a grande variabilidade nas probabilidades estimadas [12].

2. **Generalização Pobre**: ==O modelo falha em atribuir probabilidades significativas a sequências não observadas no treinamento==, limitando sua capacidade de lidar com novos dados [13].

3. **Ineficiência Computacional**: Requer armazenamento e processamento de um número exponencial de sequências, tornando o método inviável em termos práticos [14].

#### Análise Teórica

Para compreender melhor as limitações, consideremos um vocabulário de tamanho $V$ e um corpus de tamanho $M$. A probabilidade de observar uma sequência específica de comprimento $n$ é aproximadamente $(1/V)^n$, assumindo distribuição uniforme e independência entre as palavras. Para um vocabulário de $V = 10^5$ e uma sequência de $n = 20$ tokens, temos:

$$
P(\text{sequência específica}) \approx \left(\frac{1}{10^5}\right)^{20} = 10^{-100}
$$

Essa probabilidade extremamente baixa ilustra a impossibilidade prática de observar todas as sequências gramaticalmente válidas em qualquer corpus de treinamento realista [15].

### Soluções e Alternativas

Para mitigar os problemas associados à estimativa de frequência relativa pura, várias técnicas foram desenvolvidas que introduzem viés controlado nas estimativas para reduzir a variância e melhorar a generalização:

1. **Suavização (Smoothing)**: ==Técnicas como Laplace (add-one) e Lidstone adicionam pseudo-contagens para eventos não observados==, redistribuindo a massa de probabilidade de forma a evitar probabilidades zero [16].

2. **Backoff**: ==O modelo de backoff utiliza n-gramas de ordem inferior quando n-gramas de ordem superior não são observados==, permitindo aproveitar contextos menores para estimar probabilidades [17].

3. **Interpolação**: ==Combina estimativas de diferentes ordens de n-gramas através de uma média ponderada==, equilibrando o uso de contextos de tamanhos variados e reduzindo a variância das estimativas [18].

4. **Modelos Neurais de Linguagem**: Utilizam representações densas e contínuas para palavras e contextos, permitindo generalizar para sequências não observadas ao capturar semelhanças semânticas entre palavras [19].

### Respostas às Perguntas Teóricas

1. **Derive a expressão para a variância da estimativa de frequência relativa para um bigrama, assumindo um modelo de linguagem onde as palavras são geradas independentemente com probabilidades fixas.**

   **Solução:**

   Vamos considerar um vocabulário de tamanho $V$ e um corpus com $N$ palavras. As palavras são geradas independentemente com probabilidades fixas $\{p_1, p_2, ..., p_V\}$, onde $\sum_{i=1}^{V} p_i = 1$.

   **Definição do Bigrama:**

   Seja um bigrama específico $(w_i, w_j)$, onde $w_i$ e $w_j$ são palavras do vocabulário. A probabilidade teórica deste bigrama é:

   $$
   p_{ij} = p_i \cdot p_j
   $$

   **Estimativa de Frequência Relativa:**

   A estimativa de frequência relativa para $p_{ij}$ é dada por:

   $$
   \hat{p}_{ij} = \frac{C_{ij}}{N - 1}
   $$

   Onde:

   - $C_{ij}$ é o número de ocorrências do bigrama $(w_i, w_j)$ no corpus.
   - $N - 1$ é o número total de bigramas possíveis no corpus (porque cada bigrama envolve dois tokens consecutivos em uma sequência de $N$ palavras).

   **Distribuição de $C_{ij}$:**

   Como as palavras são geradas independentemente, a ocorrência de cada bigrama é independente, e $C_{ij}$ segue uma distribuição binomial:

   $$
   C_{ij} \sim \text{Binomial}(n = N - 1, p = p_i \cdot p_j)
   $$

   **Cálculo da Variância:**

   A variância da estimativa $\hat{p}_{ij}$ é dada por:

   $$
   \text{Var}(\hat{p}_{ij}) = \text{Var}\left( \frac{C_{ij}}{N - 1} \right) = \frac{1}{(N - 1)^2} \cdot \text{Var}(C_{ij})
   $$

   Sabendo que a variância de uma variável binomial é:

   $$
   \text{Var}(C_{ij}) = (N - 1) \cdot p_{ij} \cdot (1 - p_{ij})
   $$

   Substituindo na expressão da variância:

   $$
   \text{Var}(\hat{p}_{ij}) = \frac{1}{(N - 1)^2} \cdot (N - 1) \cdot p_{ij} \cdot (1 - p_{ij}) = \frac{p_{ij} \cdot (1 - p_{ij})}{N - 1}
   $$

   **Resposta Final:**

   A variância da estimativa de frequência relativa para o bigrama $(w_i, w_j)$ é:

   $$
   \text{Var}(\hat{p}_{ij}) = \frac{p_i \cdot p_j \cdot (1 - p_i \cdot p_j)}{N - 1}
   $$

   ==Isso mostra que a variância diminui com o aumento do tamanho do corpus $(N)$ e depende das probabilidades individuais das palavras que compõem o bigrama.==

---

2. **Prove que, para qualquer corpus finito, existe um comprimento de sequência $n$ tal que a probabilidade de observar qualquer sequência específica de comprimento $n$ é menor que $\epsilon$, para qualquer $\epsilon > 0$ escolhido arbitrariamente.**

   **Solução:**

   **Hipóteses:**

   - O corpus é finito e tem tamanho $N$ (número total de tokens).
   - O vocabulário tem tamanho finito $V$.
   - As palavras têm probabilidades positivas e somam 1.

   **Probabilidade de uma Sequência Específica:**

   Em um modelo de linguagem onde as palavras são geradas independentemente, a probabilidade de uma sequência específica de comprimento $n$ é:

   $$
   p_{\text{sequência}} = \prod_{k=1}^{n} p_{w_k}
   $$

   Onde $p_{w_k}$ é a probabilidade da palavra na posição $k$.

   **Estimativa do Limite Superior da Probabilidade:**

   - Seja $p_{\text{max}} = \max_{i} p_i$ a maior probabilidade dentre todas as palavras do vocabulário.
   - Então, a probabilidade máxima de qualquer sequência de comprimento $n$ é:

     $$
     p_{\text{sequência máxima}} = (p_{\text{max}})^n
     $$

   **Prova:**

   Queremos mostrar que para qualquer $\epsilon > 0$, existe um $n$ tal que:

   $$
   p_{\text{sequência máxima}} = (p_{\text{max}})^n < \epsilon
   $$

   **Tomando Logaritmo:**

   - Aplicamos o logaritmo natural em ambos os lados:

     $$
     \ln(p_{\text{max}}^n) = n \ln(p_{\text{max}}) < \ln(\epsilon)
     $$

   - Como $0 < p_{\text{max}} < 1$, temos que $\ln(p_{\text{max}}) < 0$.

   **Resolvendo para $n$:**

   $$
   n > \frac{\ln(\epsilon)}{\ln(p_{\text{max}})}
   $$

   - Como $\ln(p_{\text{max}}) < 0$, o quociente $\frac{\ln(\epsilon)}{\ln(p_{\text{max}})}$ é positivo para $\epsilon < 1$.

   **Conclusão:**

   - Para qualquer $\epsilon > 0$, podemos escolher:

     $$
     n = \left\lceil \frac{\ln(\epsilon)}{\ln(p_{\text{max}})} \right\rceil
     $$

     Onde $\lceil x \rceil$ denota o menor inteiro maior ou igual a $x$.

   - Portanto, existe um comprimento $n$ tal que a probabilidade de qualquer sequência específica de comprimento $n$ é menor que $\epsilon$.

---

3. **Analise teoricamente como o tamanho do vocabulário $V$ afeta a taxa de convergência da estimativa de frequência relativa para a verdadeira distribuição de probabilidade, assumindo um modelo de linguagem simples.**

   **Solução:**

   **Contexto:**

   - Em estimativas de frequência relativa, estamos interessados em como a estimativa $\hat{p}_i$ para a probabilidade verdadeira $p_i$ converge à medida que aumentamos o tamanho do corpus $N$.
   - A taxa de convergência é influenciada pelo número de parâmetros a serem estimados (tamanho do vocabulário $V$) e pelo número de observações disponíveis para cada parâmetro.

   **Variância da Estimativa Unigrama:**

   - Para uma palavra $w_i$, a estimativa de frequência relativa é:

     $$
     \hat{p}_i = \frac{C_i}{N}
     $$

     Onde $C_i$ é o número de ocorrências de $w_i$ no corpus.

   - A variância da estimativa é:

     $$
     \text{Var}(\hat{p}_i) = \frac{p_i (1 - p_i)}{N}
     $$

   **Impacto do Tamanho do Vocabulário:**

   - À medida que $V$ aumenta, para um corpus de tamanho fixo $N$, o número médio de observações por palavra diminui.
   - Assumindo distribuição uniforme para simplificar ($p_i = 1/V$):

     - Esperança de $C_i$:

       $$
       E[C_i] = N \cdot \frac{1}{V}
       $$

     - Variância da estimativa:

       $$
       \text{Var}(\hat{p}_i) = \frac{(1/V) (1 - 1/V)}{N}
       $$

   - Com o aumento de $V$, $E[C_i]$ diminui, levando a estimativas menos confiáveis.

   **Taxa de Convergência:**

   - A taxa de convergência da estimativa para a verdadeira probabilidade é proporcional a $1/\sqrt{N}$ para cada parâmetro.
   - Porém, o número total de parâmetros é $V$, então o erro total pode ser visto como:

     $$
     \text{Erro Total} \propto V \cdot \frac{1}{\sqrt{N}}
     $$

   - Para manter o mesmo nível de erro ao aumentar $V$, seria necessário aumentar $N$ proporcionalmente.

   **Conclusão:**

   - ==**Quanto maior o vocabulário $V$, mais dados são necessários para que as estimativas de frequência relativa convergem para as verdadeiras probabilidades.**==
   - O aumento em $V$ leva a um aumento no número de parâmetros a serem estimados, mas o número de observações por parâmetro diminui, retardando a taxa de convergência.
   - **Implicação Prática:** Em modelos com vocabulários extensos, técnicas adicionais como suavização ou modelos que compartilham parâmetros (e.g., modelos neurais) são necessárias para obter estimativas confiáveis com conjuntos de dados de tamanho razoável.

### Conclusão

A estimativa de frequência relativa, embora seja um estimador de máxima verossimilhança e conceitualmente atraente, apresenta limitações fundamentais quando aplicada a modelos de linguagem realistas. A esparsidade de dados e a alta variância das estimativas comprometem a capacidade do modelo de generalizar para sequências não observadas, um requisito essencial em tarefas de processamento de linguagem natural. Essas limitações motivaram o desenvolvimento de técnicas alternativas que introduzem viés controlado nas estimativas para reduzir a variância, como a suavização, o backoff e a interpolação, bem como a adoção de modelos neurais de linguagem que exploram representações contínuas e contextos mais amplos [20]. Compreender essas limitações e as soluções propostas é crucial para o avanço e aplicação eficaz de modelos de linguagem em diversos domínios.

### Perguntas Teóricas Avançadas

1. **Derive a expressão para o viés e a variância da estimativa de frequência relativa para um n-grama genérico, em função do tamanho do corpus e da ordem do n-grama. Compare analiticamente o trade-off entre viés e variância para diferentes ordens de n-gramas.**

2. **Considerando um modelo de linguagem baseado em frequência relativa, prove que a perplexidade no conjunto de treinamento é sempre menor ou igual à perplexidade em um conjunto de teste não visto, e identifique as condições sob as quais a igualdade é alcançada.**

3. **Desenvolva uma prova formal mostrando que, para qualquer distribuição de probabilidade de linguagem natural, existe um tamanho de corpus finito a partir do qual a estimativa de frequência relativa de n-gramas supera em desempenho (em termos de perplexidade esperada) qualquer modelo de linguagem baseado em unigramas, independentemente do método de suavização usado.**

4. **Analise teoricamente o impacto da Lei de Zipf na eficácia da estimativa de frequência relativa para modelagem de linguagem. Derive uma expressão para o número esperado de n-gramas únicos em função do tamanho do corpus, assumindo que a frequência das palavras segue exatamente a Lei de Zipf.**

5. **Prove que, para qualquer método de suavização que distribui uma massa de probabilidade fixa para eventos não observados, existe uma sequência de palavras gramaticalmente correta cuja probabilidade sob o modelo suavizado é menor do que sob a estimativa de frequência relativa pura. Discuta as implicações deste resultado para o design de técnicas de suavização.**

### Referências

[1] "A simple approach to computing the probability of a sequence of tokens is to use a relative frequency estimate." *(Trecho de Language Models_143-162.pdf.md)*

[2] "Clearly, this estimator is very data-hungry, and suffers from high variance: even grammatical sentences will have probability zero if they have not occurred in the training data." *(Trecho de Language Models_143-162.pdf.md)*

[3] "One way to estimate the probability of this sentence is," *(Trecho de Language Models_143-162.pdf.md)*

[4] "But in practice, we are asking for accurate counts over an infinite number of events, since sequences of words can be arbitrarily long." *(Trecho de Language Models_143-162.pdf.md)*

[5] "This estimator is unbiased: in the theoretical limit of infinite data, the estimate will be correct. But in practice, we are asking for accurate counts over an infinite number of events, since sequences of words can be arbitrarily long." *(Trecho de Language Models_143-162.pdf.md)*

[6] "We therefore need to introduce bias to have a chance of making reliable estimates from finite training data." *(Trecho de Language Models_143-162.pdf.md)*

[7] "p(Computers are useless, they can only give you answers) = count(Computers are useless, they can only give you answers) / count(all sentences ever spoken)" *(Trecho de Language Models_143-162.pdf.md)*

[8] "Consider the quote, attributed to Picasso, 'computers are useless, they can only give you answers.'" *(Trecho de Language Models_143-162.pdf.md)*

[9] "But in practice, we are asking for accurate counts over an infinite number of events, since sequences of words can be arbitrarily long." *(Trecho de Language Models_143-162.pdf.md)*

[10] "Even with an aggressive upper bound of, say, M = 20 tokens in the sequence, the number of possible sequences is V^20, where V = |V|. A small vocabulary for English would have V = 10^5, so there are 10^100 possible sequences." *(Trecho de Language Models_143-162.pdf.md)*

[11] "Clearly, this estimator is very data-hungry, and suffers from high variance: even grammatical sentences will have probability zero if they have not occurred in the training data." *(Trecho de Language Models_143-162.pdf.md)*

[12] "This estimator is unbiased: in the theoretical limit of infinite data, the estimate will be correct. But in practice, we are asking for accurate counts over an infinite number of events, since sequences of words can be arbitrarily long." *(Trecho de Language Models_143-162.pdf.md)*

[13] "We therefore need to introduce bias to have a chance of making reliable estimates from finite training data." *(Trecho de Language Models_143-162.pdf.md)*

[14] "Even with an aggressive upper bound of, say, M = 20 tokens in the sequence, the number of possible sequences is V^20, where V = |V|. A small vocabulary for English would have V = 10^5, so there are 10^100 possible sequences." *(Trecho de Language Models_143-162.pdf.md)*

[15] "A small vocabulary for English would have V = 10^5, so there are 10^100 possible sequences." *(Trecho de Language Models_143-162.pdf.md)*

[16] "The language models that follow in this chapter introduce bias in various ways." *(Trecho de Language Models_143-162.pdf.md)*

[17] "We begin with n-gram language models, which compute the probability of a sequence as the product of probabilities of subsequences." *(Trecho de Language Models_143-162.pdf.md)*

[18] "Instead of choosing a single n for the size of the n-gram, we can take the weighted average across several n-gram probabilities." *(Trecho de Language Models_143-162.pdf.md)*

[19] "N-gram language models have been largely supplanted by neural networks. These models do not make the n-gram assumption of restricted context; indeed, they can incorporate arbitrarily distant contextual information, while remaining computationally and statistically tractable." *(Trecho de Language Models_143-162.pdf.md)*

[20] "The language models that follow in this chapter introduce bias in various ways." *(Trecho de Language Models_143-162.pdf.md)*