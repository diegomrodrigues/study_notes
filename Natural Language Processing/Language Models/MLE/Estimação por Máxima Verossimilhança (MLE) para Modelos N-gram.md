# Estimação por Máxima Verossimilhança (MLE) para Modelos N-gram

<imagem: Um gráfico mostrando a distribuição de probabilidades de n-gramas estimadas por MLE, com curvas para diferentes ordens de n (unigrama, bigrama, trigrama) sobre um corpus de exemplo.>

## Introdução

A **Estimação por Máxima Verossimilhança (Maximum Likelihood Estimation - MLE)** é um método estatístico fundamental utilizado para estimar os parâmetros de modelos probabilísticos, maximizando a probabilidade dos dados observados [1]. No contexto de modelos de linguagem n-gram, a MLE permite estimar as probabilidades de sequências de palavras com base em suas frequências relativas em um corpus de treinamento. Este método assume que as frequências observadas são indicativas das probabilidades verdadeiras das sequências na linguagem.

Os modelos n-gram são uma classe essencial de modelos de linguagem que se baseiam na **hipótese de Markov**: a probabilidade de uma palavra depende apenas das n-1 palavras anteriores [2]. Essa suposição simplifica o problema de modelagem de linguagem ao reduzir as dependências de longo alcance para dependências locais, permitindo a construção de modelos eficientes para diversas aplicações, como reconhecimento de fala, tradução automática e sistemas de perguntas e respostas [3].

Apesar de sua simplicidade, a MLE aplicada a modelos n-gram enfrenta desafios significativos, especialmente relacionados à esparsidade de dados e ao balanceamento entre viés e variância nas estimativas. Este resumo explora em profundidade como a MLE é aplicada na estimação de probabilidades de n-gramas, discute suas vantagens e limitações, e apresenta técnicas avançadas para mitigar os desafios inerentes a esse método.

## Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **N-gram**                 | Um n-gram é uma sequência contígua de n itens (geralmente palavras) em um dado texto ou corpus. Por exemplo, um bigrama (n=2) considera pares de palavras adjacentes, enquanto um trigrama (n=3) considera trios de palavras [4]. |
| **Máxima Verossimilhança** | A estimação por máxima verossimilhança é um método para estimar os parâmetros de um modelo estatístico, maximizando a função de verossimilhança, que representa a probabilidade dos dados observados dado o modelo [5]. |
| **Frequência Relativa**    | No contexto de n-gramas, a frequência relativa é a contagem de ocorrências de um n-gram específico dividida pelo número total de n-gramas no corpus de treinamento [6]. |

> ⚠️ **Nota Importante**: A MLE para n-gramas assume que a frequência relativa de uma sequência no corpus de treinamento é uma boa estimativa de sua verdadeira probabilidade na linguagem. Esta suposição pode levar a problemas com n-gramas não observados no treinamento, atribuindo-lhes probabilidade zero [7].

## Estimação de Probabilidades N-gram por MLE

==A estimação de probabilidades n-gram por MLE baseia-se na contagem de ocorrências de sequências de palavras no corpus de treinamento==. Este processo envolve a ==aplicação direta do princípio de máxima verossimilhança para calcular as probabilidades condicionais de palavras dadas os seus contextos anteriores.==

### Formalização Matemática

Para justificar a fórmula de estimação de probabilidades n-gram por MLE, seguiremos um processo detalhado, passo a passo, fundamentado nos princípios da máxima verossimilhança.

#### 1. Definição do Problema

Queremos estimar as probabilidades condicionais $P(w_n \mid w_1, w_2, \dots, w_{n-1})$ para todas as possíveis sequências de n-gramas em um corpus, onde cada $w_i$ pertence a um vocabulário finito de palavras.

#### 2. Construção da Função de Verossimilhança

Considere um corpus de tamanho $N$, representado por uma sequência de palavras $W = (w_1, w_2, \dots, w_N)$. ==Supondo que o corpus é gerado independentemente segundo o modelo n-gram (com a hipótese de Markov de ordem $n-1$)==, a função de verossimilhança $L$ dos parâmetros (as probabilidades condicionais) dados os dados observados é:

$$
L = \prod_{i=1}^{N} P(w_i \mid w_{i-(n-1)}, \dots, w_{i-1})
$$

Esta expressão ==representa o produto das probabilidades de cada palavra $w_i$ dada o seu contexto anterior de $n-1$ palavras.==

#### 3. Reescrita da Verossimilhança em Termos de Contagens

As sequências de n-gramas podem ocorrer múltiplas vezes no corpus. Seja $C(w_{i-(n-1)}, \dots, w_i)$ a contagem de ocorrências do n-gram $(w_{i-(n-1)}, \dots, w_i)$. ==A função de verossimilhança pode ser reescrita agrupando termos idênticos:==

$$
L = \prod_{\text{todos os } n\text{-gramas}} [P(w_n \mid w_1, w_2, \dots, w_{n-1})]^{C(w_1, w_2, \dots, w_n)}
$$

#### 4. Maximização da Verossimilhança

Para facilitar a maximização, utilizamos o logaritmo natural da verossimilhança (log-verossimilhança):

$$
\ln L = \sum_{\text{todos os } n\text{-gramas}} C(w_1, w_2, \dots, w_n) \ln P(w_n \mid w_1, w_2, \dots, w_{n-1})
$$

==Nosso objetivo é encontrar as probabilidades $P(w_n \mid w_1, w_2, \dots, w_{n-1})$ que maximizam $\ln L$, sujeitas às restrições de que, para cada contexto $(w_1, w_2, \dots, w_{n-1})$,== as probabilidades condicionais devem somar 1:
$$
\sum_{w_n} P(w_n \mid w_1, w_2, \dots, w_{n-1}) = 1 \quad \forall \; (w_1, w_2, \dots, w_{n-1})
$$

#### 5. Aplicação dos Multiplicadores de Lagrange

==Para incorporar as restrições de normalização, aplicamos o método dos multiplicadores de Lagrange==. Definimos a função Lagrangiana $\mathcal{L}$:
$$
\mathcal{L} = \ln L - \sum_{\text{contextos } (w_1, w_2, \dots, w_{n-1})} \lambda_{w_1, w_2, \dots, w_{n-1}} \left( \sum_{w_n} P(w_n \mid w_1, w_2, \dots, w_{n-1}) - 1 \right)
$$

Onde $\lambda_{w_1, w_2, \dots, w_{n-1}}$ são os multiplicadores de Lagrange correspondentes a cada contexto.

#### 6. Derivação das Equações Normais

Calculamos a derivada parcial da Lagrangiana em relação a cada probabilidade $P(w_n \mid w_1, w_2, \dots, w_{n-1})$ e igualamos a zero:

$$
\frac{\partial \mathcal{L}}{\partial P(w_n \mid w_1, w_2, \dots, w_{n-1})} = \frac{C(w_1, w_2, \dots, w_n)}{P(w_n \mid w_1, w_2, \dots, w_{n-1})} - \lambda_{w_1, w_2, \dots, w_{n-1}} = 0
$$

Rearranjando a equação, obtemos:

$$
P(w_n \mid w_1, w_2, \dots, w_{n-1}) = \frac{C(w_1, w_2, \dots, w_n)}{\lambda_{w_1, w_2, \dots, w_{n-1}}}
$$

#### 7. Determinação dos Multiplicadores de Lagrange

Usamos a restrição de normalização para resolver $\lambda_{w_1, w_2, \dots, w_{n-1}}$:

$$
\sum_{w_n} P(w_n \mid w_1, w_2, \dots, w_{n-1}) = \sum_{w_n} \frac{C(w_1, w_2, \dots, w_n)}{\lambda_{w_1, w_2, \dots, w_{n-1}}} = 1
$$

Simplificando:

$$
\frac{1}{\lambda_{w_1, w_2, \dots, w_{n-1}}} \sum_{w_n} C(w_1, w_2, \dots, w_n) = 1
$$

Observamos que:

$$
\sum_{w_n} C(w_1, w_2, \dots, w_n) = C(w_1, w_2, \dots, w_{n-1})
$$

Que é a contagem total do contexto $(w_1, w_2, \dots, w_{n-1})$ no corpus.

Portanto:

$$
\frac{C(w_1, w_2, \dots, w_{n-1})}{\lambda_{w_1, w_2, \dots, w_{n-1}}} = 1 \implies \lambda_{w_1, w_2, \dots, w_{n-1}} = C(w_1, w_2, \dots, w_{n-1})
$$

#### 8. Substituição e Obtenção da Fórmula Final

Substituindo o valor de $\lambda_{w_1, w_2, \dots, w_{n-1}}$ na expressão para $P(w_n \mid w_1, w_2, \dots, w_{n-1})$, temos:

$$
P(w_n \mid w_1, w_2, \dots, w_{n-1}) = \frac{C(w_1, w_2, \dots, w_n)}{C(w_1, w_2, \dots, w_{n-1})}
$$

==Esta é a fórmula da estimação por máxima verossimilhança para probabilidades condicionais em modelos n-gram.==

#### 9. Interpretação Intuitiva

A fórmula indica que a probabilidade condicional de uma palavra $w_n$ dado o contexto $(w_1, w_2, \dots, w_{n-1})$ é estimada pela frequência relativa com que o n-gram completo $(w_1, w_2, \dots, w_n)$ ocorre no corpus em relação à frequência do contexto $(w_1, w_2, \dots, w_{n-1})$. Em outras palavras, estamos utilizando as ocorrências observadas no corpus para estimar as probabilidades de transição entre contextos e palavras subsequentes.

#### 10. Caso Especial: Bigramas

Para o caso de bigramas (n=2), o contexto é uma única palavra $w_{n-1}$, e a fórmula se simplifica para:

$$
P(w_n \mid w_{n-1}) = \frac{C(w_{n-1}, w_n)}{C(w_{n-1})}
$$

Onde:

- $C(w_{n-1}, w_n)$ é a contagem de ocorrências do bigrama específico no corpus.
- $C(w_{n-1})$ é a contagem total de ocorrências da palavra $w_{n-1}$ como contexto.

### Considerações Finais

A derivação detalhada acima demonstra que a estimação por máxima verossimilhança em modelos n-gram resulta em probabilidades condicionais que refletem diretamente as frequências relativas observadas no corpus de treinamento. Este método é intuitivo e baseado em dados, mas enfrenta desafios significativos quando aplicado a situações de esparsidade de dados. N-gramas que não ocorrem no corpus recebem probabilidade zero, o que pode levar a problemas como atribuir probabilidade zero a sequências válidas em dados de teste.

Compreender essa derivação é fundamental para apreciar as vantagens e limitações da MLE em modelos n-gram e para motivar o desenvolvimento de técnicas de suavização e métodos alternativos que buscam melhorar a estimativa de probabilidades em contextos de dados limitados.

---

### Referências

[1] **Jurafsky, D., & Martin, J. H.** (2009). *Speech and Language Processing*. Prentice Hall. (Capítulo sobre Modelos de Linguagem e n-gramas)

[2] **Manning, C. D., & Schütze, H.** (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.

[3] **Chen, S. F., & Goodman, J.** (1996). *An Empirical Study of Smoothing Techniques for Language Modeling*. Proceedings of the 34th Annual Meeting on Association for Computational Linguistics.

[4] **Kneser, R., & Ney, H.** (1995). *Improved Backing-Off for M-gram Language Modeling*. In *ICASSP*.

[5] **Jelinek, F.** (1997). *Statistical Methods for Speech Recognition*. MIT Press.

[6] **Church, K. W., & Gale, W. A.** (1991). *A Comparison of the Enhanced Good-Turing and Deleted Estimation Methods for Estimating Probabilities of English Bigrams*. *Computer Speech & Language*, 5(1), 19-54.

[7] **Chen, S. F., & Goodman, J.** (1999). *An Empirical Study of Smoothing Techniques for Language Modeling*. *Computer Speech & Language*, 13(4), 359-394.

[8] **Bahl, L. R., Jelinek, F., & Mercer, R. L.** (1983). *A Maximum Likelihood Approach to Continuous Speech Recognition*. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, (2), 179-190.

[9] **Goodman, J.** (2001). *A Bit of Progress in Language Modeling*. *Computer Speech & Language*, 15(4), 403-434.

[10] **Katz, S. M.** (1987). *Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer*. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3), 400-401.

[11] **Teoria da Informação e Codificação** (Notas de aula).