# Classificação de Texto com Multinomial Naïve Bayes

<imagem: Um diagrama mostrando um documento de texto sendo transformado em um vetor de contagem de palavras (bag-of-words), que então alimenta um modelo Naïve Bayes com distribuições multinomiais para cada classe, resultando em uma previsão de classe.>

## Introdução

O **Multinomial Naïve Bayes** é um algoritmo fundamental na classificação de texto, particularmente eficaz quando utilizamos a representação bag-of-words. Este método combina o princípio de Naïve Bayes com a distribuição multinomial para modelar a ocorrência de palavras em documentos, assumindo a independência condicional das palavras dado o rótulo do documento [1]. Esta abordagem é amplamente utilizada devido à sua simplicidade, eficiência computacional e surpreendente eficácia em muitas tarefas de classificação de texto, como filtragem de spam, categorização de notícias e análise de sentimentos.

## Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Bag-of-Words**              | Representação de um documento como um vetor de contagem de palavras, ignorando a ordem e a estrutura gramatical [2]. |
| **Distribuição Multinomial**  | Modelo probabilístico que descreve a ocorrência de eventos discretos em um número fixo de tentativas, adequado para modelar a frequência de palavras em documentos [3]. |
| **Independência Condicional** | Suposição de que as ocorrências de palavras são independentes entre si, dado o rótulo do documento [4]. |

> ⚠️ **Nota Importante**: A suposição de independência condicional, embora simplificadora, é crucial para a tratabilidade computacional do modelo Naïve Bayes, permitindo uma fatorização eficiente da probabilidade conjunta [5].

### Formulação Matemática do Multinomial Naïve Bayes

O Multinomial Naïve Bayes modela a probabilidade de um documento pertencer a uma classe específica usando a regra de Bayes e a distribuição multinomial. A probabilidade de um documento $x$ pertencer à classe $y$ é dada por [6]:

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$

Onde:
- $p(y|x)$ é a probabilidade posterior da classe $y$ dado o documento $x$
- $p(x|y)$ é a verossimilhança do documento $x$ dado a classe $y$
- $p(y)$ é a probabilidade a priori da classe $y$
- $p(x)$ é a probabilidade marginal do documento $x$

A verossimilhança $p(x|y)$ é modelada usando a distribuição multinomial [7]:

$$
p_{\text{mult}}(x; \phi_y) = B(x) \prod_{j=1}^V \phi_{y,j}^{x_j}
$$

Onde:
- $\phi_y$ é o vetor de parâmetros da distribuição multinomial para a classe $y$
- $x_j$ é a contagem da palavra $j$ no documento
- $V$ é o tamanho do vocabulário
- $B(x)$ é o coeficiente multinomial (que pode ser ignorado na classificação)

### Estimação de Parâmetros

Os parâmetros $\phi_{y,j}$ são estimados usando a estimativa de máxima verossimilhança (MLE) [8]:

$$
\phi_{y,j} = \frac{\text{count}(y,j)}{\sum_{j'=1}^V \text{count}(y,j')} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{\sum_{j'=1}^V \sum_{i:y^{(i)}=y} x_{j'}^{(i)}}
$$

Onde $\text{count}(y,j)$ é a contagem total da palavra $j$ em documentos da classe $y$.

> ✔️ **Destaque**: A estimativa MLE pode levar a problemas com palavras não vistas no conjunto de treinamento. Para mitigar isso, é comum usar suavização de Laplace (add-one smoothing) [9].

### Suavização de Laplace

A suavização de Laplace adiciona um pseudocontagem $\alpha$ a cada contagem de palavra [10]:

$$
\phi_{y,j} = \frac{\alpha + \text{count}(y,j)}{V\alpha + \sum_{j'=1}^V \text{count}(y,j')}
$$

Esta técnica evita probabilidades zero e melhora a generalização do modelo.

#### Perguntas Teóricas

1. Derive a estimativa de máxima verossimilhança para o parâmetro $\mu$ no modelo Naïve Bayes, considerando a distribuição multinomial.

2. Demonstre matematicamente por que a suavização de Laplace é necessária e como ela afeta a estimativa dos parâmetros $\phi_{y,j}$.

3. Analise teoricamente o impacto da suposição de independência condicional na performance do Multinomial Naïve Bayes em comparação com modelos que não fazem essa suposição.

## Implementação e Algoritmo

A implementação do Multinomial Naïve Bayes envolve duas fases principais: treinamento e predição.

### Fase de Treinamento

1. Calcular as probabilidades a priori $p(y)$ para cada classe.
2. Estimar os parâmetros $\phi_{y,j}$ para cada classe e palavra no vocabulário.

### Fase de Predição

Para um novo documento $x$, calcular:

$$
\hat{y} = \arg\max_y p(y) \prod_{j=1}^V \phi_{y,j}^{x_j}
$$

Na prática, é comum trabalhar com logaritmos para evitar underflow numérico [11]:

$$
\hat{y} = \arg\max_y \left(\log p(y) + \sum_{j=1}^V x_j \log \phi_{y,j}\right)
$$

```python
import numpy as np
from scipy.sparse import csr_matrix

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X: csr_matrix, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        self.class_log_prior_ = np.log(np.bincount(y) / n_samples)
        
        # Calcular contagens de palavras por classe
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            N_c = X_c.sum(axis=0).A1 + self.alpha
            self.feature_log_prob_[i, :] = np.log(N_c / N_c.sum())
    
    def predict(self, X: csr_matrix):
        return self.classes[np.argmax(self._joint_log_likelihood(X), axis=1)]
    
    def _joint_log_likelihood(self, X: csr_matrix):
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)

def safe_sparse_dot(a, b):
    if isinstance(a, csr_matrix):
        return a.dot(b)
    return np.dot(a, b)
```

Este código implementa o Multinomial Naïve Bayes de forma eficiente, utilizando matrizes esparsas para lidar com grandes conjuntos de dados de texto [12].

> 💡 **Dica**: A implementação usa logaritmos para evitar underflow numérico e melhorar a estabilidade computacional.

## Vantagens e Desvantagens

| 👍 Vantagens                                          | 👎 Desvantagens                                               |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| Eficiente em termos computacionais e de memória [13] | Suposição de independência condicional pode ser irrealista [14] |
| Funciona bem com conjuntos de dados pequenos [15]    | Sensível a características irrelevantes ou correlacionadas [16] |
| Naturalmente multiclasse                             | Pode ser superado por modelos mais complexos em grandes conjuntos de dados [17] |
| Fácil de implementar e interpretar                   | Estimativas de probabilidade podem ser imprecisas [18]       |

## Análise Teórica Avançada

### Relação com Modelos Log-lineares

O Multinomial Naïve Bayes pode ser visto como um caso especial de um modelo log-linear [19]. Considerando o logaritmo da probabilidade condicional:

$$
\log p(y|x; \theta) = \theta \cdot f(x,y) - \log \sum_{y' \in Y} \exp(\theta \cdot f(x,y'))
$$

Onde $f(x,y)$ é uma função de características que retorna um vetor de contagens de palavras para a classe $y$, e $\theta$ são os pesos do modelo. Esta formulação mostra a conexão entre Naïve Bayes e modelos discriminativos como regressão logística.

### Análise de Complexidade

A complexidade de tempo de treinamento do Multinomial Naïve Bayes é $O(ND)$, onde $N$ é o número de documentos de treinamento e $D$ é o número médio de palavras únicas por documento. A complexidade de espaço é $O(CV)$, onde $C$ é o número de classes e $V$ é o tamanho do vocabulário [20].

### Teorema de Bayes e Máxima Verossimilhança

A estimação dos parâmetros $\phi_{y,j}$ pode ser justificada como uma estimativa de máxima verossimilhança. A log-verossimilhança é dada por [21]:

$$
\mathcal{L}(\phi, \mu) = \sum_{i=1}^N \log p_{\text{mult}}(x^{(i)}; \phi_{y^{(i)}}) + \log p_{\text{cat}}(y^{(i)}; \mu)
$$

Maximizar esta função leva às estimativas de frequência relativa para $\phi_{y,j}$ e $\mu_y$.

#### Perguntas Teóricas

1. Prove que a estimativa de máxima verossimilhança para $\phi_{y,j}$ no Multinomial Naïve Bayes é equivalente à frequência relativa das palavras na classe.

2. Analise teoricamente o impacto da suavização de Laplace na variância dos estimadores de $\phi_{y,j}$. Como isso afeta o viés-variância do modelo?

3. Derive a forma fechada da estimativa de máxima a posteriori (MAP) para $\phi_{y,j}$ assumindo uma distribuição a priori Dirichlet sobre os parâmetros.

## Conclusão

O Multinomial Naïve Bayes é um classificador poderoso e eficiente para tarefas de classificação de texto, especialmente quando usado com a representação bag-of-words. Sua simplicidade matemática, eficiência computacional e surpreendente eficácia em muitas aplicações práticas o tornam uma escolha popular em aprendizado de máquina e processamento de linguagem natural [22].

Apesar de suas limitações, como a suposição de independência condicional, o Multinomial Naïve Bayes continua sendo um benchmark importante e uma ferramenta valiosa no arsenal de qualquer cientista de dados, especialmente para tarefas de classificação de texto em larga escala ou quando os recursos computacionais são limitados [23].

## Perguntas Teóricas Avançadas

1. Derive a forma da fronteira de decisão entre duas classes no espaço de características para o Multinomial Naïve Bayes. Como essa fronteira se compara com a de um classificador de regressão logística?

2. Analise teoricamente o comportamento assintótico do Multinomial Naïve Bayes à medida que o número de amostras de treinamento tende ao infinito. Sob quais condições o classificador converge para o classificador de Bayes ótimo?

3. Desenvolva uma prova formal da consistência do estimador de máxima verossimilhança para os parâmetros do Multinomial Naïve Bayes, assumindo que os dados são gerados pelo modelo verdadeiro.

4. Derive a expressão para a informação mútua entre as características e o rótulo da classe no contexto do Multinomial Naïve Bayes. Como isso se relaciona com a capacidade preditiva do modelo?

5. Formule e prove um teorema que estabeleça limites superiores no erro de generalização do Multinomial Naïve Bayes em termos do número de amostras de treinamento e da dimensionalidade do espaço de características.

## Referências

[1] "To predict a label from a bag-of-words, we can assign a score to each word in the vocabulary, measuring the compatibility with the label." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[2] "Suppose that x is a bag-of-words vector such that ∑ᵥⱼ₌₁ xⱼ = 1." *(Trecho de Exercises)*

[3] "The multinomial distribution involves a product over words, with each term in the product equal to the probability φj, exponentiated by the count xj." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[4] "Because the multinomial distribution treats each word token independently, conditioned on the class: the probability mass function factorizes across the counts." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[5] "The "naïve" independence assumption implied by the multinomial distribution, and as a result, the optimal parameters for this model are identical to those in multinomial Naïve Bayes." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[6] "p(y | x; θ) = exp (θ · f(x, y)) / ∑y'∈Y exp (θ · f(x, y'))." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[7] "p_mult(x; φ) = B(x) ∏(j=1 to V) φ(x_j)_(y,j)" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[8] "φ_{y,j} = (count(y, j)) / (∑_{j'=1}^V count(y, j')) = (∑_{i:y^{(i)}=y} x_j^{(i)}) / (∑_{j'=1}^V ∑_{i:y^{(i)}=y} x_{j'}^{(i)})" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[9] "One solution is to smooth the probabilities, by adding a "pseudocount" of α to each count, and then normalizing" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[10] "φ_y,j = (α + count(y, j)) / (Vα + ∑^V_j'=1 count(y, j'))" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[11] "log p(x | y; φ) + log p(y; μ) = log [B(x) ∏(j=1 to V) φ(x_j)_(y,j)] + log μ_y" *(Trecho de CHAPTER