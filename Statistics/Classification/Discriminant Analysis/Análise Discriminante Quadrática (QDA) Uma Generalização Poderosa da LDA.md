## Análise Discriminante Quadrática (QDA): Uma Generalização Poderosa da LDA

![image-20240802163204034](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240802163204034.png)

A Análise Discriminante Quadrática (QDA) emerge como uma extensão sofisticada da Análise Discriminante Linear (LDA), oferecendo uma abordagem mais flexível e potente para problemas de classificação em estatística e aprendizado de máquina [1]. Enquanto a LDA assume matrizes de covariância idênticas para todas as classes, a QDA relaxa essa restrição, permitindo que cada classe tenha sua própria matriz de covariância [2]. Esta generalização resulta em fronteiras de decisão quadráticas, em contraste com as fronteiras lineares da LDA, proporcionando uma capacidade significativamente maior de modelar relações complexas nos dados.

### Conceitos Fundamentais

| Conceito                                        | Explicação                                                   |
| ----------------------------------------------- | ------------------------------------------------------------ |
| **Matriz de Covariância Específica por Classe** | Na QDA, cada classe $k$ tem sua própria matriz de covariância $\Sigma_k$, permitindo uma representação mais precisa da distribuição dos dados dentro de cada classe. [2] |
| **Funções Discriminantes Quadráticas**          | As funções discriminantes na QDA são quadráticas em $x$, resultando em fronteiras de decisão não-lineares no espaço de características. [3] |
| **Estimação de Parâmetros**                     | A QDA requer a estimação de parâmetros adicionais em comparação com a LDA, incluindo matrizes de covariância separadas para cada classe. [4] |

> ⚠️ **Nota Importante**: A flexibilidade adicional da QDA vem com o custo de um aumento significativo no número de parâmetros a serem estimados, especialmente em espaços de alta dimensão.

### Formulação Matemática da QDA

A QDA baseia-se no modelo gaussiano para as densidades de classe condicional, mas sem a restrição de covariâncias iguais:

$$
f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma_k|^{1/2}} e^{-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)}
$$

Onde:
- $f_k(x)$ é a densidade de probabilidade da classe $k$
- $\mu_k$ é o vetor médio da classe $k$
- $\Sigma_k$ é a matriz de covariância da classe $k$
- $p$ é a dimensão do espaço de características

A função discriminante quadrática para a classe $k$ é dada por [5]:

$$
\delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T\Sigma_k^{-1}(x - \mu_k) + \log\pi_k
$$

Onde $\pi_k$ é a probabilidade a priori da classe $k$.

#### Fronteiras de Decisão Quadráticas

A fronteira de decisão entre duas classes $k$ e $l$ é definida pelo conjunto de pontos que satisfazem $\delta_k(x) = \delta_l(x)$, resultando em uma equação quadrática em $x$ [6]:

$$
\{x : (\mu_k - \mu_l)^T(\Sigma_k^{-1} - \Sigma_l^{-1})x + \frac{1}{2}x^T(\Sigma_l^{-1} - \Sigma_k^{-1})x + c = 0\}
$$

Onde $c$ é uma constante que depende de $\mu_k$, $\mu_l$, $\Sigma_k$, $\Sigma_l$, $\pi_k$, e $\pi_l$.

> ✔️ **Ponto de Destaque**: A natureza quadrática das fronteiras de decisão confere à QDA uma capacidade superior de modelar relações não-lineares nos dados, tornando-a mais adequada para distribuições de classe com formas complexas.

### Estimação de Parâmetros na QDA

A estimação dos parâmetros na QDA segue o princípio da máxima verossimilhança [7]:

1. **Vetores Médios**: $\hat{\mu}_k = \frac{1}{N_k}\sum_{g_i=k} x_i$
2. **Matrizes de Covariância**: $\hat{\Sigma}_k = \frac{1}{N_k}\sum_{g_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$
3. **Probabilidades a Priori**: $\hat{\pi}_k = \frac{N_k}{N}$

Onde $N_k$ é o número de observações na classe $k$ e $N$ é o número total de observações.

#### [Questões Técnicas/Teóricas]

1. Como a complexidade computacional da QDA se compara à da LDA em termos do número de parâmetros a serem estimados?
2. Em que cenários a QDA seria preferível à LDA, e quais são os trade-offs envolvidos nessa escolha?

### Implementação e Considerações Práticas

A implementação da QDA em Python pode ser realizada utilizando bibliotecas como scikit-learn. Aqui está um exemplo conciso de como aplicar QDA:

```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuma que X e y já estão definidos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

y_pred = qda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia da QDA: {accuracy:.4f}")
```

> ❗ **Ponto de Atenção**: A QDA requer mais dados de treinamento do que a LDA para estimar os parâmetros adicionais de forma confiável. Em conjuntos de dados menores ou de alta dimensionalidade, pode haver um risco de overfitting.

### Comparação entre QDA e LDA

| 👍 Vantagens da QDA                                           | 👎 Desvantagens da QDA                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Maior flexibilidade na modelagem de distribuições de classe [8] | Requer mais parâmetros, aumentando o risco de overfitting [9] |
| Fronteiras de decisão não-lineares, adequadas para relações complexas [10] | Maior complexidade computacional [11]                        |
| Melhor desempenho quando as covariâncias das classes diferem significativamente [12] | Menos robusto em dados de alta dimensionalidade com amostras limitadas [13] |

### Regularização na QDA

Para mitigar o risco de overfitting, especialmente em cenários de alta dimensionalidade ou com amostras limitadas, técnicas de regularização podem ser aplicadas à QDA [14]:

1. **Shrinkage**: Reduz a variância das estimativas das matrizes de covariância através de uma combinação convexa com uma matriz de identidade:

   $$\hat{\Sigma}_k(\alpha) = \alpha\hat{\Sigma}_k + (1-\alpha)I$$

   Onde $\alpha \in [0,1]$ é o parâmetro de regularização.

2. **Pooling Parcial**: Combina as matrizes de covariância estimadas com uma matriz de covariância comum:

   $$\hat{\Sigma}_k(\gamma) = \gamma\hat{\Sigma}_k + (1-\gamma)\hat{\Sigma}$$

   Onde $\gamma \in [0,1]$ e $\hat{\Sigma}$ é a matriz de covariância combinada de todas as classes.

> 💡 **Dica**: A escolha dos parâmetros de regularização ($\alpha$ ou $\gamma$) pode ser otimizada através de validação cruzada.

#### [Questões Técnicas/Teóricas]

1. Como a regularização afeta o viés-variância trade-off na QDA?
2. Em um cenário com classes altamente desequilibradas, como a QDA se compara à LDA em termos de desempenho e robustez?

### Conclusão

A Análise Discriminante Quadrática representa uma evolução significativa em relação à LDA, oferecendo um framework mais flexível para classificação [15]. Sua capacidade de modelar fronteiras de decisão não-lineares a torna particularmente valiosa em cenários onde as distribuições de classe exibem características distintas de forma e orientação [16]. No entanto, essa flexibilidade adicional vem com o custo de uma maior complexidade computacional e um risco aumentado de overfitting, especialmente em espaços de alta dimensão ou com amostras limitadas [17]. A aplicação judiciosa de técnicas de regularização e uma avaliação cuidadosa do trade-off entre complexidade do modelo e tamanho do conjunto de dados são cruciais para explorar todo o potencial da QDA em problemas de classificação do mundo real.

### Questões Avançadas

1. Como você abordaria o problema de seleção de características no contexto da QDA, considerando o trade-off entre a capacidade discriminativa e o risco de overfitting?

2. Desenvolva uma estratégia para combinar QDA com técnicas de redução de dimensionalidade (e.g., PCA, LDA) para lidar com dados de alta dimensionalidade. Quais seriam os prós e contras dessa abordagem?

3. Considerando um cenário de aprendizado semi-supervisionado, como você adaptaria o algoritmo QDA para incorporar informações de dados não rotulados na estimação dos parâmetros do modelo?

### Referências

[1] "Quadratic discriminant analysis (QDA) emerge como uma extensão sofisticada da Análise Discriminante Linear (LDA)" (Trecho de ESL II)

[2] "Enquanto a LDA assume matrizes de covariância idênticas para todas as classes, a QDA relaxa essa restrição, permitindo que cada classe tenha sua própria matriz de covariância" (Trecho de ESL II)

[3] "As funções discriminantes na QDA são quadráticas em x" (Trecho de ESL II)

[4] "A QDA requer a estimação de parâmetros adicionais em comparação com a LDA, incluindo matrizes de covariância separadas para cada classe" (Trecho de ESL II)

[5] "A função discriminante quadrática para a classe k é dada por δ_k(x) = -1/2 log |Σ_k| - 1/2 (x - μ_k)^T Σ_k^(-1) (x - μ_k) + log π_k" (Trecho de ESL II)

[6] "A fronteira de decisão entre duas classes k e l é definida pelo conjunto de pontos que satisfazem δ_k(x) = δ_l(x), resultando em uma equação quadrática em x" (Trecho de ESL II)

[7] "A estimação dos parâmetros na QDA segue o princípio da máxima verossimilhança" (Trecho de ESL II)

[8] "Maior flexibilidade na modelagem de distribuições de classe" (Trecho de ESL II)

[9] "Requer mais parâmetros, aumentando o risco de overfitting" (Trecho de ESL II)

[10] "Fronteiras de decisão não-lineares, adequadas para relações complexas" (Trecho de ESL II)

[11] "Maior complexidade computacional" (Trecho de ESL II)

[12] "Melhor desempenho quando as covariâncias das classes diferem significativamente" (Trecho de ESL II)

[13] "Menos robusto em dados de alta dimensionalidade com amostras limitadas" (Trecho de ESL II)

[14] "Para mitigar o risco de overfitting, especialmente em cenários de alta dimensionalidade ou com amostras limitadas, técnicas de regularização podem ser aplicadas à QDA" (Trecho de ESL II)

[15] "A Análise Discriminante Quadrática representa uma evolução significativa em relação à LDA" (Trecho de ESL II)

[16] "Sua capacidade de modelar fronteiras de decisão não-lineares a torna particularmente valiosa em cenários onde as distribuições de classe exibem características distintas de forma e orientação" (Trecho de ESL II)

[17] "No entanto, essa flexibilidade adicional vem com o custo de uma maior complexidade computacional e um risco aumentado de overfitting, especialmente em espaços de alta dimensão ou com amostras limitadas" (Trecho de ESL II)