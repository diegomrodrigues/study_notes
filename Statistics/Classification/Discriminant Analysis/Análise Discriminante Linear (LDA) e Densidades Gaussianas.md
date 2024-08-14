## Análise Discriminante Linear (LDA) e Densidades Gaussianas

![image-20240802161118321](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240802161118321.png)

A Análise Discriminante Linear (LDA) é uma técnica fundamental em aprendizado de máquina e estatística, particularmente útil para problemas de classificação. Este método baseia-se na suposição de que as classes seguem distribuições Gaussianas multivariadas com uma matriz de covariância comum, oferecendo uma abordagem poderosa e interpretável para a classificação [1].

### Conceitos Fundamentais

| Conceito                                | Explicação                                                   |
| --------------------------------------- | ------------------------------------------------------------ |
| **Distribuição Gaussiana Multivariada** | Generalização multidimensional da distribuição normal, caracterizada por um vetor de médias $\mu$ e uma matriz de covariância $\Sigma$ [1] |
| **Matriz de Covariância Comum**         | Suposição chave do LDA onde todas as classes compartilham a mesma estrutura de covariância [1] |
| **Fronteira de Decisão Linear**         | Resultado da aplicação do LDA, separando classes no espaço de características [2] |

> ⚠️ **Nota Importante**: A suposição de matriz de covariância comum é crucial para a linearidade das fronteiras de decisão no LDA.

### Formulação Matemática do LDA

O LDA modela cada classe $k$ com uma densidade Gaussiana multivariada [1]:

$$
f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} e^{-\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)}
$$

Onde:
- $x$ é o vetor de características
- $\mu_k$ é o vetor de médias da classe $k$
- $\Sigma$ é a matriz de covariância comum a todas as classes
- $p$ é a dimensão do espaço de características

A regra de classificação do LDA baseia-se na comparação das probabilidades posteriores [2]:

$$
\log \frac{Pr(G=k|X=x)}{Pr(G=l|X=x)} = \log \frac{\pi_k}{\pi_l} - \frac{1}{2}(\mu_k + \mu_l)^T \Sigma^{-1}(\mu_k - \mu_l) + x^T \Sigma^{-1}(\mu_k - \mu_l)
$$

Onde:
- $G$ é a variável de classe
- $\pi_k$ é a probabilidade a priori da classe $k$

> ✔️ **Ponto de Destaque**: A linearidade desta expressão em $x$ resulta em fronteiras de decisão lineares entre as classes.

#### Questões Técnicas/Teóricas

1. Como a suposição de matriz de covariância comum no LDA influencia a forma das fronteiras de decisão?
2. Derive a expressão para a fronteira de decisão entre duas classes no LDA, assumindo probabilidades a priori iguais.

### Estimação de Parâmetros no LDA

Os parâmetros do modelo LDA são estimados a partir dos dados de treinamento [3]:

1. Probabilidades a priori: $\hat{\pi}_k = N_k/N$, onde $N_k$ é o número de observações da classe $k$
2. Vetores de médias: $\hat{\mu}_k = \sum_{g_i=k} x_i/N_k$
3. Matriz de covariância comum: $\hat{\Sigma} = \sum_{k=1}^K \sum_{g_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T / (N - K)$

> ❗ **Ponto de Atenção**: A estimação robusta desses parâmetros é crucial para o desempenho do LDA, especialmente em dimensões elevadas.

### Comparação com Regressão Logística

| 👍 Vantagens do LDA                               | 👎 Desvantagens do LDA                                        |
| ------------------------------------------------ | ------------------------------------------------------------ |
| Eficiente com amostras pequenas [4]              | Sensível a outliers [5]                                      |
| Interpretabilidade das fronteiras de decisão [4] | Suposição de normalidade pode ser restritiva [5]             |
| Captura estrutura de covariância dos dados [4]   | Pode falhar se as classes têm variâncias muito diferentes [5] |

#### Questões Técnicas/Teóricas

1. Em que cenários o LDA pode superar a regressão logística em termos de desempenho de classificação?
2. Como você modificaria o LDA para lidar com classes que têm matrizes de covariância diferentes?

### LDA de Posto Reduzido

Uma extensão importante do LDA é a versão de posto reduzido, que projeta os dados em um subespaço de dimensão menor [6]:

1. Calcule a matriz de centroides das classes $M$ (dimensão $K \times p$)
2. Compute $M^* = MW^{-\frac{1}{2}}$, onde $W$ é a matriz de covariância intra-classe
3. Realize a decomposição em autovalores de $B^* = V^*D_BV^{*T}$, onde $B^*$ é a covariância de $M^*$

As variáveis discriminantes são dadas por $Z_l = v_l^T X$, onde $v_l = W^{-\frac{1}{2}} v_l^*$ [6].

> ✔️ **Ponto de Destaque**: Esta abordagem permite uma visualização de baixa dimensão dos dados, mantendo a separabilidade das classes.

### Implementação em Python

Aqui está um exemplo simplificado de implementação do LDA usando sklearn:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Gerar dados sintéticos
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=15, random_state=42)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciar e treinar o modelo LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Avaliar o modelo
print(f"Acurácia no conjunto de teste: {lda.score(X_test, y_test):.4f}")

# Projetar dados no espaço discriminante
X_lda = lda.transform(X_test)
```

Este código demonstra como treinar um modelo LDA, avaliar seu desempenho e projetar os dados no espaço discriminante reduzido [7].

### Conclusão

A Análise Discriminante Linear é uma técnica poderosa e interpretável para classificação, baseada em suposições de normalidade e homogeneidade das covariâncias entre classes. Sua eficácia em muitos problemas práticos, combinada com a capacidade de redução de dimensionalidade, torna o LDA uma ferramenta valiosa no arsenal de qualquer cientista de dados ou estatístico [8].

### Questões Avançadas

1. Como você adaptaria o LDA para lidar com dados de alta dimensão onde $p > N$? Discuta as implicações teóricas e práticas.
2. Compare e contraste o LDA com métodos de classificação não-paramétricos como SVM e Random Forests. Em que cenários cada método seria preferível?
3. Derive a relação entre LDA e Análise de Correlação Canônica. Como essa relação pode ser explorada para melhorar a interpretabilidade dos resultados do LDA?

### Referências

[1] "Suppose that we model each class density as multivariate Gaussian" (Trecho de ESL II)

[2] "Linear discriminant analysis (LDA) arises in the special case when we assume that the classes have a common covariance matrix Σk = Σ ∀k." (Trecho de ESL II)

[3] "In practice we do not know the parameters of the Gaussian distributions, and will need to estimate them using our training data" (Trecho de ESL II)

[4] "LDA is a very popular method for classification" (Trecho de ESL II)

[5] "The reason is not likely to be that the data are approximately Gaussian, and in addition for LDA that the covariances are approximately equal." (Trecho de ESL II)

[6] "Fisher defined optimal to mean that the projected centroids were spread out as much as possible in terms of variance." (Trecho de ESL II)

[7] "Software implementations can take advantage of these connections." (Trecho de ESL II)

[8] "LDA and QDA perform well on an amazingly large and diverse set of classification tasks." (Trecho de ESL II)