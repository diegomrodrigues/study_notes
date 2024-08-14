## Funções Discriminantes Lineares na Análise Discriminante Linear (LDA)

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240802162000958.png" alt="image-20240802162000958" style="zoom: 80%;" />

### Introdução

As funções discriminantes lineares são fundamentais na teoria de classificação estatística, particularmente na Análise Discriminante Linear (LDA). Elas fornecem um método poderoso e interpretável para separar classes em problemas de classificação multivariada [1]. Este resumo explorará em profundidade a derivação, interpretação e aplicação dessas funções no contexto da LDA.

### Conceitos Fundamentais

| Conceito                 | Explicação                                                   |
| ------------------------ | ------------------------------------------------------------ |
| **Função Discriminante** | Uma função matemática que mapeia um vetor de características de entrada para um escore, usado para tomar decisões de classificação [1]. |
| **Linearidade**          | A propriedade de uma função discriminante que permite que ela seja expressa como uma combinação linear das variáveis de entrada [2]. |
| **Fronteira de Decisão** | O locus de pontos no espaço de características onde a função discriminante é igual para duas ou mais classes [3]. |

> ✔️ **Ponto de Destaque**: As funções discriminantes lineares na LDA são derivadas assumindo que as classes têm distribuições gaussianas com matrizes de covariância iguais [4].

### Derivação das Funções Discriminantes Lineares

A derivação das funções discriminantes lineares na LDA começa com a suposição de que cada classe $k$ tem uma distribuição gaussiana multivariada [5]:

$$
f_k(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} e^{-\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)}
$$

onde $x$ é o vetor de características, $\mu_k$ é o vetor médio da classe $k$, e $\Sigma$ é a matriz de covariância comum a todas as classes.

Para classificação, usamos a regra de Bayes [6]:

$$
P(G=k|X=x) = \frac{f_k(x)\pi_k}{\sum_{l=1}^K f_l(x)\pi_l}
$$

onde $\pi_k$ é a probabilidade a priori da classe $k$.

Tomando o logaritmo e simplificando, obtemos a função discriminante linear [7]:

$$
\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log\pi_k
$$

> ❗ **Ponto de Atenção**: A linearidade em $x$ surge devido ao cancelamento dos termos quadráticos na expansão do expoente gaussiano, uma consequência direta da suposição de covariâncias iguais [8].

#### Questões Técnicas/Teóricas

1. Como a suposição de covariâncias iguais entre as classes influencia a forma das funções discriminantes na LDA?
2. Explique por que o termo quadrático em $x$ desaparece na derivação da função discriminante linear.

### Interpretação Geométrica

As funções discriminantes lineares definem hiperplanos no espaço de características [9]. A fronteira de decisão entre duas classes $k$ e $l$ é dada por:

$$
\{x : \delta_k(x) = \delta_l(x)\}
$$

Esta é uma equação linear em $x$, representando um hiperplano [10].

![image-20240802162423217](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240802162423217.png)

> ⚠️ **Nota Importante**: Em um problema com $K$ classes, haverá no máximo $K-1$ funções discriminantes linearmente independentes [11].

### Estimação dos Parâmetros

Na prática, os parâmetros $\mu_k$, $\Sigma$, e $\pi_k$ são desconhecidos e devem ser estimados a partir dos dados de treinamento [12]. As estimativas de máxima verossimilhança são:

1. $\hat{\mu}_k = \frac{1}{N_k}\sum_{g_i=k} x_i$
2. $\hat{\Sigma} = \frac{1}{N-K}\sum_{k=1}^K\sum_{g_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$
3. $\hat{\pi}_k = N_k/N$

onde $N_k$ é o número de observações na classe $k$ e $N$ é o número total de observações [13].

#### Demonstração

Começamos com as seguintes suposições:

1. Temos $K$ classes
2. Cada classe $k$ segue uma distribuição normal multivariada $N(\mu_k, \Sigma)$
3. Todas as classes compartilham a mesma matriz de covariância $\Sigma$
4. Temos $N$ observações no total, com $N_k$ observações na classe $k$

Passo 1: Função de verossimilhança

A função de verossimilhança para todas as observações é:

$$
L(\mu_1,...,\mu_K, \Sigma, \pi_1,...,\pi_K) = \prod_{i=1}^N [\pi_{g_i} f_{g_i}(x_i)]
$$

Onde $g_i$ é a classe da observação $i$, e $f_k(x)$ é a densidade da distribuição normal multivariada para a classe $k$.

Passo 2: Log-verossimilhança

Tomamos o logaritmo da função de verossimilhança:

$$
\ell = \log L = \sum_{i=1}^N [\log \pi_{g_i} + \log f_{g_i}(x_i)]
$$

Passo 3: Estimativa de $\pi_k$

Para estimar $\pi_k$, maximizamos $\ell$ sujeito à restrição $\sum_{k=1}^K \pi_k = 1$. Usando multiplicadores de Lagrange, obtemos:

$$
\frac{\partial\ell}{\partial\pi_k} = \frac{N_k}{\pi_k} - \lambda = 0
$$

Resolvendo e usando a restrição, obtemos:

$$
\hat{\pi}_k = \frac{N_k}{N}
$$

Passo 4: Estimativa de $\mu_k$

Para estimar $\mu_k$, derivamos $\ell$ em relação a $\mu_k$ e igualamos a zero:

$$
\frac{\partial\ell}{\partial\mu_k} = \sum_{g_i=k} \Sigma^{-1}(x_i - \mu_k) = 0
$$

Resolvendo, obtemos:

$$
\hat{\mu}_k = \frac{1}{N_k} \sum_{g_i=k} x_i
$$

Passo 5: Estimativa de $\Sigma$

Para estimar $\Sigma$, derivamos $\ell$ em relação a $\Sigma^{-1}$ e igualamos a zero:

$$
\frac{\partial\ell}{\partial\Sigma^{-1}} = \frac{1}{2} \sum_{k=1}^K N_k [\Sigma - \frac{1}{N_k} \sum_{g_i=k} (x_i - \mu_k)(x_i - \mu_k)^T] = 0
$$

Resolvendo e substituindo as estimativas de $\mu_k$, obtemos:

$$
\hat{\Sigma} = \frac{1}{N-K} \sum_{k=1}^K \sum_{g_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T
$$

Observe que usamos $N-K$ no denominador em vez de $N$ para obter um estimador não-viesado.

Assim, chegamos às estimativas de máxima verossimilhança:

1. $\hat{\pi}_k = \frac{N_k}{N}$
2. $\hat{\mu}_k = \frac{1}{N_k} \sum_{g_i=k} x_i$
3. $\hat{\Sigma} = \frac{1}{N-K} \sum_{k=1}^K \sum_{g_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$

Estas estimativas são intuitivas:
- $\hat{\pi}_k$ é a proporção de observações na classe $k$
- $\hat{\mu}_k$ é a média das observações na classe $k$
- $\hat{\Sigma}$ é a média ponderada das matrizes de covariância dentro de cada classe

Esta demonstração mostra como as estimativas surgem naturalmente da maximização da verossimilhança sob as suposições do modelo LDA.

### Propriedades e Limitações

#### 👍Vantagens
* Simplicidade e interpretabilidade [14]
* Eficiência computacional [15]
* Bom desempenho quando as suposições são atendidas [16]

#### 👎Desvantagens
* Sensibilidade a outliers [17]
* Desempenho subótimo quando as classes não são linearmente separáveis [18]
* Suposição restritiva de covariâncias iguais [19]

### Extensões e Variantes

1. **Análise Discriminante Quadrática (QDA)**: Relaxa a suposição de covariâncias iguais, resultando em fronteiras de decisão quadráticas [20].

2. **Análise Discriminante Regularizada**: Introduz um termo de regularização para lidar com multicolinearidade e melhorar a estabilidade [21].

$$
\hat{\Sigma}(\alpha) = \alpha\hat{\Sigma} + (1-\alpha)\hat{\sigma}^2I
$$

onde $\alpha \in [0,1]$ é o parâmetro de regularização [22].

3. **LDA de Posto Reduzido**: Restringe as médias das classes a um subespaço de dimensão menor, útil para visualização e redução de dimensionalidade [23].

#### Questões Técnicas/Teóricas

1. Como a LDA se compara à regressão logística em termos de suposições e performance?
2. Descreva um cenário em que a QDA seria preferível à LDA.

### Implementação em Python

Aqui está um exemplo simplificado de implementação de LDA em Python:

```python
import numpy as np
from scipy.linalg import inv

class LDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = [X[y == c].mean(axis=0) for c in self.classes]
        self.priors = [np.mean(y == c) for c in self.classes]
        
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for c, mean in zip(self.classes, self.means):
            Sw += np.cov(X[y == c].T) * (sum(y == c) - 1)
        self.Sw = Sw / (len(X) - len(self.classes))
        
    def predict(self, X):
        return np.argmax([self._discriminant_function(X, c) for c in range(len(self.classes))], axis=0)
    
    def _discriminant_function(self, X, c):
        return X @ inv(self.Sw) @ self.means[c] - 0.5 * self.means[c] @ inv(self.Sw) @ self.means[c] + np.log(self.priors[c])
```

> ⚠️ **Nota Importante**: Esta implementação é simplificada e não inclui otimizações ou tratamento de casos especiais que seriam necessários em uma implementação robusta para uso em produção.

### Conclusão

As funções discriminantes lineares na LDA oferecem uma abordagem poderosa e interpretável para problemas de classificação multivariada. Sua derivação a partir de princípios estatísticos sólidos, combinada com sua eficiência computacional, torna a LDA uma ferramenta valiosa no arsenal de um cientista de dados. No entanto, é crucial entender suas limitações e saber quando aplicar extensões ou métodos alternativos.

### Questões Avançadas

1. Como você modificaria a implementação de LDA para lidar com classes altamente desequilibradas? Discuta as implicações teóricas e práticas dessa modificação.

2. Derive a forma da função discriminante para a Análise Discriminante Quadrática (QDA) e explique como isso afeta a complexidade computacional e a capacidade de modelagem em comparação com a LDA.

3. Proponha e justifique um método para combinar LDA com técnicas de ensemble learning, como bagging ou boosting. Que desafios você antecipa e como os abordaria?

### Referências

[1] "A classificação linear discriminante (LDA) é uma generalização da regra discriminante de Fisher." (Trecho de ESL II)

[2] "Estas funções discriminantes lineares implicam que as fronteiras de decisão entre as classes são lineares em x." (Trecho de ESL II)

[3] "O conjunto onde Pr(G = k|X = x) = Pr(G = ℓ|X = x) é uma fronteira de decisão linear." (Trecho de ESL II)

[4] "Suponha que modelamos cada densidade de classe como multivariada Gaussiana." (Trecho de ESL II)

[5] "f_k(x) = 1/(2π)^(p/2)|Σ_k|^(1/2) e^(-(1/2)(x-μ_k)^T Σ_k^(-1)(x-μ_k))." (Trecho de ESL II)

[6] "Uma simples aplicação do teorema de Bayes nos dá Pr(G = k|X = x) = f_k(x)π_k / Σ_ℓ f_ℓ(x)π_ℓ." (Trecho de ESL II)

[7] "log Pr(G = k|X = x) = log π_k - (1/2) log |Σ| - (1/2)(x - μ_k)^T Σ^(-1)(x - μ_k) + const." (Trecho de ESL II)

[8] "A igualdade das matrizes de covariância causa o cancelamento dos fatores de normalização, bem como da parte quadrática nos expoentes." (Trecho de ESL II)

[9] "As fronteiras de decisão lineares são hiperplanos em IR^p." (Trecho de ESL II)

[10] "Isto é, claro, verdade para qualquer par de classes, então todas as fronteiras de decisão são lineares." (Trecho de ESL II)

[11] "Se K = 3, por exemplo, isso poderia nos permitir ver os dados em um gráfico bidimensional." (Trecho de ESL II)

[12] "Na prática, não conhecemos os parâmetros das distribuições Gaussianas e precisaremos estimá-los usando nossos dados de treinamento." (Trecho de ESL II)

[13] "μ̂_k = Σ_(g_i=k) x_i/N_k, onde N_k é o número de observações de classe k." (Trecho de ESL II)

[14] "Parte de sua popularidade se deve a uma restrição adicional que nos permite ver projeções informativas de baixa dimensão dos dados." (Trecho de ESL II)

[15] "A LDA é computacionalmente eficiente, especialmente quando comparada a métodos mais complexos." (Trecho de ESL II)

[16] "LDA e QDA têm bom desempenho em um conjunto surpreendentemente grande e diverso de tarefas de classificação." (Trecho de ESL II)

[17] "Isto não é toda boa notícia, porque também significa que a LDA não é robusta a outliers grosseiros." (Trecho de ESL II)

[18] "Claramente, elas podem apenas suportar fronteiras de decisão simples como linear ou quadrática." (Trecho de ESL II)

[19] "A razão é provavelmente não que os dados são aproximadamente Gaussianos, e além disso para LDA que as covariâncias são aproximadamente iguais." (Trecho de ESL II)

[20] "Se as Σ_k não são assumidas como iguais, então os cancelamentos convenientes em (4.9) não ocorrem; em particular, as peças quadráticas em x permanecem." (Trecho de ESL II)

[21] "Friedman (1989) propôs um compromisso entre LDA e QDA, que permite que se encolha as covariâncias separadas de QDA em direção a uma covariância comum como na LDA." (Trecho de ESL II)

[22] "As matrizes de covariância regularizadas têm a forma Σ̂_k(α) = αΣ̂_k + (1 - α)Σ̂." (Trecho de ESL II)

[23] "Fisher definiu ótimo para significar que os centroides projetados estavam espalhados o máximo possível em termos de variância." (Trecho de ESL II)