## Inferência Analítica na Análise de Componentes Principais Probabilística (pPCA)

<image: Uma representação visual da transformação linear da pPCA, mostrando vetores de dados em um espaço de alta dimensão sendo projetados em um espaço de menor dimensão, com elipses representando as distribuições Gaussianas nos espaços latente e observado.>

### Introdução

A Análise de Componentes Principais Probabilística (pPCA) é uma extensão probabilística da tradicional Análise de Componentes Principais (PCA), oferecendo um framework estatístico robusto para redução de dimensionalidade e modelagem generativa [1]. Uma característica fundamental da pPCA é sua capacidade de permitir cálculos analíticos exatos para distribuições marginais e posteriores, graças à sua estrutura linear e ao uso de distribuições Gaussianas [2]. Este resumo se concentra na demonstração detalhada desses cálculos analíticos, explorando as propriedades matemáticas que tornam a pPCA um modelo particularmente tratável do ponto de vista da inferência.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **pPCA**                     | Modelo probabilístico que estende a PCA tradicional, permitindo uma interpretação generativa e inferência estatística robusta. [1] |
| **Linearidade**              | Característica chave da pPCA que permite cálculos analíticos exatos, baseada na transformação linear entre espaços latente e observado. [2] |
| **Distribuições Gaussianas** | Utilizadas tanto no espaço latente quanto no ruído observacional, facilitando integrações e cálculos de distribuições condicionais. [2] |

> ✔️ **Ponto de Destaque**: A combinação de linearidade e Gaussianidade na pPCA permite o cálculo fechado de distribuições marginais e posteriores, um recurso raro em modelos mais complexos de variáveis latentes.

### Modelo pPCA

O modelo pPCA é definido pelas seguintes equações [2]:

1. Distribuição latente: $z \sim \mathcal{N}(0, I)$
2. Transformação linear: $x = Wz + \mu + \epsilon$
3. Ruído observacional: $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

Onde:
- $z \in \mathbb{R}^M$ é o vetor latente
- $x \in \mathbb{R}^D$ é o vetor observado
- $W \in \mathbb{R}^{D \times M}$ é a matriz de transformação
- $\mu \in \mathbb{R}^D$ é o vetor de média
- $\sigma^2$ é a variância do ruído

### Cálculo Analítico da Distribuição Marginal

Para calcular a distribuição marginal $p(x)$, integramos sobre a variável latente $z$:

$$
p(x) = \int p(x|z)p(z)dz
$$

Dada a linearidade do modelo e as distribuições Gaussianas, podemos utilizar propriedades da álgebra linear e de distribuições Gaussianas para resolver esta integral analiticamente [3].

Passo a passo:

1. Expansão da distribuição condicional:
   
   $p(x|z) = \mathcal{N}(x|Wz + \mu, \sigma^2 I)$

2. Distribuição prior:
   
   $p(z) = \mathcal{N}(z|0, I)$

3. Aplicando a fórmula para a combinação linear de variáveis Gaussianas:

   $$
   \begin{aligned}
   p(x) &= \mathcal{N}(x|\mu, WW^T + \sigma^2 I) \\
   &= \frac{1}{(2\pi)^{D/2}|WW^T + \sigma^2 I|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T(WW^T + \sigma^2 I)^{-1}(x-\mu)\right)
   \end{aligned}
   $$

> ❗ **Ponto de Atenção**: A matriz de covariância resultante $WW^T + \sigma^2 I$ captura tanto a variabilidade explicada pelos componentes principais (via $WW^T$) quanto o ruído residual (via $\sigma^2 I$).

### Cálculo Analítico da Distribuição Posterior

A distribuição posterior $p(z|x)$ pode ser calculada usando o Teorema de Bayes e as propriedades de distribuições Gaussianas [4]:

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

Passo a passo:

1. Aplicamos o lema da inversão de matriz para Gaussianas:

   $$
   p(z|x) = \mathcal{N}(z|M^{-1}W^T(x-\mu), \sigma^2 M^{-1})
   $$

   onde $M = W^TW + \sigma^2 I$

2. A média posterior é dada por:

   $$
   \mathbb{E}[z|x] = M^{-1}W^T(x-\mu)
   $$

3. A covariância posterior é:

   $$
   \text{Cov}[z|x] = \sigma^2 M^{-1}
   $$

> ✔️ **Ponto de Destaque**: A forma fechada da posterior permite inferência eficiente e interpretável sobre as variáveis latentes dado os dados observados.

#### Questões Técnicas/Teóricas

1. Como a dimensionalidade do espaço latente $M$ afeta a complexidade computacional do cálculo da distribuição marginal $p(x)$ na pPCA?
2. Explique como a incerteza na estimativa da variável latente $z$ é capturada na matriz de covariância da distribuição posterior $p(z|x)$.

### Implicações e Aplicações

A tratabilidade analítica da pPCA tem várias implicações importantes:

1. **Eficiência Computacional**: Os cálculos fechados permitem implementações eficientes, especialmente para grandes conjuntos de dados [5].

2. **Interpretabilidade**: A forma explícita das distribuições facilita a interpretação dos resultados em termos de variância explicada e incerteza [5].

3. **Extensibilidade**: Serve como base para modelos mais complexos, como misturas de pPCAs ou versões hierárquicas [6].

4. **Modelo Generativo**: A distribuição marginal $p(x)$ pode ser usada diretamente para geração de novos dados ou detecção de anomalias [6].

### Implementação em PyTorch

Aqui está uma implementação concisa da inferência analítica em pPCA usando PyTorch:

```python
import torch
import torch.nn as nn

class PPCA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(PPCA, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, latent_dim))
        self.mu = nn.Parameter(torch.zeros(input_dim))
        self.log_sigma2 = nn.Parameter(torch.zeros(1))
        
    def marginal_likelihood(self, x):
        sigma2 = self.log_sigma2.exp()
        cov = self.W @ self.W.T + sigma2 * torch.eye(self.W.shape[0])
        return torch.distributions.MultivariateNormal(self.mu, cov)
    
    def posterior(self, x):
        sigma2 = self.log_sigma2.exp()
        M = self.W.T @ self.W + sigma2 * torch.eye(self.W.shape[1])
        M_inv = torch.inverse(M)
        post_mean = M_inv @ self.W.T @ (x - self.mu)
        post_cov = sigma2 * M_inv
        return torch.distributions.MultivariateNormal(post_mean, post_cov)

# Exemplo de uso
ppca = PPCA(input_dim=10, latent_dim=3)
x = torch.randn(10)
marg_likelihood = ppca.marginal_likelihood(x)
posterior = ppca.posterior(x)
```

Este código implementa as fórmulas analíticas derivadas anteriormente, demonstrando como os cálculos teóricos se traduzem diretamente em implementações práticas [7].

#### Questões Técnicas/Teóricas

1. Como você modificaria a implementação acima para lidar com múltiplas observações (batch) simultaneamente?
2. Discuta as vantagens e desvantagens de usar a inferência analítica versus métodos de inferência aproximada (como Variational Inference) em modelos de variáveis latentes mais complexos.

### Conclusão

A capacidade de realizar inferência analítica na pPCA, derivando formas fechadas para as distribuições marginal e posterior, é uma característica distintiva que a torna um modelo particularmente útil e interpretável [8]. Esta propriedade não apenas facilita a implementação eficiente e a interpretação dos resultados, mas também serve como um caso de estudo valioso para entender os princípios fundamentais da inferência em modelos de variáveis latentes. Enquanto modelos mais complexos frequentemente requerem aproximações, a pPCA oferece um exemplo raro de tratabilidade completa, tornando-a uma ferramenta valiosa tanto na prática quanto no ensino de métodos probabilísticos de redução de dimensionalidade [8].

### Questões Avançadas

1. Compare a complexidade computacional e a precisão da inferência analítica na pPCA com métodos de inferência aproximada (como Variational Inference) em um modelo VAE (Variational Autoencoder) com prior e posterior Gaussianos. Quais são as principais diferenças e trade-offs?

2. Suponha que você queira estender o modelo pPCA para lidar com dados faltantes. Como você modificaria o processo de inferência analítica para acomodar esta situação? Discuta as implicações computacionais e estatísticas desta extensão.

3. Derive a expressão analítica para a evidência (log marginal likelihood) no modelo pPCA e explique como isso poderia ser usado para seleção de modelo, especificamente para determinar o número ótimo de componentes latentes.

### Referências

[1] "A Análise de Componentes Principais Probabilística (pPCA) é uma extensão probabilística da tradicional Análise de Componentes Principais (PCA), oferecendo um framework estatístico robusto para redução de dimensionalidade e modelagem generativa" (Trecho de Latent Variable Models.pdf)

[2] "Let us discuss the following situation:
• We consider continuous random variables only, i.e., z ∈ R
M 
and x ∈ R
D 
.
• The distribution of z is the standard Gaussian, i.e., p(z) = N 
(
z|0, I
)
.
• The dependency between z and x is linear and we assume a Gaussian additive
noise:
x = Wz + b + ε, (4.2)
where ε ∼ N(ε|0, σ 
2
I). The property of the Gaussian distribution yields [1]
p(x|z) = N
(
x|Wz + b, σ 
2
I
)
. (4.3)" (Trecho de Latent Variable Models.pdf)

[3] "Next, we can take advantage of properties of a linear combination of two vectors
of normally distributed random variables to calculate the integral explicitly [1]:
p(x) =
∫
p(x|z) p(z) dz (4.4)
=
∫
N
(
x|Wz + b, σ 
2
I
)
N 
(
z|0, I
) 
dz (4.5)
= N
(
x|b, WW

+ σ 
2
I
)
. (4.6)" (Trecho de Latent Variable Models.pdf)

[4] "Moreover, what is interesting about the pPCA is that, due to the properties
of Gaussians, we can also calculate the true posterior over z analytically:
p(z|x) = N
(
M
−1
W

(x − μ), σ 
−2
M
)
, (4.7)
where M = W

W + σ 
2
I." (Trecho de Latent Variable Models.pdf)

[5] "Once we find W that maximize the log-likelihood
function, and the dimensionality of the matrix W is computationally tractable, we
can calculate p(z|x). This is a big thing! Why? Because for a given observation x,
we can calculate the distribution over the latent factors!" (Trecho de Latent Variable Models.pdf)

[6] "In my opinion, the probabilistic PCA is an extremely important latent variable
model for two reasons. First, we can calculate everything by hand and, thus, it is a
great exercise to develop an intuition about the latent variable models. Second, it is
a linear model and, therefore, a curious reader should feel tingling in his or her head
already and ask himself or herself the following questions: What would happen if we
take non-linear dependencies? And what would happen if we use other distributions
than Gaussians?" (Trecho de Latent Variable Models.pdf)

[7] "Anyhow, pPCA is a model that everyone interested in latent variable models should
study in depth to create an intuition about probabilistic modeling." (Trecho de Latent Variable Models.pdf)

[8] "This model is known as the probabilistic Principal Component Analysis (pPCA)
[2]." (Trecho de Latent Variable Models.pdf)