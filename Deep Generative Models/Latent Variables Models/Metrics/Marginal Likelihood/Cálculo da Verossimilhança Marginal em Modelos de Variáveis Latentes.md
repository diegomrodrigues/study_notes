## Cálculo da Verossimilhança Marginal em Modelos de Variáveis Latentes

### ![image-20240821180547066](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821180547066.png)

### Introdução

==O cálculo da verossimilhança marginal é um componente crucial na análise e treinamento de modelos de variáveis latentes==, desempenhando um papel fundamental em áreas como aprendizado não supervisionado, clustering e compressão de dados [1]. Este resumo explorará em profundidade os métodos, desafios e implicações do cálculo da verossimilhança marginal, com foco particular na ==integração ou soma sobre variáveis latentes e nos desafios computacionais associados a modelos complexos.==

### Conceitos Fundamentais

| Conceito                                     | Explicação                                                   |
| -------------------------------------------- | ------------------------------------------------------------ |
| **Verossimilhança Marginal**                 | ==A probabilidade total dos dados observados, obtida pela integração ou soma sobre todas as possíveis configurações das variáveis latentes. [1]== |
| **Variáveis Latentes**                       | Variáveis não observadas diretamente nos dados, mas que influenciam as variáveis observadas e capturam estruturas ocultas nos dados. [2] |
| **Integração/Soma sobre Variáveis Latentes** | ==Processo matemático de considerar todas as possíveis configurações das variáveis latentes para calcular a verossimilhança marginal. [1]== |

> ⚠️ **Nota Importante**: O cálculo da verossimilhança marginal é frequentemente intratável para modelos complexos, necessitando de métodos de aproximação.

### Cálculo da Verossimilhança Marginal

A verossimilhança marginal $p(x)$ para um modelo de variável latente é definida como:

$$
p(x) = \int p(x,z) dz = \int p(x|z)p(z) dz
$$

onde $x$ são as variáveis observadas e $z$ são as variáveis latentes [1].

Para modelos discretos, a integral se torna uma soma:

$$
p(x) = \sum_z p(x,z) = \sum_z p(x|z)p(z)
$$

#### Métodos de Cálculo

1. **Integração Analítica**: Possível apenas para modelos simples com distribuições conjugadas.

2. **Integração Numérica**: Utilizada para modelos de baixa dimensionalidade.

3. **Métodos de Monte Carlo**: Aproximação da integral usando amostragem.

4. **Aproximações Variacionais**: Otimização de um limite inferior da log-verossimilhança.

#### Exemplo: Mistura de Gaussianas

Para uma mistura de K componentes gaussianas:

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
$$

==onde $\pi_k$ são os pesos da mistura, e $\mathcal{N}(x|\mu_k, \Sigma_k)$ são as densidades gaussianas [2].==

#### Questões Técnicas/Teóricas

1. Como o cálculo da verossimilhança marginal difere entre modelos com variáveis latentes contínuas e discretas?
2. Explique por que a verossimilhança marginal é importante no contexto de seleção de modelos.

### Desafios Computacionais para Modelos Complexos

O cálculo exato da verossimilhança marginal torna-se computacionalmente intratável para modelos complexos devido a:

1. **Alta Dimensionalidade**: O número de possíveis configurações das variáveis latentes cresce exponencialmente com sua dimensionalidade.

2. **Não-linearidade**: ==Modelos não-lineares, como redes neurais profundas, tornam a integração analítica impossível.==

3. **Multimodalidade**: ==Distribuições posteriores complexas com múltiplos modos dificultam a integração numérica e a amostragem eficiente.==

#### Técnicas de Aproximação

Para lidar com esses desafios, várias técnicas de aproximação são empregadas:

1. **Amostragem de Importância**: 
   
   $$
   p(x) \approx \frac{1}{S}\sum_{s=1}^S \frac{p(x|z^{(s)})p(z^{(s)})}{q(z^{(s)})}
   $$
   
   onde $z^{(s)}$ são amostras de uma distribuição proposta $q(z)$ [1].

2. **Métodos de Monte Carlo via Cadeias de Markov (MCMC)**:
   Geram amostras da distribuição posterior $p(z|x)$ para aproximar a integral.

3. **Aproximação Variacional**:
   Otimiza uma família de distribuições $q_\phi(z|x)$ para aproximar $p(z|x)$, fornecendo um limite inferior na log-verossimilhança:
   
   $$
   \log p(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p(x|z)] - KL(q_\phi(z|x)||p(z))
   $$

4. **Autocodificadores Variacionais (VAEs)**:
   Combinam redes neurais com inferência variacional para modelos de alta dimensionalidade [3].

> ✔️ **Ponto de Destaque**: A escolha do método de aproximação depende do equilíbrio entre precisão, eficiência computacional e escalabilidade para grandes conjuntos de dados.

### Implementação em Python

Aqui está um exemplo simplificado de como implementar o cálculo da verossimilhança marginal para uma mistura de gaussianas usando PyTorch:

```python
import torch
import torch.distributions as dist

def log_likelihood_gmm(x, means, covs, weights):
    K = len(weights)
    N, D = x.shape
    
    log_probs = torch.zeros(N, K)
    for k in range(K):
        gaussian = dist.MultivariateNormal(means[k], covs[k])
        log_probs[:, k] = torch.log(weights[k]) + gaussian.log_prob(x)
    
    return torch.logsumexp(log_probs, dim=1).sum()

# Exemplo de uso
N, D, K = 1000, 2, 3
x = torch.randn(N, D)
means = torch.randn(K, D)
covs = torch.stack([torch.eye(D) for _ in range(K)])
weights = torch.softmax(torch.randn(K), dim=0)

log_likelihood = log_likelihood_gmm(x, means, covs, weights)
print(f"Log-verossimilhança: {log_likelihood.item()}")
```

Este código calcula a log-verossimilhança marginal para um conjunto de dados `x` assumindo um modelo de mistura de gaussianas com `K` componentes.

#### Questões Técnicas/Teóricas

1. Como você modificaria o código acima para implementar uma aproximação variacional para o cálculo da verossimilhança marginal?
2. Discuta as vantagens e desvantagens de usar PyTorch para este tipo de cálculo em comparação com outras bibliotecas como NumPy.

### Aplicações e Implicações

O cálculo da verossimilhança marginal tem implicações significativas em várias áreas:

1. **Seleção de Modelos**: Comparação de modelos usando critérios como BIC (Bayesian Information Criterion) ou evidência marginal.

2. **Inferência Bayesiana**: Cálculo de probabilidades posteriores e fatores de Bayes.

3. **Aprendizado Não Supervisionado**: Avaliação da qualidade de agrupamentos e representações latentes.

4. **Detecção de Anomalias**: Identificação de exemplos com baixa verossimilhança sob o modelo.

> ❗ **Ponto de Atenção**: A interpretação da verossimilhança marginal deve ser feita com cautela, especialmente ao comparar modelos de complexidades diferentes.

### Conclusão

O cálculo da verossimilhança marginal é um desafio central no treinamento e análise de modelos de variáveis latentes. Enquanto métodos exatos são viáveis apenas para modelos simples, técnicas de aproximação como MCMC, inferência variacional e autocodificadores variacionais permitem lidar com modelos mais complexos e datasets de alta dimensionalidade. A escolha do método apropriado depende das características específicas do modelo e dos requisitos de precisão e eficiência computacional.

### Questões Avançadas

1. Como você abordaria o cálculo da verossimilhança marginal para um modelo hierárquico bayesiano complexo com múltiplas camadas de variáveis latentes?

2. Discuta as implicações teóricas e práticas de usar a evidência marginal (verossimilhança marginal integrada sobre os parâmetros do modelo) versus a verossimilhança marginal para seleção de modelos.

3. Proponha e justifique uma estratégia para calcular a verossimilhança marginal de um modelo de linguagem baseado em transformers com milhões de parâmetros.

### Referências

[1] "Log-Likelihood function for Partially Observed Data is hard to compute:
log(∑z∈Z p_θ(x, z))" (Trecho de cs236_lecture5.pdf)

[2] "A mixture of an infinite number of Gaussians:
1. z ∼ N(0, I)
2. p(x | z) = N(μ_θ(z), Σ_θ(z)) where μ_θ,Σ_θ are neural networks" (Trecho de cs236_lecture5.pdf)

[3] "Variational autoencoder, or VAE (Kingma and Welling, 2013; Rezende, Mohamed, and Wierstra, 2014; Doer-
sch, 2016; Kingma and Welling, 2019) instead works with an approximation to this
likelihood when training the model." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)