## Aprendizado de Máxima Verossimilhança com Dados Parcialmente Observados

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821180648240.png" alt="image-20240821180648240" style="zoom: 80%;" />

### Introdução

O aprendizado de máxima verossimilhança com dados parcialmente observados é um tópico fundamental em modelos probabilísticos e aprendizado de máquina, especialmente relevante para modelos generativos profundos e autoencoders variacionais. Este método lida com cenários onde nem todas as variáveis do modelo são observadas diretamente nos dados de treinamento, introduzindo desafios significativos na estimação de parâmetros e inferência [1].

Neste resumo, exploraremos em profundidade a formulação matemática do problema, as dificuldades computacionais associadas ao cálculo exato da verossimilhança, e as técnicas avançadas desenvolvidas para superar essas limitações. Abordaremos desde os fundamentos teóricos até as aplicações práticas em modelos de aprendizado profundo modernos.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Variáveis Latentes**       | Variáveis não observadas diretamente nos dados, mas que influenciam as variáveis observadas. Essenciais em modelos generativos para capturar estruturas complexas. [1] |
| **Verossimilhança Marginal** | A probabilidade dos dados observados, marginalizando sobre todas as possíveis configurações das variáveis latentes. Fundamental para o aprendizado, mas frequentemente intratável. [2] |
| **Inferência Variacional**   | Técnica que aproxima distribuições posteriores intratáveis por distribuições mais simples, otimizando um limite inferior na verossimilhança. [3] |

> ⚠️ **Nota Importante**: A presença de variáveis latentes torna o cálculo direto da verossimilhança computacionalmente intratável para muitos modelos complexos, necessitando de métodos aproximados.

### Formulação do Problema de Otimização

O problema central do aprendizado de máxima verossimilhança com dados parcialmente observados pode ser formalizado matematicamente da seguinte forma:

Dado um conjunto de dados $\mathcal{D} = \{x^{(1)}, ..., x^{(M)}\}$, onde cada $x^{(i)}$ representa as variáveis observadas, e um modelo probabilístico $p_\theta(x, z)$ que inclui variáveis latentes $z$, o objetivo é encontrar os parâmetros $\theta$ que maximizam a log-verossimilhança marginal:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^M \log p_\theta(x^{(i)})
$$

onde

$$
p_\theta(x) = \int p_\theta(x, z) dz
$$

Esta formulação captura a essência do problema: maximizar a probabilidade dos dados observados, considerando todas as possíveis configurações das variáveis latentes [4].

#### Desafio da Intratabilidade

O cálculo exato da verossimilhança marginal $p_\theta(x)$ é geralmente intratável para modelos complexos, especialmente aqueles com muitas variáveis latentes ou estruturas não-lineares. Isso se deve à necessidade de integrar (ou somar, no caso discreto) sobre todas as possíveis configurações das variáveis latentes [5].

Para ilustrar, considere um modelo com variáveis latentes binárias $z \in \{0, 1\}^{30}$. O cálculo da verossimilhança marginal envolveria uma soma com $2^{30}$ termos, o que é computacionalmente proibitivo [6].

> ❗ **Ponto de Atenção**: A intratabilidade do cálculo exato da verossimilhança marginal é o principal obstáculo no aprendizado de modelos com variáveis latentes, motivando o desenvolvimento de métodos aproximados.

### Técnicas de Aproximação

Para contornar a intratabilidade do cálculo exato, várias técnicas de aproximação foram desenvolvidas:

1. **Amostragem de Monte Carlo**:
   Uma abordagem direta é aproximar a integral/soma por amostragem:
   
   $$
   p_\theta(x) \approx \frac{1}{K} \sum_{k=1}^K p_\theta(x, z^{(k)})
   $$
   
   onde $z^{(k)}$ são amostras da distribuição a priori $p(z)$. No entanto, esta abordagem pode ser ineficiente para modelos complexos [7].

2. **Amostragem por Importância**:
   Melhora a eficiência da amostragem usando uma distribuição proposta $q(z)$:
   
   $$
   p_\theta(x) \approx \frac{1}{K} \sum_{k=1}^K \frac{p_\theta(x, z^{(k)})}{q(z^{(k)})}
   $$
   
   A escolha de $q(z)$ é crucial para a eficácia do método [8].

3. **Limite Inferior de Evidência (ELBO)**:
   Fornece um limite inferior tratável para a log-verossimilhança:
   
   $$
   \log p_\theta(x) \geq \mathbb{E}_{z\sim q(z)} [\log p_\theta(x, z) - \log q(z)]
   $$
   
   Este é o princípio fundamental por trás dos métodos variacionais [9].

#### Questões Técnicas/Teóricas

1. Como a escolha da distribuição proposta $q(z)$ na amostragem por importância afeta a variância do estimador da verossimilhança marginal?
2. Explique por que o ELBO fornece um limite inferior para a log-verossimilhança e como isso é útil na prática para o treinamento de modelos.

### Autoencoders Variacionais (VAEs)

Os Autoencoders Variacionais (VAEs) são uma aplicação prática e poderosa dos princípios discutidos, combinando redes neurais com inferência variacional para aprender modelos generativos complexos [10].

#### Arquitetura do VAE

Um VAE consiste em dois componentes principais:

1. **Encoder** (rede de reconhecimento): $q_\phi(z|x)$ - Aproxima a distribuição posterior das variáveis latentes.
2. **Decoder** (rede generativa): $p_\theta(x|z)$ - Modela a distribuição dos dados condicionada às variáveis latentes.

A função objetivo do VAE é o ELBO:

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$

onde $p(z)$ é uma distribuição prior sobre as variáveis latentes, geralmente uma Gaussiana padrão [11].

#### Truque de Reparametrização

Para permitir a retropropagação através da amostragem estocástica, os VAEs empregam o truque de reparametrização. Para uma distribuição Gaussiana:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Isso permite que os gradientes fluam através da operação de amostragem [12].

> ✔️ **Ponto de Destaque**: O truque de reparametrização é crucial para o treinamento eficiente de VAEs, permitindo a otimização conjunta do encoder e decoder via retropropagação.

### Implementação em PyTorch

Aqui está um exemplo simplificado de implementação de um VAE em PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # mu
        self.fc22 = nn.Linear(400, latent_dim)  # logvar
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

Este código implementa um VAE básico para dados de imagem (por exemplo, MNIST), demonstrando os componentes essenciais: encoder, decoder, reparametrização e função de perda [13].

#### Questões Técnicas/Teóricas

1. Como o balanceamento entre o termo de reconstrução e o termo KL na função de perda do VAE afeta o aprendizado do modelo?
2. Explique como o truque de reparametrização permite a retropropagação através da operação de amostragem no VAE.

### Conclusão

O aprendizado de máxima verossimilhança com dados parcialmente observados é um desafio fundamental em modelos probabilísticos e aprendizado de máquina. A intratabilidade do cálculo exato da verossimilhança marginal levou ao desenvolvimento de métodos aproximados poderosos, como inferência variacional e autoencoders variacionais.

Esses métodos não apenas permitem o treinamento eficiente de modelos complexos com variáveis latentes, mas também fornecem insights valiosos sobre a estrutura dos dados e os processos generativos subjacentes. À medida que o campo avança, esperamos ver aplicações cada vez mais sofisticadas dessas técnicas em áreas como geração de imagens, processamento de linguagem natural e modelagem de séries temporais.

### Questões Avançadas

1. Compare e contraste as abordagens de Amostragem de Monte Carlo, Amostragem por Importância e Inferência Variacional para estimar a verossimilhança marginal em modelos com variáveis latentes. Quais são as vantagens e desvantagens de cada método em termos de precisão, eficiência computacional e escalabilidade?

2. Considere um cenário onde você está trabalhando com um conjunto de dados de imagens médicas parcialmente rotuladas, onde algumas imagens têm diagnósticos completos e outras não. Como você adaptaria a arquitetura e o treinamento de um VAE para lidar com este cenário de aprendizado semi-supervisionado? Discuta as modificações necessárias na função objetivo e na arquitetura do modelo.

3. Os VAEs tradicionais às vezes produzem amostras borradas ou de baixa qualidade. Proponha e discuta extensões ou modificações ao framework do VAE que poderiam melhorar a qualidade das amostras geradas, considerando avanços recentes na literatura de modelos generativos profundos.

### Referências

[1] "Uma central goal of deep learning is to discover representations of data that are useful for one or more subsequent applications." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[2] "Likelihood function p
θ 
(x) for Partially Observed Data is hard to compute:
p
θ 
(x) = 
X
All values of z
p
θ 
(x, z) =
 |Z| 
X
z∈Z
1
|Z| 
p
θ 
(x, z) = |Z|E
z∼Uniform(Z) 
[p
θ 
(x, z)]" (Trecho de cs236_lecture5.pdf)

[3] "Variational inference: pick ϕ so that q(z; ϕ) is as close as possible to
p(z|x; θ)." (Trecho de cs236_lecture5.pdf)

[4] "Maximum likelihood learning:
ln p(D|w) =
N
∑
n=1
L
n 
+
N
∑
n=1
KL (q
n
(z
n
)‖p(z
n
|x
n
, w))" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[5] "Evaluating log 
P
z 
p(x, z; θ) can be intractable. Suppose we have 30 binary
latent features, z ∈ {0, 1}
30
. Evaluating

P
z 
p(x, z; θ) involves a sum with
2
30 
terms." (Trecho de cs236_lecture5.pdf)

[6] "For continuous variables, log

R
z 
p(x, z; θ)dz is often intractable." (Trecho de cs236_lecture5.pdf)

[7] "We could try to approximate the integral over z
n 
with a
simple Monte Carlo estimator:
∫
q(z
n
|x
n
, φ) ln p(x
n
|z
n
, w) dz
n 
' 
1
L
L
∑
l=1
ln p
(x
n
|z
(l)
n 
, w)" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[8] "Importance Sampling
Likelihood function p
θ 
(x) for Partially Observed Data is hard to compute:
p
θ 
(x) = 
X
All possible values of z
p
θ 
(x, z) =

X
z∈Z
q(z)
q(z) 
p
θ 
(x, z) = E
z∼q(z)

p
θ 
(x, z)
q(z)
" (Trecho de cs236_lecture5.pdf)

[9] "Evidence lower bound (ELBO) holds for any q
log p(x; θ) ≥ 
X
z
q(z) log

p
θ
(x, z)
q(z)
" (Trecho de cs236_lecture5.pdf)

[10] "The variational autoencoder
, or
VAE (Kingma and Welling, 2013; Rezende, Mohamed, and Wierstra, 2014; Doer-
sch, 2016; Kingma and Welling, 2019) instead works with an approximation to this
likelihood when training the model." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[11] "A typical choice for the encoder is a Gaussian distribution with a diagonal co-
variance matrix whose mean and variance parameters, μ
j 
and σ