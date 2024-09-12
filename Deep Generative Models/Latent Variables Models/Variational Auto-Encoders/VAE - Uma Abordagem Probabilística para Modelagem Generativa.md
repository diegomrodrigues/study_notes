## Variational Autoencoders: Uma Abordagem Probabilística para Modelagem Generativa

<image: Um diagrama mostrando a arquitetura de um Variational Autoencoder, com um encoder neural que mapeia dados de entrada para uma distribuição no espaço latente, e um decoder neural que mapeia amostras do espaço latente de volta para o espaço de dados>

### Introdução

O Variational Autoencoder (VAE) é uma poderosa técnica de aprendizado de máquina que combina princípios de inferência variacional com redes neurais profundas para criar modelos generativos probabilísticos. Neste resumo, exploraremos em detalhes o funcionamento do VAE, sua formulação matemática e sua aplicação na modelagem de dados de alta dimensionalidade, como imagens de dígitos manuscritos do conjunto de dados MNIST [1].

### Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Modelo Latente**         | O VAE modela a distribuição de dados de alta dimensão $p_\theta(x)$ através de variáveis latentes $z$, permitindo uma representação mais compacta e semanticamente significativa dos dados [1]. |
| **Inferência Variacional** | Devido à intratabilidade do cálculo exato da distribuição posterior, o VAE utiliza uma aproximação variacional $q_\phi(z|x)$ para realizar inferência eficiente [3]. |
| **Amortização**            | O VAE emprega um único encoder neural para aproximar a distribuição posterior para todos os pontos de dados, tornando a inferência escalável para grandes conjuntos de dados [3]. |
| **Evidence Lower Bound**   | A otimização do VAE é baseada na maximização de um limite inferior da evidência (ELBO), que equilibra a qualidade da reconstrução com a regularização da distribuição latente [4]. |

> ⚠️ **Nota Importante**: O VAE não apenas aprende a comprimir e reconstruir dados, mas também a modelar a distribuição subjacente dos dados, permitindo a geração de novas amostras.

### Formulação Matemática do VAE

<image: Um gráfico mostrando a relação entre o espaço de dados X e o espaço latente Z, com setas bidirecionais representando o encoder e o decoder>

O VAE é definido por um processo generativo que mapeia variáveis latentes para o espaço de dados observáveis [1]. Matematicamente, temos:

1. **Distribuição Prévia**: 
   $$p(z) = \mathcal{N}(z | 0, I)$$

2. **Modelo Gerador (Decoder)**:
   $$p_\theta(x | z) = \text{Bern}(x | f_\theta(z))$$

Onde $f_\theta(\cdot)$ é uma rede neural que mapeia $z$ para os logits de variáveis Bernoulli que modelam os pixels da imagem.

3. **Modelo de Inferência (Encoder)**:
   $$q_\phi(z | x) = \mathcal{N}(z | \mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$$

Aqui, $\mu_\phi(x)$ e $\sigma^2_\phi(x)$ são produzidos por uma rede neural que recebe $x$ como entrada.

4. **Evidence Lower Bound (ELBO)**:
   $$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] - D_{KL}(q_\phi(z | x) || p(z))$$

O objetivo de treinamento do VAE é maximizar o ELBO, que é equivalente a minimizar:

$$-\text{ELBO} = \text{reconstruction loss} + D_{KL}(q_\phi(z | x) || p(z))$$

> ✔️ **Ponto de Destaque**: A decomposição do ELBO em um termo de reconstrução e um termo de divergência KL fornece uma interpretação intuitiva do processo de aprendizagem do VAE.

#### Questões Técnicas/Teóricas

1. Como a escolha da dimensionalidade do espaço latente $z$ afeta o desempenho e a capacidade generativa do VAE?
2. Explique como o termo de divergência KL no ELBO atua como um regularizador no treinamento do VAE.

### Implementação do VAE em PyTorch

A implementação de um VAE em PyTorch envolve a definição de duas redes neurais principais: o encoder e o decoder. Aqui está um esboço de como isso pode ser estruturado:

````python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class Encoder(nn.Module):
    # Implementação do encoder

class Decoder(nn.Module):
    # Implementação do decoder

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
````

> ❗ **Ponto de Atenção**: A implementação do "reparameterization trick" é crucial para permitir a retropropagação através da amostragem estocástica no espaço latente.

### Análise do ELBO e sua Relação com a Formulação da Aula

O ELBO, como apresentado no problema, pode ser relacionado à formulação vista em aula da seguinte maneira [4]:

1. **Formulação da Aula**:
   $$\log p(x; \theta) \geq \sum_z q(z; \phi) \log p(z, x; \theta) + H(q(z; \phi)) = \mathcal{L}(x; \theta, \phi)$$

2. **Formulação do Problema**:
   $$\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x | z)] - D_{KL}(q_\phi(z | x) || p(z))$$

A principal diferença está na introdução da distribuição amortizada $q_\phi(z|x)$, que aprende a mapear diretamente de $x$ para os parâmetros da distribuição variacional, em vez de ter parâmetros $\phi$ separados para cada ponto de dados.

> 💡 **Insight**: A amortização da inferência permite que o VAE escale eficientemente para grandes conjuntos de dados, aprendendo uma função de inferência global em vez de parâmetros específicos para cada exemplo.

#### Questões Técnicas/Teóricas

1. Como a amortização da inferência afeta o trade-off entre eficiência computacional e qualidade da aproximação posterior no VAE?
2. Discuta as implicações da decomposição do ELBO em termos de reconstrução e regularização para o treinamento de VAEs.

### Conclusão

O Variational Autoencoder representa uma abordagem poderosa e flexível para modelagem generativa, combinando princípios de inferência variacional com a expressividade de redes neurais profundas. Através da otimização do ELBO, o VAE aprende simultaneamente a comprimir dados em um espaço latente significativo e a gerar novas amostras plausíveis. A formulação matemática e a implementação prática do VAE oferecem insights valiosos sobre a interseção entre aprendizado profundo e modelagem probabilística.

### Questões Avançadas

1. Como o VAE poderia ser estendido para lidar com dados sequenciais ou temporais, e quais modificações seriam necessárias na arquitetura e na função objetivo?

2. Discuta as limitações do VAE em termos de qualidade de amostras geradas em comparação com outros modelos generativos, como GANs, e proponha possíveis abordagens para mitigar essas limitações.

3. Explique como o conceito de "information bottleneck" se relaciona com a arquitetura e o treinamento do VAE, e como isso poderia ser explorado para melhorar a representação latente aprendida.

### Referências

[1] "For this problem we will be using PyTorch to implement the variational autoencoder (VAE) and learn a probabilistic model of the MNIST dataset of handwritten digits. Formally, we observe a sequence of binary pixels x ∈ {0, 1}d, and let z ∈ Rk denote a set of latent variables. Our goal is to learn a latent variable model pθ(x) of the high-dimensional data distribution pdata(x)." (Trecho de Variational Autoencoder Stanford Notes)

[2] "The VAE is a latent variable model that learns a specific parameterization pθ(x) =
R
pθ(x, z)dz =
R
p(z)pθ(x | z)dz. Specifically, the VAE is defined by the following generative process:
p(z) = N(z | 0, I)
pθ(x | z) = Bern (x | fθ (z))" (Trecho de Variational Autoencoder Stanford Notes)

[3] "Although we would like to maximize the marginal likelihood pθ(x), computation of pθ(x) =
R
p(z)pθ(x | z)dz is generally intractable as it involves integration over all possible values of z. Therefore, we posit a variational approximation to the true posterior and perform amortized inference as we have seen in class:
qϕ(z | x) = N(z | μϕ (x) , diag(σ2ϕ(x)))" (Trecho de Variational Autoencoder Stanford Notes)

[4] "We then maximize the lower bound to the marginal log-likelihood to obtain an expression known as the evidence lower bound (ELBO):
log pθ(x) ≥ ELBO(x; θ, ϕ) = Eqϕ(z|x)[log pθ(x | z)] − DKL (qϕ (z | x) || p (z))" (Trecho de Variational Autoencoder Stanford Notes)

[5] "Notice that the negatation of the ELBO decomposes into two terms: (1) the reconstruction loss: −Eqϕ(z|x)[log pθ(x | z)], and (2) the Kullback-Leibler (KL) term: DKL(qϕ(z | x) || p(z)), e.g. -ELBO = recon. loss + KL div. Hence, VAE learning objective is to minimize the negative ELBO." (Trecho de Variational Autoencoder Stanford Notes)