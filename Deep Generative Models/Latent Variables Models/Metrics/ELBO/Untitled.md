## Derivação e Interpretação do ELBO (Evidence Lower BOund)

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821181924998.png" alt="image-20240821181924998" style="zoom: 80%;" />

<image: Uma representação visual da desigualdade de Jensen, mostrando uma função côncava f(x) e uma corda secante entre dois pontos, ilustrando como a função está sempre acima da corda.>

### Introdução

O **Evidence Lower BOund (ELBO)** é um conceito fundamental em inferência variacional e aprendizado de máquina probabilístico, especialmente no contexto de modelos latentes e autoencoders variacionais (VAEs) [1][2]. O ELBO fornece uma aproximação tratável da log-verossimilhança marginal, que é frequentemente intratável em modelos complexos. Este resumo explorará a derivação matemática do ELBO, sua interpretação e suas aplicações em aprendizado de máquina profundo e modelos generativos.

### Conceitos Fundamentais

| Conceito                         | Explicação                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Log-verossimilhança marginal** | A probabilidade logarítmica dos dados observados marginalizados sobre todas as variáveis latentes possíveis. Frequentemente intratável em modelos complexos. [1] |
| **Inferência variacional**       | Técnica que aproxima distribuições posteriores intratáveis usando uma família de distribuições mais simples. [2] |
| **Desigualdade de Jensen**       | Propriedade fundamental de funções convexas/côncavas, crucial para a derivação do ELBO. [3] |

> ✔️ **Ponto de Destaque**: O ELBO é simultaneamente um limite inferior da log-verossimilhança marginal e o negativo da divergência KL entre a distribuição variacional e a verdadeira posterior.

### Derivação Matemática do ELBO

A derivação do ELBO começa com a log-verossimilhança marginal para um modelo com variáveis observadas $x$ e latentes $z$:

$$
\log p(x) = \log \int p(x, z) dz
$$

Introduzimos uma distribuição variacional $q(z)$ e aplicamos a desigualdade de Jensen [3]:

$$
\log p(x) = \log \int p(x, z) dz = \log \int q(z) \frac{p(x, z)}{q(z)} dz \geq \int q(z) \log \frac{p(x, z)}{q(z)} dz
$$

Esta última expressão é o ELBO, que podemos expandir:

$$
\text{ELBO} = \mathbb{E}_{q(z)}[\log p(x, z)] - \mathbb{E}_{q(z)}[\log q(z)]
$$

<image: Um diagrama mostrando a relação entre a log-verossimilhança verdadeira, o ELBO, e a divergência KL, com setas indicando como o ELBO é um limite inferior.>

> ⚠️ **Nota Importante**: A igualdade no ELBO é alcançada quando $q(z) = p(z|x)$, ou seja, quando a distribuição variacional coincide com a verdadeira posterior.

### Interpretação Variacional

O ELBO pode ser interpretado de várias maneiras:

1. **Limite Inferior**: 
   
   $$\log p(x) \geq \text{ELBO}$$
   
   Isto fornece uma aproximação tratável da log-verossimilhança marginal. [4]

2. **Decomposição KL**:
   
   $$\log p(x) = \text{ELBO} + D_{KL}(q(z)||p(z|x))$$
   
   Onde $D_{KL}$ é a divergência Kullback-Leibler. Maximizar o ELBO é equivalente a minimizar esta divergência KL. [5]

3. **Energia Livre Variacional**:
   
   O ELBO é o negativo da energia livre variacional na física estatística.

> ❗ **Ponto de Atenção**: A escolha da família de distribuições para $q(z)$ é crucial e impacta diretamente a qualidade da aproximação.

#### Questões Técnicas/Teóricas

1. Como a escolha da distribuição variacional $q(z)$ afeta a qualidade da aproximação do ELBO à verdadeira log-verossimilhança?
2. Explique por que o ELBO é particularmente útil em modelos com variáveis latentes como VAEs.

### Aplicações em Modelos Generativos Profundos

O ELBO é especialmente relevante para Autoencoders Variacionais (VAEs) e outros modelos generativos profundos [6]. Em VAEs, o ELBO é usado como função objetivo:

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x)||p(z))
$$

Onde:
- $q_\phi(z|x)$ é o encoder (distribuição variacional)
- $p_\theta(x|z)$ é o decoder
- $p(z)$ é a prior sobre as variáveis latentes

<image: Um diagrama de um VAE, mostrando o fluxo de dados através do encoder e decoder, com anotações indicando onde o ELBO é aplicado.>

A otimização do ELBO em VAEs leva a:

1. Maximização da verossimilhança dos dados reconstruídos
2. Minimização da divergência KL entre a distribuição latente e a prior

> ✔️ **Ponto de Destaque**: O ELBO em VAEs equilibra a qualidade da reconstrução com a regularização do espaço latente.

### Otimização do ELBO

A otimização do ELBO geralmente envolve técnicas de gradiente estocástico. Um desafio é estimar gradientes através de variáveis aleatórias. Algumas técnicas incluem:

1. **Reparametrização**: Usada em VAEs para permitir a propagação do gradiente através do sampling [7].

   $$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

2. **REINFORCE**: Método de estimativa de gradiente baseado em amostragem para variáveis discretas.

3. **Gradientes Retos (Straight-Through Estimators)**: Aproximações de gradiente para funções não-diferenciáveis.

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def elbo_loss(self, x_recon, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

# Uso
vae = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters())

# Treinamento
for epoch in range(num_epochs):
    for batch in dataloader:
        x_recon, mu, logvar = vae(batch)
        loss = vae.elbo_loss(x_recon, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Este código implementa um VAE básico, demonstrando como o ELBO é usado na prática para treinar modelos generativos profundos.

#### Questões Técnicas/Teóricas

1. Como o trick de reparametrização ajuda na otimização do ELBO em VAEs?
2. Discuta as diferenças entre otimizar o ELBO e otimizar diretamente a log-verossimilhança em modelos complexos.

### Extensões e Variantes do ELBO

1. **β-VAE**: Introduz um hiperparâmetro β para controlar o trade-off entre reconstrução e regularização [8].

   $$\mathcal{L}_{\beta} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x)||p(z))$$

2. **IWAE (Importance Weighted Autoencoder)**: Usa múltiplas amostras para uma estimativa mais precisa da log-verossimilhança [9].

   $$\mathcal{L}_K = \mathbb{E}_{z_1,...,z_K \sim q_\phi(z|x)}\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(x,z_k)}{q_\phi(z_k|x)}\right]$$

3. **VampPrior**: Substitui a prior padrão por uma mistura de posteriors aprendidas, melhorando a expressividade do modelo [10].

> ❗ **Ponto de Atenção**: Estas variantes do ELBO oferecem diferentes trade-offs entre complexidade computacional, qualidade da aproximação e interpretabilidade do modelo.

### Conclusão

O Evidence Lower BOund (ELBO) é uma ferramenta matemática poderosa que forma a base de muitos métodos modernos em aprendizado de máquina probabilístico e modelos generativos profundos. Sua derivação através da desigualdade de Jensen e sua interpretação variacional fornecem insights profundos sobre a relação entre inferência aproximada e otimização de modelos.

A aplicação do ELBO em Autoencoders Variacionais e suas extensões demonstra sua versatilidade e importância na prática. Compreender o ELBO é crucial para desenvolver e otimizar modelos generativos avançados, bem como para interpretar seus resultados de forma significativa.

À medida que o campo de modelos generativos continua a evoluir, é provável que vejamos novas variantes e aplicações do ELBO, potencialmente levando a avanços significativos em áreas como geração de imagens, processamento de linguagem natural e modelagem de séries temporais.

### Questões Avançadas

1. Como você modificaria o ELBO para lidar com dados parcialmente observados em um modelo generativo profundo?

2. Discuta as implicações teóricas e práticas de usar uma distribuição variacional que é mais complexa que a verdadeira posterior em termos do ELBO e da qualidade do modelo resultante.

3. Proponha uma extensão do ELBO que poderia ser particularmente útil para modelos generativos hierárquicos, justificando matematicamente sua abordagem.

### Referências

[1] "Likelihood function p
θ 
(x) for Partially Observed Data is hard to compute:" (Trecho de cs236_lecture5.pdf)

[2] "Variational inference: pick ϕ so that q(z; ϕ) is as close as possible to p(z|x; θ)." (Trecho de cs236_lecture5.pdf)

[3] "Idea: use Jensen Inequality (for concave functions)" (Trecho de cs236_lecture5.pdf)

[4] "log p(x; θ) ≥ 
X
z
q(z) log

p
θ
(x, z)
q(z)

= 
X
z
q(z) log p
θ
(x, z) − 
X
z
q(z) log q(z)" (Trecho de cs236_lecture5.pdf)

[5] "In general, log p(x; θ) = ELBO + D
KL
(q(z)∥p(z|x; θ)). The closer q(z) is to p(z|x; θ), the closer the ELBO is to the true log-likelihood" (Trecho de cs236_lecture5.pdf)

[6] "Variational Autoencoder Marginal Likelihood" (Trecho de cs236_lecture5.pdf)

[7] "Reparameterization trick replaces a direct sample of z by one that is calculated from a sample of an independent random variable , thereby allowing the error signal to be backpropagated to the encoder network." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[8] "β-VAE: Introduz um hiperparâmetro β para controlar o trade-off entre reconstrução e regularização" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[9] "IWAE (Importance Weighted Autoencoder): Usa múltiplas amostras para uma estimativa mais precisa da log-verossimilhança" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[10] "VampPrior: Substitui a prior padrão por uma mistura de posteriors aprendidas, melhorando a expressividade do modelo" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)