## Evidência Lower Bound (ELBO), Derivação e Otimização em Modelos de Variáveis Latentes

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240826143714331.png" alt="image-20240826143714331" style="zoom: 60%;" />

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240826143759086.png" alt="image-20240826143759086" style="zoom:80%;" />

### Introdução

A Evidência Lower Bound (ELBO) é um conceito fundamental na inferência variacional e no treinamento de modelos de variáveis latentes, como os Variational Autoencoders (VAEs). ==O ELBO fornece um limite inferior tratável para a log-verossimilhança dos dados, que é muitas vezes intratável de se calcular diretamente [1].== Esta técnica é crucial para otimizar modelos probabilísticos complexos, ==permitindo a aprendizagem eficiente de distribuições posteriores aproximadas sobre variáveis latentes.==

Neste resumo, exploraremos a derivação detalhada do ELBO, sua interpretação matemática e estatística, e como ele é utilizado na prática para treinar modelos de variáveis latentes. Abordaremos também as nuances da otimização do ELBO e suas implicações para a qualidade dos modelos resultantes.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Log-verossimilhança**      | A log-verossimilhança $\log p(x)$ é uma medida da qualidade do modelo, representando a probabilidade logarítmica dos dados observados sob o modelo [2]. |
| **Variáveis Latentes**       | Variáveis não observadas $z$ que o modelo usa para capturar a estrutura subjacente dos dados [3]. |
| **Inferência Variacional**   | Técnica para aproximar distribuições posteriores intratáveis usando otimização [4]. |
| **Distribuição Variacional** | Uma distribuição $q_\phi(z|x)$ que aproxima a verdadeira posterior $p(z|x)$ [5]. |

> ✔️ **Ponto de Destaque**: ==O ELBO é fundamentalmente uma reformulação da log-verossimilhança que introduz uma distribuição variacional==, tornando o problema de inferência tratável computacionalmente.

### Derivação do ELBO

A derivação do ELBO começa com a log-verossimilhança dos dados observados $x$:

$$
\log p(x) = \log \int p(x,z) dz
$$

Onde $p(x,z)$ é a distribuição conjunta dos dados observados $x$ e das variáveis latentes $z$ [6].

Introduzimos a distribuição variacional $q_\phi(z|x)$:

$$
\log p(x) = \log \int p(x,z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz
$$

Aplicando a desigualdade de Jensen [7]:

$$
\log p(x) \geq \int q_\phi(z|x) \log \frac{p(x,z)}{q_\phi(z|x)} dz
$$

Esta desigualdade define o ELBO:

$$
\text{ELBO}(x; \phi) = \mathbb{E}_{q_\phi(z|x)} \left[\log \frac{p(x,z)}{q_\phi(z|x)}\right]
$$

Expandindo a fração dentro do logaritmo:

$$
\text{ELBO}(x; \phi) = \mathbb{E}_{q_\phi(z|x)} [\log p(x|z)] - \text{KL}(q_\phi(z|x) || p(z))
$$

Onde $\text{KL}(q_\phi(z|x) || p(z))$ é a divergência de Kullback-Leibler entre a distribuição variacional e a prior sobre $z$ [8].

> ❗ **Ponto de Atenção**: A decomposição do ELBO em um termo de reconstrução e um termo de regularização KL é crucial para entender seu comportamento durante a otimização.

#### Questões Técnicas/Teóricas

1. Como a desigualdade de Jensen é aplicada na derivação do ELBO e qual é sua interpretação geométrica neste contexto?
2. Explique como o termo de divergência KL no ELBO atua como um regularizador durante o treinamento de um VAE.

### Interpretação e Significado do ELBO

O ELBO pode ser interpretado de várias maneiras:

1. **Limite Inferior**: ==O ELBO fornece um limite inferior para a log-verossimilhança $\log p(x)$ [9].==

2. **Balanço Reconstrução-Regularização**: O primeiro termo $\mathbb{E}_{q_\phi(z|x)} [\log p(x|z)]$ incentiva uma boa reconstrução dos dados, enquanto o termo KL age como regularizador [10].

3. **Minimização da Divergência**: Maximizar o ELBO é equivalente a minimizar a divergência KL entre a distribuição variacional e a verdadeira posterior $p(z|x)$ [11].

<image: Um diagrama de Venn mostrando a relação entre a verdadeira posterior p(z|x), a distribuição variacional q_φ(z|x), e como a maximização do ELBO minimiza a divergência entre elas.>

> ⚠️ **Nota Importante**: A maximização do ELBO não garante que a distribuição variacional convergirá exatamente para a verdadeira posterior, mas fornece a melhor aproximação possível dentro da família de distribuições escolhida.

### Otimização do ELBO

==A otimização do ELBO é geralmente realizada usando técnicas de gradiente estocástico.== O desafio principal está em ==estimar o gradiente do ELBO com respeito aos parâmetros do modelo $\theta$ e os parâmetros variacionais $\phi$ [12].==

Um método popular é o estimador reparametrizado, introduzido no contexto dos VAEs [13]:

$$
\nabla_{\phi,\theta} \text{ELBO} \approx \frac{1}{L} \sum_{l=1}^L \nabla_{\phi,\theta} [\log p(x|z^{(l)}) + \log p(z^{(l)}) - \log q_\phi(z^{(l)}|x)]
$$

Onde $z^{(l)} = g_\phi(\epsilon^{(l)}, x)$ e $\epsilon^{(l)} \sim p(\epsilon)$ [14].

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
    
    def elbo(self, x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div
```

Este código implementa um VAE básico e calcula o ELBO, demonstrando como a reparametrização é usada na prática [15].

#### Questões Técnicas/Teóricas

1. Como o truque de reparametrização ajuda na estimativa do gradiente do ELBO, e por que isso é importante para o treinamento de VAEs?
2. Discuta as vantagens e desvantagens de usar uma distribuição variacional com diagonal de covariância versus uma covariância completa no contexto da otimização do ELBO.

### Desafios e Considerações Práticas

1. **Gap de Amortização**: Em VAEs, o uso de um codificador amortizado pode levar a um gap entre o ELBO e a verdadeira log-verossimilhança [16].

2. **Colapso Posterior**: Em alguns casos, o termo KL pode dominar, levando a um fenômeno conhecido como colapso posterior [17].

3. **Escolha da Família Variacional**: A escolha da família de distribuições para $q_\phi(z|x)$ afeta significativamente a qualidade da aproximação [18].

| 👍 Vantagens                                | 👎 Desvantagens                            |
| ------------------------------------------ | ----------------------------------------- |
| Permite inferência em modelos complexos    | Pode levar a aproximações subótimas       |
| Fornece um objetivo de otimização tratável | Sensível à escolha da família variacional |
| Balanceia reconstrução e regularização     | Pode sofrer de colapso posterior          |

### Extensões e Variantes do ELBO

1. **β-VAE**: Introduz um hiperparâmetro $\beta$ para controlar o termo KL [19]:

   $$
   \text{ELBO}_\beta = \mathbb{E}_{q_\phi(z|x)} [\log p(x|z)] - \beta \cdot \text{KL}(q_\phi(z|x) || p(z))
   $$

2. **Importance Weighted Autoencoder (IWAE)**: Usa amostragem por importância para obter um limite inferior mais apertado [20]:

   $$
   \log p(x) \geq \mathbb{E}_{z_1,...,z_K \sim q_\phi(z|x)} \left[\log \frac{1}{K} \sum_{k=1}^K \frac{p(x,z_k)}{q_\phi(z_k|x)}\right]
   $$

3. **Variational Inference with Monte Carlo Objectives (VIMCO)**: Estende o IWAE para modelos com variáveis latentes discretas [21].

> ✔️ **Ponto de Destaque**: Estas extensões visam abordar limitações específicas do ELBO padrão, como o trade-off entre reconstrução e regularização ou a qualidade da estimativa da log-verossimilhança.

### Conclusão

O ELBO é uma ferramenta poderosa e versátil na inferência variacional e no treinamento de modelos de variáveis latentes. Sua derivação e otimização fornecem insights profundos sobre o comportamento desses modelos e as compensações envolvidas na aproximação de distribuições posteriores complexas.

Compreender o ELBO é fundamental para avançar no campo de modelos generativos profundos e inferência probabilística. As extensões e variantes do ELBO continuam a ser uma área ativa de pesquisa, prometendo melhorias na qualidade dos modelos e na eficiência computacional.

### Questões Avançadas

1. Como você modificaria o ELBO para incorporar conhecimento prévio sobre a estrutura das variáveis latentes em um domínio específico?

2. Discuta as implicações teóricas e práticas de usar o ELBO versus outros critérios de otimização (como a log-verossimilhança exata) em modelos hierárquicos profundos.

3. Proponha e justifique uma nova variante do ELBO que poderia potencialmente superar as limitações das abordagens existentes em cenários de dados esparsos ou de alta dimensionalidade.

### Referências

[1] "A natural question is whether it would be better to use a distribution defined on the hypersphere." (Trecho de Latent Variable Models.pdf)

[2] "The logarithm of the marginal distribution could be approximated as follows:" (Trecho de Latent Variable Models.pdf)

[3] "Let us consider a family of variational distributions parameterized by φ, {q_φ(z)}φ." (Trecho de Latent Variable Models.pdf)

[4] "We know the form of these distributions, and we assume that they assign non-zero probability mass to all z ∈ Z^M." (Trecho de Latent Variable Models.pdf)

[5] "Then, the logarithm of the marginal distribution could be approximated as follows:" (Trecho de Latent Variable Models.pdf)

[6] "ln p(x) = ln ∫ p(x|z)p(z) dz" (Trecho de Latent Variable Models.pdf)

[7] "In the fourth line we used Jensen's inequality." (Trecho de Latent Variable Models.pdf)

[8] "The second part of the ELBO, E_z~q_φ(z|x)[ln q_φ(z|x) − ln p(z)], could be seen as a regularizer and it coincides with the Kullback–Leibler (KL) divergence." (Trecho de Latent Variable Models.pdf)

[9] "As a result, we obtain an auto-encoder-like model, with a stochastic encoder, q_φ(z|x), and a stochastic decoder, p(x|z)." (Trecho de Latent Variable Models.pdf)

[10] "The first part of the ELBO, E_z~q_φ(z|x) [ln p(x|z)], is referred to as the (negative) reconstruction error, because x is encoded to z and then decoded back." (Trecho de Latent Variable Models.pdf)

[11] "The second part of the ELBO, E_z~q_φ(z|x)[ln q_φ(z|x) − ln p(z)], could be seen as a regularizer and it coincides with the Kullback–Leibler (KL) divergence." (Trecho de Latent Variable Models.pdf)

[12] "There are two questions left to get the full picture of the VAEs:" (Trecho de Latent Variable Models.pdf)

[13] "As observed by Kingma and Welling [6] and Rezende et al. [7], we can drastically reduce the variance of the gradient by using this reparameterization of the Gaussian distribution." (Trecho de Latent Variable Models.pdf)

[14] "Even better, since we learn the VAE using stochastic gradient descent, it is enough to sample z only once during training!" (Trecho de Latent Variable Models.pdf)

[15] "We went through a lot of theory and discussions, and you might think it is impossible to implement a VAE. However, it is actually simpler than it might look." (Trecho de Latent Variable Models.pdf)

[16] "As pointed out by Huszár [62], the reason for that is the inductive bias of the chosen class of models." (Trecho de Latent Variable Models.pdf)

[17] "It turns out that learning the two-level VAE is even more problematic than a VAE with a single latent because even for a relatively simple decoder the second latent variable z_2 is mostly unused [15, 70]. This effect is called the posterior collapse." (Trecho de Latent Variable Models.pdf)
