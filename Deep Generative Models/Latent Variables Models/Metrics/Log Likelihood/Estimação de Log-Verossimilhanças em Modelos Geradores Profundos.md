## Estimação de Log-Verossimilhanças em Modelos Geradores Profundos

![image-20240821181623164](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821181623164.png)

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240826113245454.png" alt="image-20240826113245454" style="zoom:80%;" />

### Introdução

A estimação de log-verossimilhanças é um desafio fundamental na aprendizagem de modelos geradores profundos, especialmente em cenários com variáveis latentes. Este resumo explorará os métodos e desafios associados à estimação não-enviesada de log-verossimilhanças, com foco particular em técnicas de amostragem e aproximações variacionais. Abordaremos desde conceitos básicos até métodos avançados, proporcionando uma compreensão profunda das complexidades envolvidas na avaliação e treinamento de modelos geradores modernos [19].

### Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Log-verossimilhança**       | ==Medida fundamental da qualidade do ajuste do modelo aos dados, expressa como $\log p(x;\theta)$ para dados observados $x$ e parâmetros do modelo $\theta$. [15]== |
| **Variáveis Latentes**        | Variáveis não observadas que capturam estrutura oculta nos dados, comumente denotadas como $z$ em modelos geradores. [15] |
| **Amostragem de Monte Carlo** | ==Técnica para aproximar integrais intratáveis através de amostras aleatórias. [18]== |
| **Lower Bound (ELBO)**        | ==Aproximação tratável da log-verossimilhança==, fundamental para inferência variacional. [22] |

> ⚠️ **Nota Importante**: A estimação precisa de log-verossimilhanças é crucial para a avaliação e comparação de modelos geradores, mas apresenta desafios significativos em modelos com variáveis latentes complexas.

### Desafios na Estimação de Log-Verossimilhanças

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240826114105771.png" alt="image-20240826114105771" style="zoom: 67%;" />

A estimação de log-verossimilhanças em modelos geradores profundos enfrenta vários desafios, principalmente devido à presença de variáveis latentes e à complexidade das distribuições envolvidas [15][19].

1. **Intratabilidade da Integral**

   A log-verossimilhança para modelos com variáveis latentes é dada por:

   $$
   \log p(x;\theta) = \log \int p(x,z;\theta)dz
   $$

   Esta integral é geralmente intratável analiticamente para modelos complexos [15].

2. **Alta Dimensionalidade**

   Em modelos profundos, o espaço latente pode ser altamente dimensional, tornando a integração numérica direta impraticável [19].

3. **Variância de Estimadores**

   ==Estimadores baseados em amostragem podem sofrer de alta variância, especialmente quando as distribuições envolvidas têm caudas pesadas ou são multimodais [18].==

#### Questões Técnicas/Teóricas

1. Por que a estimação direta da log-verossimilhança é problemática em modelos com variáveis latentes? Explique matematicamente.
2. Como a dimensionalidade do espaço latente afeta a dificuldade de estimação da log-verossimilhança?

### Métodos de Estimação

#### 1. Amostragem Ingênua de Monte Carlo

A abordagem mais direta para estimar a log-verossimilhança é através de amostragem uniforme:

$$
p_\theta(x) = \sum_{z \in Z} p_\theta(x,z) = |Z|E_{z\sim \text{Uniform}(Z)}[p_\theta(x,z)]
$$

Esta estimativa pode ser aproximada por:

$$
\sum_{z} p_\theta(x,z) \approx |Z| \frac{1}{k}\sum_{j=1}^k p_\theta(x,z^{(j)})
$$

onde $z^{(j)}$ são amostras uniformes do espaço latente [18].

> ❗ **Ponto de Atenção**: ==Este método é ineficiente para espaços latentes grandes ou complexos, pois a maioria das amostras terá baixa probabilidade.==

#### 2. Amostragem por Importância

Uma melhoria sobre a amostragem uniforme é a ==amostragem por importância:==

$$
p_\theta(x) = \sum_{z \in Z} \frac{q(z)}{q(z)} p_\theta(x,z) = E_{z\sim q(z)}\left[\frac{p_\theta(x,z)}{q(z)}\right]
$$

A estimativa de Monte Carlo correspondente é:

$$
p_\theta(x) \approx \frac{1}{k}\sum_{j=1}^k \frac{p_\theta(x,z^{(j)})}{q(z^{(j)})}
$$

onde $z^{(j)}$ são amostras de $q(z)$ [19].

> ✔️ **Ponto de Destaque**: ==A escolha de $q(z)$ é crítica para a eficiência do estimador. Idealmente, $q(z)$ deve ser próxima da distribuição posterior $p(z|x;\theta)$.==

#### 3. Estimação do Logaritmo

Para estimar $\log(p_\theta(x))$, poderíamos usar:

$$
\log(p_\theta(x)) \approx \log\left(\frac{1}{k}\sum_{j=1}^k \frac{p_\theta(x,z^{(j)})}{q(z^{(j)})}\right)
$$

No entanto, ==este estimador é enviesado devido à não-linearidade do logaritmo [20].==

> ⚠️ **Nota Importante**: $E_{z^{(1)}\sim q(z)}[\log(\frac{p_\theta(x,z^{(1)})}{q(z^{(1)})})] \neq \log(E_{z^{(1)}\sim q(z)}[\frac{p_\theta(x,z^{(1)})}{q(z^{(1)})}])$

### Evidence Lower Bound (ELBO)

Para contornar o problema do viés na estimação do logaritmo, introduzimos o Evidence Lower Bound (ELBO):

$$
\log p(x;\theta) \geq \sum_z q(z) \log \frac{p_\theta(x,z)}{q(z)} = \text{ELBO}
$$

==Esta desigualdade é derivada da desigualdade de Jensen, aproveitando a concavidade da função logarítmica [21][22].==

O ELBO pode ser decomposto em:

$$
\text{ELBO} = \sum_z q(z) \log p_\theta(x,z) - \sum_z q(z) \log q(z)
$$

onde o ==segundo termo é a entropia de $q(z)$ [23].==

> ✔️ **Ponto de Destaque**: ==O ELBO é igual à log-verossimilhança quando $q(z) = p(z|x;\theta)$, tornando-o uma aproximação poderosa e tratável.==

#### Questões Técnicas/Teóricas

1. Como o ELBO se relaciona com a divergência KL entre $q(z)$ e a verdadeira posterior $p(z|x;\theta)$?
2. Por que o ELBO é preferível como função objetivo em relação à estimativa direta da log-verossimilhança em muitos cenários de treinamento?

### Inferência Variacional

==A inferência variacional busca otimizar o ELBO em relação a uma família de distribuições $q(z;\phi)$, parametrizada por $\phi$ [25]:==
$$
\phi^* = \arg\max_\phi \text{ELBO}(\theta, \phi)
$$

Isto é ==equivalente a minimizar a divergência KL entre $q(z;\phi)$ e $p(z|x;\theta)$:==

$$
D_{KL}(q(z;\phi) || p(z|x;\theta)) = \log p(x;\theta) - \text{ELBO}(\theta, \phi)
$$

> ❗ **Ponto de Atenção**: A escolha da família variacional $q(z;\phi)$ é crucial e deve equilibrar flexibilidade e tratabilidade computacional.

### Reparametrização para Gradientes de Baixa Variância

Para treinar modelos variacionais eficientemente, ==é crucial obter estimativas de gradiente de baixa variância. A técnica de reparametrização é fundamental para isso [26]:==

1. Expressamos $z$ como uma função determinística de uma variável aleatória auxiliar $\epsilon$ e os parâmetros $\phi$:

   $$
   z = g(\epsilon, \phi), \quad \epsilon \sim p(\epsilon)
   $$

2. Para uma distribuição Gaussiana, isto se torna:

   $$
   z = \mu(\phi) + \sigma(\phi) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
   $$

3. Isto permite reescrever expectativas em relação a $q(z;\phi)$ como:

   $$
   E_{q(z;\phi)}[f(z)] = E_{p(\epsilon)}[f(g(\epsilon, \phi))]
   $$

> ✔️ **Ponto de Destaque**: ==A reparametrização permite o cálculo de gradientes através de amostragem==, crucial para o treinamento de autoencoders variacionais (VAEs) e outros modelos geradores profundos.

### Implementação em PyTorch

Aqui está um exemplo simplificado de como implementar a estimação do ELBO para um VAE em PyTorch:

```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
        
    def elbo(self, x, x_recon, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

# Uso
model = VAE(latent_dim=20)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in data_loader:
        x_recon, mu, logvar = model(batch)
        loss = -model.elbo(batch, x_recon, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Este código implementa um VAE básico, demonstrando como o ELBO é calculado e otimizado na prática [27].

#### Questões Técnicas/Teóricas

1. Como a técnica de reparametrização é implementada no código acima e por que ela é crucial para o treinamento eficiente do VAE?
2. Explique como o termo KL no ELBO é calculado para uma distribuição Gaussiana no espaço latente.

### Conclusão

A estimação precisa de log-verossimilhanças em modelos geradores profundos permanece um desafio significativo, especialmente para modelos com estruturas latentes complexas. Métodos como amostragem por importância e inferência variacional, juntamente com técnicas como a reparametrização, fornecem ferramentas poderosas para abordar este problema. ==O ELBO emerge como uma aproximação chave, permitindo a otimização tratável de modelos complexos.==

A compreensão profunda destes métodos e seus desafios associados é crucial para o desenvolvimento e avaliação eficazes de modelos geradores modernos em aprendizado de máquina e inteligência artificial [28].

### Questões Avançadas

1. Compare e contraste as vantagens e desvantagens da estimação de log-verossimilhança via ELBO versus métodos de amostragem por importância aninhada (IWAE). Quando você escolheria um sobre o outro?

2. Discuta as implicações da "maldição da dimensionalidade" na estimação de log-verossimilhanças para modelos com espaços latentes de alta dimensão. Como isso afeta a escolha de arquiteturas de modelo e métodos de estimação?

3. Explique como o conceito de "amortização" na inferência variacional (como usado em VAEs) afeta a qualidade das estimativas de log-verossimilhança e o tradeoff entre precisão e eficiência computacional.

4. Proponha e discuta uma abordagem para melhorar a estimação de log-verossimilhanças em cenários onde a distribuição posterior verdadeira é altamente multimodal, um cenário comum em modelos geradores complexos.

5. Analise criticamente o uso de log-verossimilhanças estimadas como métrica de avaliação para modelos geradores. Quais são as limitações desta abordagem e que métricas alternativas ou complementares você sugeriria para uma avaliação mais abrangente?

### Referências

[15] "Likelihood function p θ (x) for Partially Observed Data is hard to compute:
p θ (x) = X All possible values of z p θ (x, z) = X z∈Z q(z) q(z) p θ (x, z) = E z∼q(z)  p θ (x, z) q(z)" (Trecho de cs236_lecture5.pdf)

[18] "We can think of it as an (intractable) expectation. Monte Carlo to the rescue:
1 Sample z (1) , · · · , z (k) uniformly at random
2 Approximate expectation with sample average
X z p θ (x, z) ≈ |Z| 1 k k X j=1 p θ (x, z (j)