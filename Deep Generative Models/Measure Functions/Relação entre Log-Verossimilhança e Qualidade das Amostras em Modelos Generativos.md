## Relação entre Log-Verossimilhança e Qualidade das Amostras em Modelos Generativos

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820172648732.png" alt="image-20240820172648732" style="zoom: 50%;" />

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820172716405.png" alt="image-20240820172716405" style="zoom:50%;" />

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820172736300.png" alt="image-20240820172736300" style="zoom:50%;" />

### Introdução

A avaliação de modelos generativos é um desafio fundamental na aprendizagem de máquina. Embora a log-verossimilhança seja uma métrica amplamente utilizada, sua relação com a qualidade das amostras geradas não é sempre direta ou intuitiva. Este resumo explora as complexidades dessa relação, focando nas limitações do Maximum Likelihood Estimation (MLE) como métrica de avaliação para modelos generativos profundos [1][2].

### Conceitos Fundamentais

| Conceito                                | Explicação                                                   |
| --------------------------------------- | ------------------------------------------------------------ |
| **Log-verossimilhança**                 | Medida que quantifica quão bem um modelo probabilístico explica os dados observados. Matematicamente, é o logaritmo da probabilidade dos dados sob o modelo [3]. |
| **Maximum Likelihood Estimation (MLE)** | Método de estimação de parâmetros que maximiza a log-verossimilhança dos dados observados [4]. |
| **Qualidade das amostras**              | Avaliação subjetiva ou quantitativa da fidelidade e diversidade das amostras geradas por um modelo em relação à distribuição de dados reais [5]. |

> ⚠️ **Nota Importante**: A log-verossimilhança alta nem sempre implica em amostras de alta qualidade, e vice-versa [6].

### Relação entre Log-Verossimilhança e Qualidade das Amostras

A relação entre log-verossimilhança e qualidade das amostras em modelos generativos é complexa e muitas vezes contraintuitiva [7]. Enquanto a log-verossimilhança mede quão bem o modelo se ajusta aos dados de treinamento, a qualidade das amostras é uma medida mais subjetiva e multifacetada.

#### 👍Vantagens da Log-Verossimilhança
* Métrica objetiva e matematicamente bem fundamentada [8]
* Fácil de calcular para muitos modelos probabilísticos [9]

#### 👎Desvantagens da Log-Verossimilhança
* Pode ser enganosa para modelos com alta dimensionalidade [10]
* Não captura diretamente aspectos qualitativos das amostras geradas [11]

### Limitações do MLE como Métrica de Avaliação

O Maximum Likelihood Estimation (MLE) é amplamente utilizado para treinar modelos generativos, mas apresenta limitações significativas como métrica de avaliação [12]:

1. **Sensibilidade a outliers**: MLE pode atribuir probabilidades muito baixas a amostras raras mas válidas, levando a uma superestimação da qualidade do modelo [13].

2. **Problema de dimensionalidade**: Em espaços de alta dimensão, a log-verossimilhança pode ser dominada por fatores que não são perceptualmente relevantes [14].

3. **Foco em detalhes locais**: MLE tende a priorizar a reconstrução precisa de detalhes locais, potencialmente à custa da estrutura global [15].

4. **Insensibilidade a modos**: Um modelo que captura apenas um subconjunto dos modos da distribuição real pode ainda obter alta log-verossimilhança [16].

Matematicamente, podemos expressar o problema de MLE como:

$$
\theta^* = \arg\max_\theta \frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)
$$

Onde $\theta^*$ são os parâmetros ótimos do modelo, $x_i$ são as amostras de treinamento, e $p_\theta(x)$ é a densidade de probabilidade do modelo [17].

> ❗ **Ponto de Atenção**: A otimização de MLE pode levar a modelos que atribuem probabilidade muito alta a regiões do espaço de entrada que não correspondem a dados realistas [18].

#### Questões Técnicas/Teóricas

1. Como a curse of dimensionality afeta a interpretação da log-verossimilhança em modelos generativos de alta dimensão?
2. Descreva um cenário em que um modelo com alta log-verossimilhança poderia gerar amostras de baixa qualidade.

### Métricas Alternativas e Complementares

Dadas as limitações do MLE, várias métricas alternativas e complementares têm sido propostas para avaliar modelos generativos [19]:

1. **Inception Score (IS)**: Mede a qualidade e diversidade das amostras geradas usando uma rede neural pré-treinada [20].

2. **Fréchet Inception Distance (FID)**: Compara as estatísticas das amostras geradas com as dos dados reais em um espaço de características aprendido [21].

3. **Precision e Recall**: Adaptados para modelos generativos, medem a qualidade e cobertura das amostras geradas [22].

4. **Kernel Inception Distance (KID)**: Uma variante do FID que é imparcial e pode ser usado com menos amostras [23].

Estas métricas tentam capturar aspectos diferentes da qualidade das amostras que não são necessariamente refletidos na log-verossimilhança [24].

> ✔️ **Ponto de Destaque**: A combinação de múltiplas métricas, incluindo avaliação humana, geralmente fornece uma visão mais completa da qualidade do modelo generativo [25].

### Implicações para o Design e Treinamento de Modelos

A compreensão das limitações do MLE tem implicações importantes para o design e treinamento de modelos generativos [26]:

1. **Regularização**: Técnicas de regularização podem ser usadas para evitar que o modelo se concentre excessivamente em detalhes locais [27].

2. **Arquiteturas hierárquicas**: Modelos que incorporam estruturas hierárquicas podem capturar melhor as dependências de longo alcance [28].

3. **Objetivos de treinamento alternativos**: Alguns modelos, como GANs, usam objetivos de treinamento que não se baseiam diretamente na log-verossimilhança [29].

4. **Avaliação multi-facetada**: É crucial avaliar modelos generativos usando uma combinação de métricas quantitativas e avaliação qualitativa [30].

Uma abordagem promissora é o uso de modelos híbridos que combinam as vantagens de diferentes paradigmas de modelagem [31]:

```python
import torch
import torch.nn as nn

class HybridGenerativeModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.vae_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.gan_generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.gan_discriminator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def vae_forward(self, x):
        mu, logvar = self.vae_encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return self.vae_decoder(z), mu, logvar
    
    def gan_forward(self, z):
        return self.gan_generator(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, z):
        vae_out, mu, logvar = self.vae_forward(x)
        gan_out = self.gan_forward(z)
        return vae_out, gan_out, mu, logvar
```

Este modelo híbrido combina um Variational Autoencoder (VAE) com um Generative Adversarial Network (GAN), permitindo explorar tanto a maximização da log-verossimilhança (via VAE) quanto a geração de amostras de alta qualidade (via GAN) [32].

#### Questões Técnicas/Teóricas

1. Como o tradeoff entre fidelidade local e coerência global se manifesta em diferentes arquiteturas de modelos generativos?
2. Proponha uma métrica de avaliação que poderia capturar aspectos da qualidade das amostras que a log-verossimilhança tende a ignorar.

### Conclusão

A relação entre log-verossimilhança e qualidade das amostras em modelos generativos é complexa e muitas vezes não-linear. Enquanto o MLE continua sendo uma ferramenta valiosa para o treinamento de modelos, suas limitações como métrica de avaliação são significativas, especialmente para modelos de alta dimensionalidade [33]. 

A comunidade de aprendizagem de máquina tem respondido a essas limitações desenvolvendo métricas complementares e abordagens de modelagem alternativas. No entanto, a avaliação abrangente de modelos generativos continua sendo um desafio em aberto, requerendo uma combinação cuidadosa de métricas quantitativas, avaliação qualitativa e consideração do contexto específico da aplicação [34].

À medida que o campo avança, é provável que vejamos o desenvolvimento de métodos de avaliação mais sofisticados que possam capturar melhor a qualidade multifacetada dos modelos generativos, indo além da simples maximização da log-verossimilhança [35].

### Questões Avançadas

1. Considerando as limitações do MLE, proponha uma função objetivo alternativa para treinar modelos generativos que poderia potencialmente superar algumas dessas limitações. Justifique matematicamente sua proposta.

2. Em um cenário onde você tem acesso a um grande conjunto de dados não rotulados e um pequeno conjunto de dados rotulados de alta qualidade, como você poderia combinar técnicas de aprendizado semi-supervisionado com modelos generativos para melhorar tanto a qualidade das amostras geradas quanto a utilidade do modelo para tarefas downstream?

3. Discuta as implicações éticas e práticas de usar modelos generativos treinados para maximizar a log-verossimilhança em aplicações do mundo real, como geração de conteúdo ou tomada de decisões automatizada. Como as limitações discutidas neste resumo poderiam impactar essas aplicações?

### Referências

[1] "Unfortunately, evaluating or maximizing the log-likelihood requires not just confronting the problem of intractable inference to marginalize out the latent variables, but also the problem of an intractable partition function within the undirected model of the top two layers." (Trecho de DLB - Deep Generative Models.pdf)

[2] "Higher log-likelihood doesn't necessarily mean better looking samples" (Trecho de cs236_lecture4.pdf)

[3] "The Kullback-Leibler divergence (KL-divergence) between two distributions p and q is defined as D(p∥q) = X x p(x) log p(x) q(x) ." (Trecho de cs236_lecture4.pdf)

[4] "Maximum likelihood learning is then: max P θ 1 |D| X x∈D log P θ (x)" (Trecho de cs236_lecture4.pdf)

[5] "Being able to generate realistic samples from the data distribution is one of the goals of a generative model, practitioners often evaluate generative models by visually inspecting the samples." (Trecho de DLB - Deep Generative Models.pdf)

[6] "Unfortunately, it is possible for a very poor probabilistic model to produce very good samples." (Trecho de DLB - Deep Generative Models.pdf)

[7] "Theis et al. (2015) review many of the issues involved in evaluating generative models, including many of the ideas described above. They highlight the fact that there are many different uses of generative models and that the choice of metric must match the intended use of the model." (Trecho de DLB - Deep Generative Models.pdf)

[8] "KL-divergence is one possibility: D(P data ||P θ ) = E x∼P data  log  P data (x) P θ (x)  = X x P data (x) log P data (x) P θ (x)" (Trecho de cs236_lecture4.pdf)

[9] "Because of log, samples x where P θ (x) ≈ 0 weigh heavily in objective" (Trecho de cs236_lecture4.pdf)

[10] "For example, real-valued models of MNIST can obtain arbitrarily high likelihood by assigning arbitrarily low variance to background pixels that never change." (Trecho de DLB - Deep Generative Models.pdf)

[11] "Models and algorithms that detect these constant features can reap unlimited rewards, even though this is not a very useful thing to do." (Trecho de DLB - Deep Generative Models.pdf)

[12] "The potential to achieve a cost approaching negative infinity is present for any kind of maximum likelihood problem with real values, but it is especially problematic for generative models of MNIST because so many of the output values are trivial to predict." (Trecho de DLB - Deep Generative Models.pdf)

[13] "This strongly suggests a need for developing other ways of evaluating generative models." (Trecho de DLB - Deep Generative Models.pdf)

[14] "For example, some generative models are better at assigning high probability to most realistic points while other generative models are better at rarely assigning high probability to unrealistic points." (Trecho de DLB - Deep Generative Models.pdf)

[15] "These differences can result from whether a generative model is designed to minimize D KL (p data p model ) or D KL (p model p data ), as illustrated in figure 3.6." (Trecho de DLB - Deep Generative Models.pdf)

[16] "Unfortunately, even when we restrict the use of each metric to the task it is most suited for, all of the metrics currently in use continue to have serious weaknesses." (Trecho de DLB - Deep Generative Models.pdf)

[17] "Maximum likelihood learning is then: max P θ 1 |D| X x∈D log P θ (x)" (Trecho de cs236_lecture4.pdf)

[18] "One of the most important research topics in generative modeling is therefore not just how to improve generative models, but in fact, designing new techniques to measure our progress." (Trecho de DLB - Deep Generative Models.pdf)

[19] "Researchers studying generative models often need to compare one generative model to another, usually in order to demonstrate that a newly invented generative model is better at capturing some distribution than the pre-existing models." (Trecho de DLB - Deep Generative Models.pdf)

[20] "In many cases, we can not actually evaluate the log probability of the data under the model, but only an approximation." (Trecho de DLB - Deep Generative Models.pdf)

[21] "In these cases, it is important to think