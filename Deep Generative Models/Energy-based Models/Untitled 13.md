## Máquinas de Boltzmann: Aprendendo Distribuições de Probabilidade Arbitrárias sobre Vetores Binários

<image: Uma rede neural com nós interconectados representando uma máquina de Boltzmann, com alguns nós visíveis e outros ocultos. Setas bidirecionais entre os nós indicam as conexões simétricas.>

### Introdução

As máquinas de Boltzmann são modelos probabilísticos poderosos que foram originalmente introduzidos como uma abordagem "conexionista" geral para aprender distribuições de probabilidade arbitrárias sobre vetores binários [1]. Estes modelos desempenham um papel fundamental no campo do aprendizado profundo e da modelagem generativa, oferecendo uma estrutura flexível para capturar complexas relações entre variáveis binárias.

Neste resumo, exploraremos em profundidade os conceitos fundamentais, a formulação matemática, as variantes e as aplicações das máquinas de Boltzmann. Nosso foco será na compreensão de como esses modelos conseguem aprender e representar distribuições de probabilidade complexas, bem como seus desafios e limitações.

### Conceitos Fundamentais

| Conceito                        | Explicação                                                   |
| ------------------------------- | ------------------------------------------------------------ |
| **Máquina de Boltzmann**        | Um modelo de rede neural estocástica baseado em energia, capaz de aprender distribuições de probabilidade arbitrárias sobre vetores binários [1]. |
| **Função de Energia**           | Uma função que atribui um valor escalar (energia) a cada configuração possível das variáveis do modelo [2]. |
| **Distribuição de Boltzmann**   | A distribuição de probabilidade definida pela função de energia, seguindo a forma de uma distribuição de Gibbs [2]. |
| **Unidades Visíveis e Ocultas** | As variáveis do modelo, onde as visíveis representam os dados observáveis e as ocultas capturam dependências de ordem superior [3]. |

> ⚠️ **Nota Importante**: A capacidade das máquinas de Boltzmann de modelar distribuições arbitrárias as torna extremamente poderosas, mas também computacionalmente desafiadoras de treinar e amostrar.

### Formulação Matemática

A máquina de Boltzmann é definida sobre um vetor binário aleatório $x \in \{0,1\}^d$ de dimensão $d$. A distribuição de probabilidade conjunta é dada pela distribuição de Boltzmann [2]:

$$
P(x) = \frac{\exp(-E(x))}{Z}
$$

Onde:
- $E(x)$ é a função de energia
- $Z$ é a função de partição que assegura a normalização da distribuição

A função de energia da máquina de Boltzmann é definida como [2]:

$$
E(x) = -x^TUx - b^Tx
$$

Onde:
- $U$ é a matriz de "peso" dos parâmetros do modelo
- $b$ é o vetor de vieses

> ✔️ **Ponto de Destaque**: A função de energia captura as interações entre todas as unidades do modelo, permitindo representar complexas dependências entre variáveis.

A função de partição $Z$ é dada por:

$$
Z = \sum_x \exp(-E(x))
$$

Esta soma é realizada sobre todas as configurações possíveis de $x$, o que torna o cálculo de $Z$ intratável para modelos de grande escala.

#### Inferência e Amostragem

A inferência em máquinas de Boltzmann geralmente envolve a amostragem da distribuição posterior $P(h|v)$, onde $h$ são as unidades ocultas e $v$ são as unidades visíveis. Isso é tipicamente realizado usando métodos de Monte Carlo via Cadeias de Markov (MCMC), como a amostragem de Gibbs [4].

Para uma máquina de Boltzmann totalmente conectada, a probabilidade condicional de uma unidade $x_i$ dado o estado das outras unidades é:

$$
P(x_i = 1|x_{-i}) = \sigma\left(\sum_{j \neq i} U_{ij}x_j + b_i\right)
$$

Onde $\sigma(x) = \frac{1}{1+\exp(-x)}$ é a função sigmoide.

#### Aprendizagem

O treinamento de uma máquina de Boltzmann visa maximizar a log-verossimilhança dos dados observados. O gradiente da log-verossimilhança em relação aos parâmetros do modelo é [5]:

$$
\frac{\partial \log P(v)}{\partial \theta} = \mathbb{E}_{P(h|v)}\left[\frac{\partial E(v,h)}{\partial \theta}\right] - \mathbb{E}_{P(v,h)}\left[\frac{\partial E(v,h)}{\partial \theta}\right]
$$

Onde $\theta$ representa os parâmetros do modelo (elementos de $U$ e $b$).

> ❗ **Ponto de Atenção**: O cálculo exato deste gradiente é intratável para modelos de grande escala devido à expectativa sobre a distribuição conjunta $P(v,h)$. Métodos aproximados, como Contrastive Divergence (CD), são frequentemente utilizados na prática [6].

#### Questões Técnicas/Teóricas

1. Como a intratabilidade da função de partição $Z$ afeta o treinamento e a inferência em máquinas de Boltzmann? Discuta possíveis abordagens para lidar com este desafio.

2. Explique por que a amostragem de Gibbs é um método eficaz para realizar inferência em máquinas de Boltzmann. Quais são suas limitações?

### Variantes de Máquinas de Boltzmann

#### Máquina de Boltzmann Restrita (RBM)

A Máquina de Boltzmann Restrita (RBM) é uma variante importante que impõe uma estrutura bipartida ao modelo [7]. As unidades são divididas em uma camada visível e uma camada oculta, sem conexões entre unidades da mesma camada.

<image: Diagrama de uma RBM mostrando a estrutura bipartida com conexões apenas entre camadas visível e oculta>

A função de energia para uma RBM é dada por:

$$
E(v,h) = -v^TWh - b^Tv - c^Th
$$

Onde:
- $v$ e $h$ são os vetores de unidades visíveis e ocultas, respectivamente
- $W$ é a matriz de pesos entre as camadas visível e oculta
- $b$ e $c$ são os vetores de viés para as unidades visíveis e ocultas, respectivamente

A estrutura bipartida da RBM permite inferência exata das unidades ocultas dado as visíveis (e vice-versa) em um único passo:

$$
P(h_j = 1|v) = \sigma(c_j + W_{:,j}^Tv)
$$

$$
P(v_i = 1|h) = \sigma(b_i + W_{i,:}h)
$$

> ✔️ **Ponto de Destaque**: A estrutura da RBM permite uma amostragem de Gibbs muito mais eficiente, alternando entre atualizar todas as unidades ocultas e todas as unidades visíveis em paralelo.

#### Máquina de Boltzmann Profunda (DBM)

As Máquinas de Boltzmann Profundas (DBMs) estendem a ideia das RBMs para múltiplas camadas ocultas [8]. A função de energia para uma DBM com duas camadas ocultas é:

$$
E(v,h^{(1)},h^{(2)}) = -v^TW^{(1)}h^{(1)} - h^{(1)T}W^{(2)}h^{(2)}
$$

Onde $h^{(1)}$ e $h^{(2)}$ são as primeira e segunda camadas ocultas, respectivamente.

> ⚠️ **Nota Importante**: DBMs são capazes de aprender hierarquias de características, mas sua inferência e treinamento são significativamente mais desafiadores que as RBMs devido às interações entre múltiplas camadas ocultas.

#### Questões Técnicas/Teóricas

1. Compare e contraste as vantagens e desvantagens das RBMs em relação às máquinas de Boltzmann totalmente conectadas em termos de capacidade de modelagem e eficiência computacional.

2. Discuta como a estrutura em camadas das DBMs permite a aprendizagem de representações hierárquicas. Quais são os desafios específicos no treinamento de DBMs?

### Aplicações e Implementação

As máquinas de Boltzmann, especialmente as RBMs, têm sido aplicadas com sucesso em várias tarefas de aprendizado de máquina, incluindo:

1. Redução de dimensionalidade
2. Extração de características
3. Pré-treinamento de redes neurais profundas
4. Modelagem de tópicos em processamento de linguagem natural

Aqui está um exemplo simplificado de como implementar uma RBM usando PyTorch:

```python
import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
    
    def sample_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h, torch.bernoulli(p_h)
    
    def sample_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v, torch.bernoulli(p_v)
    
    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

    def forward(self, v, k=1):
        h_, _ = self.sample_h(v)
        for _ in range(k):
            _, h = self.sample_h(v)
            _, v = self.sample_v(h)
        return v, h_
```

Este código define uma classe `RBM` que implementa as operações básicas de uma Máquina de Boltzmann Restrita, incluindo amostragem das unidades ocultas e visíveis, cálculo da energia livre, e execução de k passos de amostragem de Gibbs.

> ❗ **Ponto de Atenção**: Na prática, o treinamento de RBMs geralmente envolve o uso de técnicas como Contrastive Divergence (CD) ou Persistent Contrastive Divergence (PCD) para aproximar o gradiente da log-verossimilhança.

#### Questões Técnicas/Teóricas

1. Como você modificaria a implementação acima para incorporar o algoritmo de Contrastive Divergence para treinamento? Quais considerações práticas são importantes ao implementar CD?

2. Discuta as vantagens e limitações de usar RBMs como extratores de características em comparação com outras técnicas de aprendizado não supervisionado.

### Conclusão

As máquinas de Boltzmann representam uma classe poderosa de modelos probabilísticos capazes de aprender distribuições complexas sobre vetores binários [9]. Sua formulação baseada em energia e a capacidade de incorporar unidades ocultas as tornam particularmente adequadas para capturar dependências de alta ordem nos dados.

Enquanto as máquinas de Boltzmann originais enfrentam desafios computacionais significativos, variantes como RBMs e DBMs trouxeram avanços práticos importantes [10]. Essas variantes não apenas tornaram o treinamento e a inferência mais tratáveis, mas também abriram caminho para o desenvolvimento de técnicas de aprendizado profundo mais avançadas.

Apesar dos recentes avanços em outros tipos de modelos generativos, as máquinas de Boltzmann continuam a ser uma área ativa de pesquisa, oferecendo insights valiosos sobre representação de conhecimento, inferência probabilística e aprendizado não supervisionado [11].

### Questões Avançadas

1. Compare o poder expressivo das máquinas de Boltzmann com outros modelos generativos modernos, como VAEs e GANs. Em quais cenários as máquinas de Boltzmann ainda podem oferecer vantagens únicas?

2. Discuta as implicações teóricas e práticas de usar máquinas de Boltzmann com unidades contínuas em vez de binárias. Como isso afetaria a formulação do modelo e os algoritmos de treinamento?

3. Proponha e discuta uma arquitetura híbrida que combine elementos de máquinas de Boltzmann com redes neurais feedforward modernas. Quais seriam os potenciais benefícios e desafios de tal abordagem?

### Referências

[1] "Boltzmann machines were originally introduced as a general "connectionist" approach to learning arbitrary probability distributions over binary vectors" (Trecho de DLB - Deep Generative Models.pdf)

[2] "The Boltzmann machine is an energy-based model (section 16.2.4), meaning we define the joint probability distribution using an energy function: P ( ) =x exp ( ( ))−E x Z , (20.1) where E(x ) is the energy function and Z is the partition function that ensures that x P ( ) = 1x . The energy function of the Boltzmann machine is given by E( ) =x −x U x b− x, (20.2) where U is the "weight" matrix of model parameters and b is the vector of bias parameters." (Trecho de DLB - Deep Generative Models.pdf)

[3] "The Boltzmann machine becomes more powerful when not all the variables are observed. In this case, the latent variables, can act similarly to hidden units in a multi-layer perceptron and model higher-order interactions among the visible units." (Trecho de DLB - Deep Generative Models.pdf)

[4] "Inference over the hidden units given the visible units is intractable. Mean field inference is also intractable because the variational lower bound involves taking expectations of cliques that encompass entire layers." (Trecho de DLB - Deep Generative Models.pdf)

[5] "Learning algorithms for Boltzmann machines are usually based on maximum likelihood. All Boltzmann machines have an intractable partition function, so the maximum likelihood gradient must be approximated using the techniques described in chapter 18." (Trecho de DLB - Deep Generative Models.pdf)

[6] "Contrastive divergence and persistent contrastive divergence with Gibbs sampling are still applicable. There is no need to invert any matrix." (Trecho de DLB - Deep Generative Models.pdf)

[7] "Restricted Boltzmann machines may be developed for many exponential family conditional distributions (Welling et al., 2005). Of these, the most common is the RBM with binary hidden units and real-valued visible units, with the conditional distribution over the visible units being a Gaussian distribution whose mean is a function of the hidden units." (Trecho de DLB - Deep Generative Models.pdf)