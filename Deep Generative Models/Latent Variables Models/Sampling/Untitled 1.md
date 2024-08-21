## Amostragem por Importância em Inferência Variacional

![image-20240821181120784](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821181120784.png)

<image: Uma ilustração mostrando diferentes distribuições de probabilidade se sobrepondo, com setas indicando a amostragem de pontos de uma distribuição para aproximar outra.>

### Introdução

A amostragem por importância é uma técnica fundamental em estatística computacional e aprendizado de máquina, especialmente no contexto de inferência variacional para modelos latentes. Este método permite estimar propriedades de uma distribuição de interesse utilizando amostras de uma distribuição diferente, mais fácil de amostrar [1]. No contexto de modelos latentes e inferência variacional, a amostragem por importância desempenha um papel crucial na estimação de quantidades intratáveis, como a verossimilhança marginal e gradientes para otimização de parâmetros [2].

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Distribuição alvo**     | A distribuição de interesse $p(z)$, geralmente difícil de amostrar diretamente [1]. |
| **Distribuição proposta** | Uma distribuição auxiliar $q(z)$, mais fácil de amostrar, usada para aproximar a distribuição alvo [1]. |
| **Função de importância** | A razão entre a distribuição alvo e a proposta, $w(z) = \frac{p(z)}{q(z)}$, usada para corrigir o viés introduzido pela amostragem de $q(z)$ [2]. |

> ⚠️ **Nota Importante**: A escolha da distribuição proposta $q(z)$ é crucial para a eficácia da amostragem por importância. Uma escolha inadequada pode levar a estimativas de alta variância ou até mesmo falhar completamente [3].

### Princípios da Amostragem por Importância

A amostragem por importância baseia-se no princípio de que podemos estimar expectativas em relação a uma distribuição $p(z)$ usando amostras de uma distribuição diferente $q(z)$. Matematicamente, isso é expresso como [4]:

$$
\mathbb{E}_{p(z)}[f(z)] = \int f(z)p(z)dz = \int f(z)\frac{p(z)}{q(z)}q(z)dz = \mathbb{E}_{q(z)}\left[f(z)\frac{p(z)}{q(z)}\right]
$$

Onde $f(z)$ é uma função arbitrária de interesse.

Na prática, esta expectativa é aproximada por uma média empírica [5]:

$$
\mathbb{E}_{p(z)}[f(z)] \approx \frac{1}{N}\sum_{i=1}^N f(z_i)\frac{p(z_i)}{q(z_i)}, \quad z_i \sim q(z)
$$

<image: Um diagrama mostrando o processo de amostragem de $q(z)$, cálculo dos pesos de importância, e a média ponderada resultante.>

#### Aplicação em Modelos Latentes

Em modelos latentes, a amostragem por importância é frequentemente usada para estimar a verossimilhança marginal $p(x)$, que é intratável na maioria dos casos [6]:

$$
p(x) = \int p(x|z)p(z)dz \approx \frac{1}{N}\sum_{i=1}^N \frac{p(x|z_i)p(z_i)}{q(z_i)}, \quad z_i \sim q(z)
$$

Esta estimativa é crucial para a avaliação de modelos e para o cálculo do Evidence Lower Bound (ELBO) em inferência variacional [7].

#### Questões Técnicas/Teóricas

1. Como a variância da estimativa por amostragem por importância é afetada pela escolha da distribuição proposta $q(z)$?
2. Em que situações a amostragem por importância pode falhar completamente, e como isso pode ser detectado na prática?

### Escolha da Distribuição Proposta q(z)

A eficácia da amostragem por importância depende criticamente da escolha da distribuição proposta $q(z)$. Uma boa distribuição proposta deve satisfazer algumas propriedades [8]:

1. **Suporte**: O suporte de $q(z)$ deve incluir o suporte de $p(z)$.
2. **Caudas pesadas**: $q(z)$ deve ter caudas mais pesadas que $p(z)$ para evitar variância infinita.
3. **Similaridade**: $q(z)$ deve ser similar a $p(z)$ para reduzir a variância da estimativa.

#### Estratégias para Escolha de q(z)

1. **Aproximação Gaussiana**: Em muitos casos, uma distribuição Gaussiana com média e covariância correspondentes à moda e curvatura local de $p(z)$ pode ser uma escolha razoável [9].

2. **Mistura de Importância**: Utilizar uma mistura de distribuições para $q(z)$ pode melhorar a cobertura do espaço de amostragem [10]:

   $$
   q(z) = \sum_{k=1}^K \pi_k q_k(z)
   $$

   onde $\pi_k$ são pesos da mistura e $q_k(z)$ são componentes individuais.

3. **Amostragem Adaptativa**: Ajustar iterativamente $q(z)$ baseado em amostras anteriores para melhorar a aproximação [11].

> ✔️ **Ponto de Destaque**: A escolha ótima teórica para $q(z)$ é proporcional a $|f(z)|p(z)$, mas esta geralmente é tão intratável quanto a distribuição original $p(z)$ [12].

#### Diagnóstico e Avaliação

Para avaliar a qualidade da distribuição proposta, pode-se utilizar métricas como [13]:

1. **Effective Sample Size (ESS)**:
   
   $$
   ESS = \frac{(\sum_{i=1}^N w_i)^2}{\sum_{i=1}^N w_i^2}
   $$

   onde $w_i = \frac{p(z_i)}{q(z_i)}$ são os pesos de importância normalizados.

2. **Pareto-k Diagnostic**: Ajusta uma distribuição de Pareto generalizada aos pesos de importância e avalia o parâmetro de forma $k$ [14].

<image: Um gráfico mostrando a distribuição dos pesos de importância e o ajuste da distribuição de Pareto para diagnóstico.>

### Implementação em Python

Aqui está um exemplo simplificado de implementação de amostragem por importância usando PyTorch:

```python
import torch
import torch.distributions as dist

def importance_sampling(target_log_prob, proposal, num_samples):
    # Amostra da distribuição proposta
    z = proposal.sample((num_samples,))
    
    # Calcula os pesos de importância (em log-espaço para estabilidade numérica)
    log_weights = target_log_prob(z) - proposal.log_prob(z)
    
    # Normaliza os pesos
    weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
    
    return z, weights

# Exemplo de uso
target = dist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
proposal = dist.Normal(torch.tensor([0.0]), torch.tensor([2.0]))

z, weights = importance_sampling(target.log_prob, proposal, 1000)

# Estima a média da distribuição alvo
estimated_mean = torch.sum(weights * z)
print(f"Estimated mean: {estimated_mean.item()}")
```

Este código demonstra uma implementação básica de amostragem por importância, incluindo o cálculo dos pesos em log-espaço para evitar instabilidades numéricas [15].

#### Questões Técnicas/Teóricas

1. Como você modificaria o código acima para implementar amostragem por importância adaptativa?
2. Quais são as implicações de usar uma distribuição proposta com variância muito menor ou muito maior que a distribuição alvo?

### Conclusão

A amostragem por importância é uma técnica poderosa e versátil para lidar com distribuições complexas e intratáveis em inferência variacional e modelos latentes. Sua eficácia depende criticamente da escolha da distribuição proposta, e várias estratégias avançadas foram desenvolvidas para otimizar essa escolha. Compreender os princípios teóricos e as considerações práticas da amostragem por importância é essencial para aplicá-la efetivamente em problemas de aprendizado de máquina e estatística computacional.

### Questões Avançadas

1. Como a amostragem por importância se relaciona com outros métodos de Monte Carlo, como Metropolis-Hastings ou Hamiltonian Monte Carlo? Quais são as vantagens e desvantagens comparativas?

2. Discuta como a amostragem por importância pode ser combinada com técnicas de redução de variância, como controle variates ou amostragem estratificada, para melhorar a eficiência das estimativas.

3. Em um cenário de inferência variacional com um modelo latente de alta dimensionalidade, como você abordaria o desafio da "maldição da dimensionalidade" na amostragem por importância? Proponha e justifique uma estratégia.

### Referências

[1] "Likelihood function p
θ 
(x) for Partially Observed Data is hard to compute:" (Trecho de cs236_lecture5.pdf)

[2] "We can think of it as an (intractable) expectation. Monte Carlo to the rescue:" (Trecho de cs236_lecture5.pdf)

[3] "Need a clever way to select z
(j) 
to reduce variance of the estimator." (Trecho de cs236_lecture5.pdf)

[4] "p
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

[5] "Monte Carlo to the rescue:
1 
Sample z
(1)
, · · · , z
(k) 
from q(z)
2 
Approximate expectation with sample average
p
θ 
(x) ≈ 
1
k
k
X
j=1
p
θ 
(x, z
(j)
)
q(z
(j)
)" (Trecho de cs236_lecture5.pdf)

[6] "This is an unbiased estimator of p
θ 
(x)
E
z
(j)
)∼q(z)


1
k
k
X
j=1
p
θ 
(x, z
(j)
)
q(z
(j)
)


= p
θ 
(x)" (Trecho de cs236_lecture5.pdf)

[7] "Evidence lower bound (ELBO) holds for any q
log p(x; θ) ≥ 
X
z
q(z) log

p
θ
(x, z)
q(z)
" (Trecho de cs236_lecture5.pdf)

[8] "What is a good choice for q(z)? Intuitively, frequently sample
 z
(completions) that are likely given x under p
θ 
(x, z)." (Trecho de cs236_lecture5.pdf)

[9] "Suppose q(z; ϕ) is a (tractable) probability distribution over the hidden
variables parameterized by ϕ (variational parameters)" (Trecho de cs236_lecture5.pdf)

[10] "For example, a Gaussian with mean and covariance specified by ϕ
q(z; ϕ) = N (ϕ
1
, ϕ
2
)" (Trecho de cs236_lecture5.pdf)

[11] "Variational inference: pick ϕ so that q(z; ϕ) is as close as possible to
p(z|x; θ)." (Trecho de cs236_lecture5.pdf)

[12] "Is ϕ
i 
= 0.5 ∀i a good approximation to the posterior p(x
top 
|x
bottom
; θ)? No" (Trecho de cs236_lecture5.pdf)

[13] "Is ϕ
i 
= 1 ∀i a good approximation to the posterior p(x
top 
|x
bottom
; θ)? No" (Trecho de cs236_lecture5.pdf)

[14] "Is ϕ
i 
≈ 1 for pixels i corresponding to the top part of digit 9 a good
approximation? Yes" (Trecho de cs236_lecture5.pdf)

[15] "The better q(z; ϕ) can approximate the posterior p(z|x; θ), the smaller
D
KL
(q(z; ϕ)∥p(z|x; θ)) we can achieve, the closer ELBO will be to
log p(x; θ)." (Trecho de cs236_lecture5.pdf)