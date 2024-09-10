## Propriedades do ELBO (Evidence Lower Bound) e sua Relação com a Divergência KL

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240826133205060.png" alt="image-20240826133205060" style="zoom: 80%;" />

### Introdução

O **Evidence Lower Bound** (ELBO) é um conceito fundamental em inferência variacional e aprendizado de modelos latentes. Ele fornece ==uma aproximação tratável da log-verossimilhança== em modelos com variáveis latentes, onde o cálculo direto da verossimilhança é muitas vezes intratável [1]. Neste resumo, exploraremos as ==propriedades do ELBO, sua relação com a divergência Kullback-Leibler (KL), e as condições sob as quais o ELBO se iguala à log-verossimilhança verdadeira.==

### Conceitos Fundamentais

| Conceito                | Explicação                                                   |
| ----------------------- | ------------------------------------------------------------ |
| **ELBO**                | ==O Evidence Lower Bound é um limite inferior da log-verossimilhança marginal em modelos com variáveis latentes.== É utilizado como um substituto tratável para a log-verossimilhança em otimização variacional [1]. |
| **Divergência KL**      | ==A divergência Kullback-Leibler é uma medida de dissimilaridade entre duas distribuições de probabilidade.== Em inferência variacional, mede a diferença entre a ==distribuição variacional e a distribuição posterior verdadeira== [2]. |
| **Log-verossimilhança** | ==A log-verossimilhança é o logaritmo natural da probabilidade dos dados observados sob um modelo específico.== Maximizá-la é o objetivo do aprendizado de máxima verossimilhança [3]. |

> ⚠️ **Nota Importante**: O ELBO é sempre menor ou igual à log-verossimilhança verdadeira, daí o termo "lower bound" (limite inferior).

### Relação entre ELBO e Log-verossimilhança

A relação fundamental entre o ELBO e a log-verossimilhança é dada pela seguinte equação [4]:

$$
\log p(x; \theta) = \text{ELBO}(q) + D_{KL}(q(z) || p(z|x; \theta))
$$

Onde:
- $\log p(x; \theta)$ é a log-verossimilhança verdadeira
- $\text{ELBO}(q)$ é o Evidence Lower Bound
- ==$D_{KL}(q(z) || p(z|x; \theta))$ é a divergência KL entre a distribuição variacional $q(z)$ e a posterior verdadeira $p(z|x; \theta)$==

Esta equação demonstra que a ==log-verossimilhança pode ser decomposta em dois termos==: o ELBO e a divergência KL [5].

> ✔️ **Ponto de Destaque**: A diferença entre a log-verossimilhança e o ELBO é exatamente a divergência KL entre a distribuição variacional e a posterior verdadeira.

### Propriedades do ELBO

1. **Limite Inferior**: O ELBO é sempre menor ou igual à log-verossimilhança verdadeira [6]:

   $$
   \text{ELBO}(q) \leq \log p(x; \theta)
   $$

2. **Maximização**: ==Maximizar o ELBO é equivalente a minimizar a divergência KL entre $q(z)$ e $p(z|x; \theta)$ [7].==

3. **Decomposição**: O ELBO pode ser decomposto em dois termos [8]:

   $$
   \text{ELBO}(q) = \mathbb{E}_{q(z)}[\log p(x, z; \theta)] - \mathbb{E}_{q(z)}[\log q(z)]
   $$

   O primeiro termo é a esperança da log-verossimilhança conjunta sob $q(z)$, e o segundo é a entropia negativa de $q(z)$.

4. **Concavidade**: O ELBO é côncavo em relação à distribuição variacional $q(z)$ [9].

#### Questões Técnicas/Teóricas

1. Como a maximização do ELBO se relaciona com a minimização da divergência KL entre a distribuição variacional e a posterior verdadeira?
2. Por que o ELBO é sempre um limite inferior da log-verossimilhança verdadeira? Explique matematicamente.

### Condições para Igualdade com a Log-verossimilhança Verdadeira

==O ELBO se iguala à log-verossimilhança verdadeira quando a distribuição variacional $q(z)$ é exatamente igual à posterior verdadeira $p(z|x; \theta)$ [10].== Matematicamente, isso ocorre quando:
$$
D_{KL}(q(z) || p(z|x; \theta)) = 0
$$

Neste caso:

$$
\log p(x; \theta) = \text{ELBO}(q)
$$

> ❗ **Ponto de Atenção**: ==Na prática, é raro alcançar a igualdade exata, mas o objetivo é aproximar $q(z)$ o máximo possível de $p(z|x; \theta)$.==

### Implicações Práticas

1. **Otimização**: ==Ao maximizar o ELBO, estamos simultaneamente aproximando a distribuição variacional da posterior verdadeira e maximizando um limite inferior da log-verossimilhança [11].==

2. **Escolha da Família Variacional**: A escolha da família de distribuições para $q(z)$ afeta diretamente o quão próximo o ELBO pode chegar da log-verossimilhança verdadeira [12].

3. **Avaliação de Modelos**: O ELBO pode ser usado como uma métrica de avaliação de modelos, servindo como uma aproximação da log-verossimilhança [13].

### Aplicação em Variational Autoencoders (VAEs)

Em VAEs, o ELBO é usado como função objetivo de treinamento [14]:

$$
\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))
$$

Onde:
- $q(z|x)$ é o encoder (distribuição variacional)
- $p(x|z)$ é o decoder
- $p(z)$ é a prior sobre as variáveis latentes

==Esta formulação permite o treinamento end-to-end do VAE usando backpropagation e o "reparameterization trick" [15].==

#### Questões Técnicas/Teóricas

1. Como a escolha da família variacional afeta o desempenho de um VAE?
2. Explique como o "reparameterization trick" permite o treinamento eficiente de VAEs usando o ELBO como função objetivo.

### Conclusão

O ELBO é uma ferramenta poderosa em inferência variacional e aprendizado de modelos latentes. Sua relação com a divergência KL e a log-verossimilhança verdadeira fornece insights valiosos sobre o processo de otimização em modelos variacionais. Compreender as propriedades do ELBO e as condições para sua igualdade com a log-verossimilhança é crucial para o desenvolvimento e aplicação eficaz de métodos variacionais em machine learning e estatística.

### Questões Avançadas

1. Como você poderia modificar o ELBO para lidar com modelos que têm múltiplas camadas de variáveis latentes, como em hierarquias profundas de VAEs?

2. Discuta as vantagens e desvantagens de usar o ELBO versus outros métodos de aproximação da log-verossimilhança, como Annealed Importance Sampling (AIS) ou Markov Chain Monte Carlo (MCMC).

3. Em cenários onde a posterior verdadeira é multimodal, como a escolha da família variacional afeta a qualidade da aproximação do ELBO? Proponha uma estratégia para melhorar a aproximação nestes casos.

### Referências

[1] "O Evidence Lower Bound (ELBO) é um limite inferior da log-verossimilhança marginal em modelos com variáveis latentes." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[2] "A divergência Kullback-Leibler é uma medida de dissimilaridade entre duas distribuições de probabilidade." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[3] "Log-Likelihood function for Partially Observed Data is hard to compute" (Trecho de cs236_lecture5.pdf)

[4] "log p(x; θ) = L(x; θ, ϕ) + D_KL(q(z; ϕ)∥p(z|x; θ))" (Trecho de cs236_lecture5.pdf)

[5] "The better q(z; ϕ) can approximate the posterior p(z|x; θ), the smaller D_KL(q(z; ϕ)∥p(z|x; θ)) we can achieve, the closer ELBO will be to log p(x; θ)." (Trecho de cs236_lecture5.pdf)

[6] "log p(x|w) ≥ L" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[7] "Variational inference: pick ϕ so that q(z; ϕ) is as close as possible to p(z|x; θ)." (Trecho de cs236_lecture5.pdf)

[8] "L_n = ∫ q_n(z_n) ln { p(x_n|z_n, w)p(z_n) / q_n(z_n) } dz_n" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[9] "log() is a concave function. log(px + (1 − p)x′) ≥ p log(x) + (1 − p) log(x′)." (Trecho de cs236_lecture5.pdf)

[10] "Equality holds if q = p(z|x; θ)" (Trecho de cs236_lecture5.pdf)

[11] "Next: jointly optimize over θ and ϕ to maximize the ELBO over a dataset" (Trecho de cs236_lecture5.pdf)

[12] "Is ϕ_i ≈ 1 for pixels i corresponding to the top part of digit 9 a good approximation? Yes" (Trecho de cs236_lecture5.pdf)

[13] "ELBO can be evaluated using a Monte Carlo estimate. Hence it provides an approximation to the true log likelihood." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[14] "In VAEs, the ELBO is used as training objective function" (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)

[15] "The reparameterization trick can be extended to other distributions but is limited to continuous variables." (Trecho de Deep Learning Foundation and Concepts-574-590.pdf)