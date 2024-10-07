# Exemplo Numérico do Importance Weighted Autoencoder (IWAE)

Vamos ilustrar o funcionamento do IWAE através de um exemplo numérico simples. Neste exemplo, utilizaremos distribuições unidimensionais para facilitar os cálculos manuais.

## Configuração do Modelo

- **Variável observada**: $x = 1$
- **Variável latente**: $z$
- **Prior**: $p(z) = \mathcal{N}(0, 1)$ (distribuição normal padrão)
- **Likelihood**: $p(x|z) = \mathcal{N}(z, 1)$
- **Posterior variacional**: $q(z|x) = \mathcal{N}(\mu=0.5, \sigma^2=0.25)$

## Passos do Cálculo

### 1. Amostragem de $z$ da Posterior Variacional

Vamos usar $m = 2$ amostras para o IWAE.

Calculemos o desvio padrão da posterior variacional:

$$
\sigma = \sqrt{0.25} = 0.5
$$

Utilizando o truque da reparametrização:

$$
z_i = \mu + \sigma \cdot \epsilon_i
$$

Onde $\epsilon_i \sim \mathcal{N}(0, 1)$.

Para simplificar, vamos escolher valores específicos para $\epsilon_i$:

- $\epsilon_1 = 0$
- $\epsilon_2 = 1$

Calculamos as amostras:

- $z_1 = 0.5 + 0.5 \times 0 = 0.5$
- $z_2 = 0.5 + 0.5 \times 1 = 1.0$

### 2. Cálculo das Densidades

#### a) Prior $p(z_i)$

Utilizando a densidade da normal padrão:

$$
p(z_i) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{z_i^2}{2} \right)
$$

- Para $z_1 = 0.5$:

  $$
  p(z_1) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{0.5^2}{2} \right) \approx 0.3521
  $$

- Para $z_2 = 1.0$:

  $$
  p(z_2) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{1.0^2}{2} \right) \approx 0.2419
  $$

#### b) Likelihood $p(x|z_i)$

$$
p(x|z_i) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(x - z_i)^2}{2} \right)
$$

- Para $z_1 = 0.5$:

  $$
  p(1|0.5) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(1 - 0.5)^2}{2} \right) \approx 0.3521
  $$

- Para $z_2 = 1.0$:

  $$
  p(1|1.0) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(1 - 1.0)^2}{2} \right) \approx 0.3989
  $$

#### c) Posterior Variacional $q(z_i|x)$

$$
q(z_i|x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(z_i - \mu)^2}{2\sigma^2} \right)
$$

- Para $z_1 = 0.5$:

  $$
  q(0.5|1) = \frac{1}{0.5 \sqrt{2\pi}} \exp\left( -\frac{(0.5 - 0.5)^2}{2 \times 0.25} \right) \approx 0.7979
  $$

- Para $z_2 = 1.0$:

  $$
  q(1.0|1) = \frac{1}{0.5 \sqrt{2\pi}} \exp\left( -\frac{(1.0 - 0.5)^2}{2 \times 0.25} \right) \approx 0.4839
  $$

### 3. Cálculo dos Pesos de Importância $w_i$

$$
w_i = \frac{p(x|z_i) \cdot p(z_i)}{q(z_i|x)}
$$

- Para $i = 1$:

  $$
  w_1 = \frac{0.3521 \times 0.3521}{0.7979} \approx \frac{0.1240}{0.7979} \approx 0.1554
  $$

- Para $i = 2$:

  $$
  w_2 = \frac{0.3989 \times 0.2419}{0.4839} \approx \frac{0.0965}{0.4839} \approx 0.1994
  $$

### 4. Cálculo do Objetivo IWAE

$$
\mathcal{L}_{IWAE} = \ln\left( \frac{1}{m} \sum_{i=1}^m w_i \right)
$$

Calculando a média dos pesos:

$$
\frac{1}{2} (w_1 + w_2) = \frac{1}{2} (0.1554 + 0.1994) = 0.1774
$$

Portanto:

$$
\mathcal{L}_{IWAE} = \ln(0.1774) \approx -1.727
$$

### 5. Comparação com o ELBO Padrão

Para comparar, calculamos o ELBO usando apenas a primeira amostra ($m = 1$):

$$
\mathcal{L}_{ELBO} = \mathbb{E}_{q(z|x)} \left[ \ln p(x, z) - \ln q(z|x) \right]
$$

Calculando os termos:

- $\ln p(x, z_1) = \ln p(x|z_1) + \ln p(z_1)$

  $$
  \ln p(1|0.5) + \ln p(0.5) = \ln(0.3521) + \ln(0.3521) \approx -1.0439 + (-1.0439) = -2.0878
  $$

- $\ln q(z_1|x) = \ln(0.7979) \approx -0.2258$

Portanto:

$$
\mathcal{L}_{ELBO} = -2.0878 - (-0.2258) = -1.8620
$$

### 6. Análise dos Resultados

- **IWAE ($m = 2$)**: $\mathcal{L}_{IWAE} \approx -1.727$
- **ELBO ($m = 1$)**: $\mathcal{L}_{ELBO} \approx -1.862$

Observamos que o IWAE fornece um limite inferior mais apertado (valor maior) para a log-verossimilhança marginal em comparação com o ELBO padrão. Isso ilustra como o aumento do número de amostras $m$ no IWAE pode levar a estimativas mais precisas.

## Conclusão

Neste exemplo numérico, demonstramos passo a passo como calcular o objetivo do IWAE e como ele se compara ao ELBO padrão. A utilização de múltiplas amostras na estimação permite obter um limite inferior mais apertado, melhorando a aproximação da log-verossimilhança marginal e potencialmente levando a melhores representações latentes no modelo.