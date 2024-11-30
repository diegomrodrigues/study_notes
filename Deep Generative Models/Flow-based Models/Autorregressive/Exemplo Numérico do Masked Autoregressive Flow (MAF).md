**Exemplo Numérico do Masked Autoregressive Flow (MAF)**

Para ilustrar o funcionamento do Masked Autoregressive Flow, vamos construir um exemplo numérico simples em duas dimensões. Este exemplo demonstrará como as transformações autoregressivas mascaradas são aplicadas para transformar uma distribuição latente simples em uma distribuição complexa.

### Configuração do Exemplo

**Variável Latente**:

- Considere uma variável latente $\mathbf{z} = (z_1, z_2)$, onde cada $z_i$ é amostrado independentemente de uma distribuição normal padrão $\mathcal{N}(0, 1)$.

**Transformação MAF**:

- Queremos transformar $\mathbf{z}$ em uma variável observável $\mathbf{x} = (x_1, x_2)$ usando as seguintes equações autoregressivas:

  $$
  \begin{align*}
  x_1 &= \mu_1 + \sigma_1 \cdot z_1 \\
  x_2 &= \mu_2(x_1) + \sigma_2(x_1) \cdot z_2
  \end{align*}
  $$

**Parâmetros**:

- As funções $\mu_i$ e $\sigma_i$ são definidas como:

  $$
  \begin{align*}
  \mu_1 &= 0 \quad (\text{constante}) \\
  \sigma_1 &= \exp(s_1) \quad \text{com } s_1 = 0 \quad (\text{constante}) \\
  \mu_2(x_1) &= 0.5 \cdot x_1 \\
  \sigma_2(x_1) &= \exp(s_2) \quad \text{com } s_2 = 0.1 \cdot x_1
  \end{align*}
  $$

### Passo a Passo do Cálculo

**1. Amostragem da Variável Latente $\mathbf{z}$**:

- Suponha que extraímos as seguintes amostras de $\mathcal{N}(0, 1)$:
  - $z_1 = 1.0$
  - $z_2 = -0.5$

**2. Cálculo de $x_1$**:

- **Cálculo de $\mu_1$ e $\sigma_1$**:
  - $\mu_1 = 0$
  - $s_1 = 0$
  - $\sigma_1 = \exp(s_1) = \exp(0) = 1$

- **Transformação**:
  - $x_1 = \mu_1 + \sigma_1 \cdot z_1 = 0 + 1 \times 1.0 = 1.0$

**3. Cálculo de $x_2$**:

- **Cálculo de $\mu_2(x_1)$ e $\sigma_2(x_1)$**:
  - $\mu_2 = 0.5 \times x_1 = 0.5 \times 1.0 = 0.5$
  - $s_2 = 0.1 \times x_1 = 0.1 \times 1.0 = 0.1$
  - $\sigma_2 = \exp(s_2) = \exp(0.1) \approx 1.1052$

- **Transformação**:
  - $x_2 = \mu_2 + \sigma_2 \cdot z_2 = 0.5 + 1.1052 \times (-0.5) = 0.5 - 0.5526 = -0.0526$

**4. Cálculo do Determinante do Jacobiano**:

- O Jacobiano da transformação é triangular, e seu determinante é o produto dos elementos diagonais:

  $$
  \left| \frac{\partial \mathbf{x}}{\partial \mathbf{z}} \right| = \sigma_1 \times \sigma_2
  $$

- **Cálculo**:
  - $\sigma_1 = 1$
  - $\sigma_2 = 1.1052$
  - $\left| J \right| = 1 \times 1.1052 = 1.1052$

**5. Cálculo da Densidade de Probabilidade de $\mathbf{x}$**:

- **Densidade da variável latente**:
  - $p(z_1) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{z_1^2}{2} \right) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(1.0)^2}{2} \right) \approx 0.24197$
  - $p(z_2) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{z_2^2}{2} \right) = \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{(-0.5)^2}{2} \right) \approx 0.35206$

- **Densidade conjunta**:
  - $p(\mathbf{z}) = p(z_1) \times p(z_2) = 0.24197 \times 0.35206 \approx 0.08500$

- **Aplicando a Mudança de Variáveis**:
  - $p(\mathbf{x}) = p(\mathbf{z}) \left| \frac{1}{\det J} \right| = 0.08500 \times \left| \frac{1}{1.1052} \right| \approx 0.08500 \times 0.90484 \approx 0.07691$

### Resumo dos Resultados

- **Amostras**:
  - Variável latente: $\mathbf{z} = (1.0, -0.5)$
  - Variável observável: $\mathbf{x} = (1.0, -0.0526)$

- **Densidade de Probabilidade**:
  - Densidade em $\mathbf{x}$: $p(\mathbf{x}) \approx 0.07691$

### Interpretação

Neste exemplo, demonstramos como o MAF transforma uma amostra da distribuição latente $\mathcal{N}(0, 1)$ em uma nova amostra $\mathbf{x}$, incorporando dependências autoregressivas. A primeira componente $x_1$ é transformada diretamente de $z_1$, enquanto a segunda componente $x_2$ depende de $x_1$, incorporando assim a estrutura autoregressiva.

A densidade de probabilidade de $\mathbf{x}$ é calculada utilizando o determinante do Jacobiano da transformação, garantindo que a probabilidade total seja preservada. Este processo é crucial para o treinamento de modelos MAF, onde o objetivo é ajustar os parâmetros para que a distribuição transformada corresponda a uma distribuição de dados observados.

### Extensão do Exemplo

Este exemplo pode ser estendido para dimensões superiores, onde cada componente $x_i$ depende dos componentes anteriores $x_{1:i-1}$. A complexidade computacional aumenta, mas o princípio permanece o mesmo:

- **Transformações Autoregressivas**:
  - Cada $x_i$ é calculado usando $\mu_i$ e $\sigma_i$ que podem ser funções complexas de $x_{1:i-1}$, tipicamente modeladas por redes neurais profundas.

- **Máscaras em Redes Neurais**:
  - Para garantir a propriedade autoregressiva, máscaras binárias são aplicadas aos pesos das redes neurais, impedindo que $x_i$ dependa de $x_{j}$ para $j \geq i$.

### Conclusão

O exemplo numérico fornecido ilustra o funcionamento básico do Masked Autoregressive Flow em transformar variáveis latentes em observáveis através de transformações invertíveis e autoregressivas. Este processo permite modelar distribuições de probabilidade complexas, mantendo a capacidade de calcular a densidade de probabilidade de forma exata, o que é essencial para tarefas de aprendizado não supervisionado e modelagem generativa.