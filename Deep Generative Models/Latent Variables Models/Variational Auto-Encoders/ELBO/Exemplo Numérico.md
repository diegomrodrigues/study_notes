## Configuração do Exemplo

- **Tamanho do lote (batch size)**: 2
- **Dimensão dos dados de entrada ($\text{data\_dim}$)**: 3
- **Dimensão do espaço latente ($z_{\text{dim}}$)**: 2

### Dados de Entrada ($\mathbf{x}$)

Temos um lote de dois exemplos:

$$
\mathbf{x} = \begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 1 \\
\end{bmatrix}
$$

## Passo a Passo do Método `negative_elbo_bound`

### 1. **Codificação dos Dados de Entrada**

**Objetivo**: Obter os parâmetros da distribuição posterior $q(\mathbf{z}|\mathbf{x})$ a partir dos dados de entrada.

Suponhamos que o encoder produz as seguintes médias ($\mathbf{m}$) e variâncias ($\mathbf{v}$):

Para o primeiro exemplo ($\mathbf{x}_1$):

$$
\mathbf{m}_1 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix} 0.2 \\ 0.2 \end{bmatrix}
$$

Para o segundo exemplo ($\mathbf{x}_2$):

$$
\mathbf{m}_2 = \begin{bmatrix} -0.3 \\ 0.7 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0.1 \\ 0.3 \end{bmatrix}
$$

Assim, temos:

$$
\mathbf{m} = \begin{bmatrix}
0.5 & -0.5 \\
-0.3 & 0.7 \\
\end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix}
0.2 & 0.2 \\
0.1 & 0.3 \\
\end{bmatrix}
$$
**Explicação Teórica**:

- $\mathbf{m}$ representa as médias das distribuições gaussianas latentes para cada exemplo.
- $\mathbf{v}$ representa as variâncias correspondentes.

### 2. **Amostragem do Espaço Latente**

**Objetivo**: Amostrar $\mathbf{z}$ da distribuição $q(\mathbf{z}|\mathbf{x})$ usando o truque de reparametrização.

Utilizando o truque de reparametrização:

$$
\mathbf{z} = \mathbf{m} + \sqrt{\mathbf{v}} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

Suponhamos que $\boldsymbol{\epsilon}$ para cada exemplo seja:

Para o primeiro exemplo:

$$
\boldsymbol{\epsilon}_1 = \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix}
$$

Para o segundo exemplo:

$$
\boldsymbol{\epsilon}_2 = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}
$$

Calculando $\sqrt{\mathbf{v}}$:

$$
\sqrt{\mathbf{v}} = \begin{bmatrix}
\sqrt{0.2} & \sqrt{0.2} \\
\sqrt{0.1} & \sqrt{0.3} \\
\end{bmatrix} \approx \begin{bmatrix}
0.4472 & 0.4472 \\
0.3162 & 0.5477 \\
\end{bmatrix}
$$

Calculando $\mathbf{z}$ para cada exemplo:

Para o primeiro exemplo:

$$
\mathbf{z}_1 = \mathbf{m}_1 + \sqrt{\mathbf{v}_1} \odot \boldsymbol{\epsilon}_1 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} + \begin{bmatrix} 0.4472 \\ 0.4472 \end{bmatrix} \odot \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix} = \begin{bmatrix} 0.5 + 0.4472 \\ -0.5 - 0.4472 \end{bmatrix} \approx \begin{bmatrix} 0.9472 \\ -0.9472 \end{bmatrix}
$$

Para o segundo exemplo:

$$
\mathbf{z}_2 = \mathbf{m}_2 + \sqrt{\mathbf{v}_2} \odot \boldsymbol{\epsilon}_2 = \begin{bmatrix} -0.3 \\ 0.7 \end{bmatrix} + \begin{bmatrix} 0.3162 \\ 0.5477 \end{bmatrix} \odot \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} = \begin{bmatrix} -0.3 + 0.1581 \\ 0.7 + 0.2739 \end{bmatrix} \approx \begin{bmatrix} -0.1419 \\ 0.9739 \end{bmatrix}
$$

**Explicação Teórica**:

- O truque de reparametrização permite que o gradiente flua através da operação estocástica de amostragem.
- $\boldsymbol{\epsilon}$ é uma variável aleatória proveniente de uma distribuição normal padrão.

### 3. **Decodificação e Cálculo dos Logits**

**Objetivo**: Reconstruir os dados originais a partir das amostras latentes $\mathbf{z}$.

Suponhamos que o decoder produz os seguintes logits:

Para o primeiro exemplo:

$$
\text{logits}_1 = \begin{bmatrix} 0.8 \\ -0.2 \\ 0.5 \end{bmatrix}
$$

Para o segundo exemplo:

$$
\text{logits}_2 = \begin{bmatrix} -0.1 \\ 0.9 \\ 0.3 \end{bmatrix}
$$

Assim, os logits são:

$$
\text{logits} = \begin{bmatrix}
0.8 & -0.2 & 0.5 \\
-0.1 & 0.9 & 0.3 \\
\end{bmatrix}
$$

**Explicação Teórica**:

- Os logits são as saídas brutas antes da aplicação da função de ativação final (por exemplo, sigmoid).
- Representam as probabilidades não normalizadas das características reconstruídas.

### 4. **Cálculo da Perda de Reconstrução**

**Objetivo**: Quantificar o erro entre os dados originais $\mathbf{x}$ e os reconstruídos $\hat{\mathbf{x}}$.

A perda de reconstrução usando a entropia cruzada binária com logits é dada por:

$$
\text{BCE\_logits}(l, t) = \max(l, 0) - l \cdot t + \ln\left(1 + e^{-|l|}\right)
$$

Calculando a perda para cada elemento e exemplo.

#### Para o primeiro exemplo ($\mathbf{x}_1$):

Dados:

- $\text{logits}_1 = \begin{bmatrix} 0.8 \\ -0.2 \\ 0.5 \end{bmatrix}$
- $\mathbf{x}_1 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$

Calculando a perda de reconstrução elemento por elemento:

1. **Primeira característica ($t=0$, $l=0.8$):**

$$
\text{BCE\_logits}(0.8, 0) = 0.8 + \ln\left(1 + e^{-0.8}\right) \approx 0.8 + 0.3711 = 1.1711
$$

2. **Segunda característica ($t=1$, $l=-0.2$):**

$$
\text{BCE\_logits}(-0.2, 1) = 0.2 + \ln\left(1 + e^{0.2}\right) \approx 0.2 + 0.7981 = 0.9981
$$

3. **Terceira característica ($t=0$, $l=0.5$):**

$$
\text{BCE\_logits}(0.5, 0) = 0.5 + \ln\left(1 + e^{-0.5}\right) \approx 0.5 + 0.4741 = 0.9741
$$

Soma das perdas para o primeiro exemplo:

$$
\text{Rec}_1 = 1.1711 + 0.9981 + 0.9741 = 3.1433
$$

#### Para o segundo exemplo ($\mathbf{x}_2$):

Dados:

- $\text{logits}_2 = \begin{bmatrix} -0.1 \\ 0.9 \\ 0.3 \end{bmatrix}$
- $\mathbf{x}_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$

Calculando a perda de reconstrução elemento por elemento:

1. **Primeira característica ($t=1$, $l=-0.1$):**

$$
\text{BCE\_logits}(-0.1, 1) = 0.1 + \ln\left(1 + e^{0.1}\right) \approx 0.1 + 0.7444 = 0.8444
$$

2. **Segunda característica ($t=0$, $l=0.9$):**

$$
\text{BCE\_logits}(0.9, 0) = 0.9 + \ln\left(1 + e^{-0.9}\right) \approx 0.9 + 0.3412 = 1.2412
$$

3. **Terceira característica ($t=1$, $l=0.3$):**

$$
\text{BCE\_logits}(0.3, 1) = 0 + \ln\left(1 + e^{-0.3}\right) \approx 0 + 0.5544 = 0.5544
$$

Soma das perdas para o segundo exemplo:

$$
\text{Rec}_2 = 0.8444 + 1.2412 + 0.5544 = 2.6400
$$

Calculando a perda de reconstrução média sobre o lote:

$$
\text{Rec} = \frac{\text{Rec}_1 + \text{Rec}_2}{2} = \frac{3.1433 + 2.6400}{2} = 2.8916
$$

**Explicação Teórica**:

- A função de perda de reconstrução mede o quão bem o modelo consegue reconstruir os dados de entrada.
- A entropia cruzada binária com logits é usada para melhorar a estabilidade numérica.

### 5. **Cálculo da Divergência KL**

**Objetivo**: Medir a diferença entre a distribuição posterior $q(\mathbf{z}|\mathbf{x})$ e o prior $p(\mathbf{z})$.

Assumindo que o prior $p(\mathbf{z})$ é uma distribuição normal padrão com média zero e variância 1:

$$
\mathbf{m}_0 = \mathbf{0}, \quad \mathbf{v}_0 = \mathbf{1}
$$

A divergência KL entre duas gaussianas multivariadas diagonais é dada por:

$$
\text{KL}(q || p) = \frac{1}{2} \sum_{i=1}^{d} \left( \frac{v_i}{v_{0i}} + \frac{(m_i - m_{0i})^2}{v_{0i}} - 1 + \ln\left(\frac{v_{0i}}{v_i}\right) \right)
$$
Calculando para cada exemplo:

#### Para o primeiro exemplo ($\mathbf{m}_1$, $\mathbf{v}_1$):

1. **Termos individuais:**

- $\frac{v_i}{v_{0i}}$:

$$
\frac{v}{v_0} = \begin{bmatrix} \frac{0.2}{1.0} \\ \frac{0.2}{1.0} \end{bmatrix} = \begin{bmatrix} 0.2 \\ 0.2 \end{bmatrix}
$$

- $\frac{(m_i - m_{0i})^2}{v_{0i}}$:

$$
\frac{(m - m_0)^2}{v_0} = \begin{bmatrix} \frac{(0.5)^2}{1.0} \\ \frac{(-0.5)^2}{1.0} \end{bmatrix} = \begin{bmatrix} 0.25 \\ 0.25 \end{bmatrix}
$$

- $\ln\left(\frac{v_{0i}}{v_i}\right)$:

$$
\ln\left(\frac{v_0}{v}\right) = \begin{bmatrix} \ln\left(\frac{1.0}{0.2}\right) \\ \ln\left(\frac{1.0}{0.2}\right) \end{bmatrix} = \begin{bmatrix} \ln(5) \\ \ln(5) \end{bmatrix} \approx \begin{bmatrix} 1.6094 \\ 1.6094 \end{bmatrix}
$$

2. **Calculando a KL para o primeiro exemplo:**

$$
\text{KL}_1 = \frac{1}{2} \left( \left(0.2 + 0.25 - 1 + 1.6094\right) + \left(0.2 + 0.25 - 1 + 1.6094\right) \right) = \frac{1}{2} \times 2 \times 1.0594 = 1.0594
$$

#### Para o segundo exemplo ($\mathbf{m}_2$, $\mathbf{v}_2$):

1. **Termos individuais:**

- $\frac{v_i}{v_{0i}}$:

$$
\frac{v}{v_0} = \begin{bmatrix} 0.1 \\ 0.3 \end{bmatrix}
$$

- $\frac{(m_i - m_{0i})^2}{v_{0i}}$:

$$
\frac{(m - m_0)^2}{v_0} = \begin{bmatrix} 0.09 \\ 0.49 \end{bmatrix}
$$

- $\ln\left(\frac{v_{0i}}{v_i}\right)$:

$$
\ln\left(\frac{v_0}{v}\right) = \begin{bmatrix} \ln(10) \\ \ln\left(\frac{10}{3}\right) \end{bmatrix} \approx \begin{bmatrix} 2.3026 \\ 1.2030 \end{bmatrix}
$$

2. **Calculando a KL para o segundo exemplo:**

$$
\text{KL}_2 = \frac{1}{2} \left( \left(0.1 + 0.09 - 1 + 2.3026\right) + \left(0.3 + 0.49 - 1 + 1.2030\right) \right) = \frac{1}{2} \left(1.4916 + 0.9930\right) = \frac{1}{2} \times 2.4846 = 1.2423
$$

Calculando a divergência KL média sobre o lote:

$$
\text{KL} = \frac{\text{KL}_1 + \text{KL}_2}{2} = \frac{1.0594 + 1.2423}{2} = 1.1509
$$

**Explicação Teórica**:

- A divergência KL penaliza desvios da distribuição posterior em relação ao prior.
- Uma KL alta indica que a posterior está muito diferente do prior.

### 6. **Cálculo do NELBO (Negative Evidence Lower Bound)**

**Objetivo**: Combinar a perda de reconstrução e a divergência KL para formar o NELBO.

O NELBO é dado por:

$$
\text{NELBO} = \text{KL} + \text{Rec}
$$

Calculando o NELBO:

$$
\text{NELBO} = 1.1509 + 2.8916 = 4.0425
$$

**Explicação Teórica**:

- O NELBO combina a perda de reconstrução e a divergência KL.
- Minimizar o NELBO é equivalente a maximizar o ELBO, balanceando a fidelidade da reconstrução e a regularização imposta pela divergência KL.

## Resumo dos Resultados

- **Perda de Reconstrução (Rec)**: $2.8916$
- **Divergência KL (KL)**: $1.1509$
- **Negative ELBO (NELBO)**: $4.0425$

## Observações Finais

- **Truque de Reparametrização**: Essencial para permitir o treinamento do modelo usando gradiente descendente, apesar da amostragem estocástica.
- **Balanceamento entre KL e Reconstrução**: Um aspecto crítico no treinamento de VAEs é encontrar o equilíbrio entre a precisão da reconstrução e a regularização da distribuição latente.
- **Interpretação do NELBO**: Um NELBO menor indica um modelo que reconstrói bem os dados e cuja distribuição latente não se desvia muito do prior.

---

Este exemplo numérico ilustra cada etapa do cálculo da função de perda em um VAE, proporcionando uma compreensão prática dos conceitos teóricos envolvidos, utilizando equações matemáticas para uma apresentação mais clara e precisa.