# Exemplo Numérico Passo a Passo do MAF

Vamos criar um exemplo numérico detalhado para ilustrar a formulação do **Masked Autoregressive Flow (MAF)** conforme descrito:

$$
\log p(x) = \log p_z(f^{-1}(x)) + \sum_{j=1}^k \log \left|\det\left(\frac{\partial f_j^{-1}(x_j)}{\partial x_j}\right)\right|
$$

Onde:
- $p_z(z)$ é a distribuição prior (geralmente uma gaussiana padrão).
- $f_j$ são transformações invertíveis.

## Configuração do Exemplo

Vamos considerar um exemplo unidimensional para simplificar os cálculos. Definiremos:
- Uma distribuição prior $p_z(z)$ como uma **gaussiana padrão**: $p_z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.
- Dois fluxos (transformações) $f_1$ e $f_2$.

### Definição das Transformações

1. **Primeira Transformação $f_1$**:
   - $s_1 = \log 2$ (então $e^{s_1} = 2$).
   - $t_1 = 1$.
   - **Função direta**: $f_1(z) = e^{s_1} z + t_1 = 2z + 1$.
   - **Função inversa**: $f_1^{-1}(x) = \frac{x - t_1}{e^{s_1}} = \frac{x - 1}{2}$.

2. **Segunda Transformação $f_2$**:
   - $s_2 = \log 3$ (então $e^{s_2} = 3$).
   - $t_2 = -2$.
   - **Função direta**: $f_2(z) = e^{s_2} z + t_2 = 3z - 2$.
   - **Função inversa**: $f_2^{-1}(x) = \frac{x - t_2}{e^{s_2}} = \frac{x + 2}{3}$.

### Composição das Transformações

A transformação total é:

$$
x = f_2(f_1(z)) = f_2(2z + 1) = 3(2z + 1) - 2 = 6z + 1
$$

A função inversa composta é:

$$
z = f_1^{-1}(f_2^{-1}(x)) = f_1^{-1}\left(\frac{x + 2}{3}\right) = \frac{\left(\frac{x + 2}{3}\right) - 1}{2} = \frac{x - 1}{6}
$$

## Cálculo Passo a Passo

### Passo 1: Escolher um Valor para $x$

Vamos escolher $x = 7$.

### Passo 2: Calcular $z = f^{-1}(x)$

Usando a função inversa composta:

$$
z = \frac{x - 1}{6} = \frac{7 - 1}{6} = \frac{6}{6} = 1
$$

### Passo 3: Calcular $\log p_z(z)$

Como $z = 1$:

$$
p_z(1) = \frac{1}{\sqrt{2\pi}} e^{-1^2/2} = \frac{1}{\sqrt{2\pi}} e^{-0.5} \approx 0.24197
$$

Então:

$$
\log p_z(1) = \log\left(0.24197\right) \approx -1.41894
$$

### Passo 4: Calcular o Determinante Jacobiano

O determinante Jacobiano total é o produto dos determinantes das funções inversas:

1. **Determinante de $f_2^{-1}(x)$**:

   $$
   \frac{d f_2^{-1}(x)}{dx} = \frac{1}{e^{s_2}} = \frac{1}{3}
   $$

2. **Determinante de $f_1^{-1}(x)$** (aplicado em $z_1 = f_2^{-1}(x)$):

   $$
   \frac{d f_1^{-1}(z_1)}{dz_1} = \frac{1}{e^{s_1}} = \frac{1}{2}
   $$

3. **Determinante Jacobiano Total**:

   $$
   \left|\det\left(\frac{\partial f^{-1}(x)}{\partial x}\right)\right| = \left|\frac{1}{3} \times \frac{1}{2}\right| = \frac{1}{6}
   $$

### Passo 5: Calcular $\sum_{j=1}^k \log \left|\det\left(\frac{\partial f_j^{-1}(x_j)}{\partial x_j}\right)\right|$

$$
\log \left|\det\left(\frac{\partial f^{-1}(x)}{\partial x}\right)\right| = \log\left(\frac{1}{6}\right) = -\log 6 \approx -1.79176
$$

### Passo 6: Calcular $\log p(x)$

$$
\log p(x) = \log p_z(z) + \log \left|\det\left(\frac{\partial f^{-1}(x)}{\partial x}\right)\right| = -1.41894 - 1.79176 = -3.2107
$$

### Passo 7: Calcular $p(x)$

$$
p(x) = e^{\log p(x)} = e^{-3.2107} \approx 0.04045
$$

Alternativamente, podemos calcular diretamente:

$$
p(x) = p_z(z) \times \left|\det\left(\frac{\partial f^{-1}(x)}{\partial x}\right)\right| = 0.24197 \times \frac{1}{6} \approx 0.04033
$$

## Resumo dos Resultados

- **Valor de $z$**: 1
- **$\log p_z(z)$**: \(-1.41894\)
- **Determinante Jacobiano Total**: \(\frac{1}{6}\)
- **$\log$ Determinante Jacobiano**: \(-1.79176\)
- **$\log p(x)$**: \(-3.2107\)
- **$p(x)$**: $\approx 0.0404$

## Interpretação

Este exemplo ilustra como o MAF transforma uma variável $z$ de uma distribuição gaussiana padrão em uma variável $x$ com distribuição mais complexa. A transformação composta $x = f_2(f_1(z))$ alonga e desloca a distribuição original, enquanto o cálculo do determinante Jacobiano ajusta a densidade de probabilidade para refletir essa transformação.

## Observações Importantes

- **Composição de Transformações**: Cada transformação $f_j$ contribui para a complexidade da distribuição final, permitindo que o MAF modele distribuições complexas a partir de uma prior simples.
- **Cálculo do Jacobiano**: O determinante Jacobiano é crucial para ajustar a densidade de probabilidade após a transformação. No MAF, esse cálculo é eficiente devido à estrutura autoregressiva das transformações.
- **Flexibilidade do MAF**: Ao ajustar os parâmetros $s_j$ e $t_j$, podemos modelar uma ampla variedade de distribuições complexas.

## Referências

[2] Papamakarios, G., Pavlakou, T., & Murray, I. (2017). Masked Autoregressive Flow for Density Estimation.

[4] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions.

---

💡 **Destaque**: Este exemplo numérico demonstra como o MAF utiliza transformações invertíveis e o cálculo eficiente do determinante Jacobiano para modelar distribuições complexas de maneira computacionalmente tratável.