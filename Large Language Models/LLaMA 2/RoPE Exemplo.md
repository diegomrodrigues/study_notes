# Explicando Detalhadamente a Matemática do RoPE com um Exemplo Numérico

## Introdução

O **Rotary Position Embedding (RoPE)** é uma técnica inovadora introduzida no **RoFormer** para incorporar informações posicionais em modelos Transformer. Ao contrário das abordagens tradicionais que adicionam embeddings posicionais às representações das palavras, o RoPE aplica uma **rotação** aos embeddings das palavras, baseada em suas posições na sequência. Isso permite capturar tanto informações de posição absoluta quanto relativa, melhorando a capacidade do modelo de lidar com dependências de longo alcance.

Neste guia, vamos explorar detalhadamente a matemática por trás do RoPE através de um exemplo numérico. Abordaremos cada passo, desde a definição das matrizes de rotação até o cálculo das pontuações de atenção.

## 1. Conceitos Básicos

Antes de mergulharmos no exemplo numérico, vamos revisar os conceitos fundamentais:

- **Embeddings das Palavras ($x_m$)**: Vetores que representam palavras em um espaço de alta dimensão.
- **Matrizes de Projeção ($W_q, W_k$)**: Matrizes que transformam os embeddings das palavras em **queries** e **keys**.
- **Matriz de Rotação ($R_{\Theta,m}^d$)**: Matriz que aplica uma rotação aos embeddings, incorporando informações posicionais.

A fórmula central do RoPE é:

$$
f_{q,k}(x_m, m) = R_{\Theta,m}^d W_{q,k} x_m
$$

Onde:

- $f_{q,k}(x_m, m)$ é o query ou key transformado.
- $R_{\Theta,m}^d$ é a matriz de rotação para a posição $m$.

## 2. Definindo os Parâmetros do Exemplo

Para facilitar o cálculo, vamos definir parâmetros simples:

- **Dimensão dos Embeddings ($d$)**: 4
- **Posições das Palavras**: $m = 2$, $n = 3$
- **Embeddings das Palavras**:
  - $x_2 = [1.0, 2.0, 3.0, 4.0]^T$
  - $x_3 = [1.5, 2.5, 3.5, 4.5]^T$
- **Matrizes de Projeção ($W_q, W_k$)**: Identidade (para simplificar)
- **Frequências ($\theta_i$)**:
  - Para $d = 4$, temos $i = 1, 2$
  - $\theta_1 = 1$
  - $\theta_2 = 0.01$

## 3. Calculando as Frequências $\theta_i$

As frequências $\theta_i$ são calculadas usando a fórmula:

$$
\theta_i = 10000^{-2(i-1)/d}
$$

- Para $i = 1$:

  $$
  \theta_1 = 10000^{-2(1-1)/4} = 10000^0 = 1
  $$

- Para $i = 2$:

  $$
  \theta_2 = 10000^{-2(2-1)/4} = 10000^{-0.5} = \frac{1}{\sqrt{10000}} = 0.01
  $$

## 4. Calculando os Ângulos de Rotação

Para cada par de dimensões $(i, i+1)$ e posição $m$, o ângulo de rotação é:

$$
\theta_{i,m} = m \times \theta_i
$$

- Para $m = 2$:

  - $\theta_{1,2} = 2 \times 1 = 2$ radianos
  
- Para $n = 3$:

  - $\theta_{1,3} = 3 \times 1 = 3$ radianos
  - $\theta_{2,3} = 3 \times 0.01 = 0.03$ radianos

## 5. Construindo as Matrizes de Rotação

A matriz de rotação para cada par de dimensões é:

$$
R(\theta) = \begin{pmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{pmatrix}
$$

### Para $m = 2$:

- **Primeiro Par ($\theta = 2$)**:

  $$
  R(2) = \begin{pmatrix}
  \cos 2 & -\sin 2 \\
  \sin 2 & \cos 2
  \end{pmatrix} \approx \begin{pmatrix}
  -0.4161 & -0.9093 \\
  0.9093 & -0.4161
  \end{pmatrix}
  $$

- **Segundo Par ($\theta = 0.02$)**:

  $$
  R(0.02) = \begin{pmatrix}
  0.9998 & -0.0200 \\
  0.0200 & 0.9998
  \end{pmatrix}
  $$

### Para $n = 3$:

- **Primeiro Par ($\theta = 3$)**:

  $$
  R(3) = \begin{pmatrix}
  -0.9900 & -0.1411 \\
  0.1411 & -0.9900
  \end{pmatrix}
  $$

- **Segundo Par ($\theta = 0.03$)**:

  $$
  R(0.03) = \begin{pmatrix}
  0.9996 & -0.0300 \\
  0.0300 & 0.9996
  \end{pmatrix}
  $$

## 6. Aplicando as Matrizes de Rotação aos Embeddings

### Para $m = 2$:

- **Embedding Original**: $x_2 = [1.0, 2.0, 3.0, 4.0]^T$
- **Aplicando $R(2)$ ao Primeiro Par ($x_0, x_1$)**:

  $$
  \begin{pmatrix}
  x_0' \\
  x_1'
  \end{pmatrix} = R(2) \begin{pmatrix}
  1.0 \\
  2.0
  \end{pmatrix} = \begin{pmatrix}
  -0.4161 \times 1.0 + (-0.9093) \times 2.0 \\
  0.9093 \times 1.0 + (-0.4161) \times 2.0
  \end{pmatrix} = \begin{pmatrix}
  -2.2347 \\
  0.0770
  \end{pmatrix}
  $$

- **Aplicando $R(0.02)$ ao Segundo Par ($x_2, x_3$)**:

  $$
  \begin{pmatrix}
  x_2' \\
  x_3'
  \end{pmatrix} = R(0.02) \begin{pmatrix}
  3.0 \\
  4.0
  \end{pmatrix} = \begin{pmatrix}
  2.9194 \\
  4.0592
  \end{pmatrix}
  $$

- **Embedding Transformado**:

  $$
  x_2' = [-2.2347, 0.0770, 2.9194, 4.0592]^T
  $$

### Para $n = 3$:

- **Embedding Original**: $x_3 = [1.5, 2.5, 3.5, 4.5]^T$
- **Aplicando $R(3)$ ao Primeiro Par ($x_0, x_1$)**:

  $$
  \begin{pmatrix}
  x_0' \\
  x_1'
  \end{pmatrix} = R(3) \begin{pmatrix}
  1.5 \\
  2.5
  \end{pmatrix} = \begin{pmatrix}
  -1.8378 \\
  -2.2633
  \end{pmatrix}
  $$

- **Aplicando $R(0.03)$ ao Segundo Par ($x_2, x_3$)**:

  $$
  \begin{pmatrix}
  x_2' \\
  x_3'
  \end{pmatrix} = R(0.03) \begin{pmatrix}
  3.5 \\
  4.5
  \end{pmatrix} = \begin{pmatrix}
  3.3634 \\
  4.6030
  \end{pmatrix}
  $$

- **Embedding Transformado**:

  $$
  x_3' = [-1.8378, -2.2633, 3.3634, 4.6030]^T
  $$

## 7. Calculando a Pontuação de Atenção

A pontuação de atenção entre as posições $m$ e $n$ é o produto interno dos embeddings transformados:

$$
\text{score}_{mn} = (x_m')^T x_n'
$$

### Cálculo:

1. **Produto dos Componentes**:

   - \(-2.2347 \times -1.8378 = 4.1069\)
   - \(0.0770 \times -2.2633 = -0.1742\)
   - \(2.9194 \times 3.3634 = 9.8252\)
   - \(4.0592 \times 4.6030 = 18.7025\)

2. **Somando os Produtos**:

   $$
   \text{score}_{mn} = 4.1069 - 0.1742 + 9.8252 + 18.7025 = 32.4604
   $$

## 8. Comparando com a Atenção Convencional

Para entender o impacto do RoPE, vamos calcular a pontuação de atenção usando os embeddings originais (sem rotação):

$$
\text{score}_{mn}^{\text{orig}} = x_m^T x_n
$$

### Cálculo:

1. **Produto dos Componentes**:

   - \(1.0 \times 1.5 = 1.5\)
   - \(2.0 \times 2.5 = 5.0\)
   - \(3.0 \times 3.5 = 10.5\)
   - \(4.0 \times 4.5 = 18.0\)

2. **Somando os Produtos**:

   $$
   \text{score}_{mn}^{\text{orig}} = 1.5 + 5.0 + 10.5 + 18.0 = 35.0
   $$

**Observação**: A pontuação de atenção usando o RoPE é ligeiramente menor, indicando que o RoPE ajusta as pontuações com base na posição relativa.

## 9. Observando o Decaimento com a Distância

Para ilustrar como o RoPE incorpora dependências de posição relativa, vamos calcular a pontuação de atenção entre $m = 2$ e $n = 4$.

### Passos Adicionais:

1. **Atualizando os Ângulos de Rotação para $n = 4$**:

   - $\theta_{1,4} = 4 \times 1 = 4$ radianos
   - $\theta_{2,4} = 4 \times 0.01 = 0.04$ radianos

2. **Calculando as Matrizes de Rotação para $n = 4$**:

   - $R(4)$ e $R(0.04)$

3. **Aplicando as Matrizes ao Embedding de $n = 4$**:

   - Obter $x_4'$

4. **Calculando a Pontuação de Atenção**:

   $$
   \text{score}_{m=2, n=4} = (x_2')^T x_4'
   $$

### Resultado:

- A pontuação de atenção diminui ainda mais, ilustrando o **decaimento com o aumento da distância relativa**.

## 10. Conclusão

O **RoPE** aplica uma rotação aos embeddings das palavras baseada em suas posições, o que permite:

- **Capturar Dependências de Longo Alcance**: O decaimento das pontuações de atenção com a distância ajuda o modelo a considerar a relevância das palavras distantes.
- **Incorporar Informações de Posição Relativa**: A rotação diferencia não apenas as posições absolutas, mas também as relações entre posições.

Este exemplo numérico detalhado demonstra como o RoPE transforma os embeddings e como isso afeta as pontuações de atenção no mecanismo de self-attention. Através dessa técnica, o **RoFormer** melhora a capacidade dos modelos Transformer de lidar com sequências mais longas e capturar relações contextuais complexas.

---

**Referências**:

- [1] Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- [2] Vaswani, A., et al. (2017). Attention is All You Need.