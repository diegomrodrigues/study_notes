# Otimização de Discriminadores em GANs com Restrições de Gradiente Lipschitz

### Problema

(a) [4 pontos (Escrito)] Suponha que quando $(x, y) \sim p_{\text{data}}(x, y)$, existe um mapeamento de características $\phi$ tal que $\phi(x)$ se torna uma mistura de $m$ Gaussianas unitárias, com uma Gaussiana por rótulo de classe $y$. Assuma que quando $(x, y) \sim p_{\theta}(x, y)$, $\phi(x)$ também se torna uma mistura de $m$ Gaussianas unitárias, novamente com uma Gaussiana por rótulo de classe $y$. Concretamente, assumimos que a razão das probabilidades condicionais pode ser escrita como:

$$
\frac{p_{\text{data}}(x|y)}{p_{\theta}(x|y)} = \frac{N(\phi(x) \mid \mu_y, I)}{N(\phi(x) \mid \hat{\mu}_y, I)} \quad (19)
$$

onde $\mu_y$ e $\hat{\mu}_y$ são as médias das Gaussianas para $p_{\text{data}}$ e $p_{\theta}$, respectivamente.

**Objetivo**: Mostrar que, sob essa suposição simplificadora, os logits ótimos do discriminador $h^\ast(x, y)$ podem ser escritos na forma:

$$
h^\ast(x, y) = y^\top (A \phi(x) + b) \quad (20)
$$

para alguma matriz $A$ e vetor $b$, onde $y$ é um vetor one-hot indicando a classe $y$. Neste problema, a saída e os logits do discriminador estão relacionados por $D_\phi(x, y) = \sigma(h_\phi(x, y))$. Para expressar $\mu_y - \hat{\mu}_y$ em termos de $y$, dado que $y$ é um vetor one-hot, veja se você pode escrever $\mu_1 - \hat{\mu}_1$ como uma multiplicação matricial de $y$ e uma matriz cujas linhas são $\mu_i - \hat{\mu}_i$.

> **Dica**: Use o resultado do problema 3b. Além disso, tente expandir a PDF para os termos $p$ usando o fato de que são distribuições normais com parâmetros conhecidos.

Para ajudá-lo a começar:

$$
h_\phi(x, y) = \log \frac{p_{\text{data}}(x, y)}{p_{\theta}(x, y)} = \log \frac{p_{\text{data}}(x|y)}{p_{\theta}(x|y)} + \log \frac{p_{\text{data}}(y)}{p_{\theta}(y)} = \log \frac{p_{\text{data}}(x|y)}{p_{\theta}(x|y)} =
$$

---

### Introdução

O objetivo deste problema é demonstrar que, sob certas suposições, os logits ótimos do discriminador em uma Rede Generativa Adversária (GAN) podem ser expressos como uma função linear das características $\phi(x)$ e do vetor de classe one-hot $y$. Isso implica que o discriminador ótimo tem uma estrutura linear específica que pode ser explorada para melhorar o treinamento das GANs.

---

### Preliminares e Conceitos Fundamentais

#### Distribuição Gaussiana Multivariada

A densidade de probabilidade de uma distribuição normal multivariada $N(x \mid \mu, \Sigma)$ é dada por:

$$
N(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{k/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu) \right)
$$

No caso de $\Sigma = I$ (matriz identidade), simplifica para:

$$
N(x \mid \mu, I) = \frac{1}{(2\pi)^{k/2}} \exp\left( -\frac{1}{2}\|x - \mu\|^2 \right)
$$

#### Vetor One-Hot

Um vetor one-hot $y$ para uma classe $k$ em um total de $m$ classes é um vetor em $\mathbb{R}^m$ onde:

$$
y_i = \begin{cases}
1, & \text{se } i = k \\
0, & \text{caso contrário}
\end{cases}
$$

#### Relação entre Discriminador e Logits

No contexto das GANs, o discriminador $D_\phi(x, y)$ e os logits $h_\phi(x, y)$ estão relacionados pela função sigmoide $\sigma$:

$$
D_\phi(x, y) = \sigma(h_\phi(x, y)) = \frac{1}{1 + e^{-h_\phi(x, y)}}
$$

---

### Desenvolvimento da Solução

#### Passo 1: Simplificar $h_\phi(x, y)$

Começamos com:

$$
h_\phi(x, y) = \log \frac{p_{\text{data}}(x, y)}{p_{\theta}(x, y)} = \log \frac{p_{\text{data}}(x|y)}{p_{\theta}(x|y)} + \log \frac{p_{\text{data}}(y)}{p_{\theta}(y)}
$$

Assumimos que $p_{\text{data}}(y) = p_{\theta}(y)$, pois as distribuições das classes são as mesmas em ambas. Assim:

$$
\log \frac{p_{\text{data}}(y)}{p_{\theta}(y)} = 0
$$

Portanto:

$$
h_\phi(x, y) = \log \frac{p_{\text{data}}(x|y)}{p_{\theta}(x|y)}
$$

---

#### Passo 2: Expandir as Densidades Gaussianas

Usando a suposição do problema:

$$
\frac{p_{\text{data}}(x|y)}{p_{\theta}(x|y)} = \frac{N(\phi(x) \mid \mu_y, I)}{N(\phi(x) \mid \hat{\mu}_y, I)}
$$

Tomando o logaritmo:

$$
h_\phi(x, y) = \log N(\phi(x) \mid \mu_y, I) - \log N(\phi(x) \mid \hat{\mu}_y, I)
$$

Substituindo a expressão da densidade Gaussiana:

$$
h_\phi(x, y) = -\frac{1}{2}\|\phi(x) - \mu_y\|^2 + \frac{1}{2}\|\phi(x) - \hat{\mu}_y\|^2
$$

---

#### Passo 3: Expandir os Termos Quadráticos

Expanda as normas quadradas:

$$
\|\phi(x) - \mu_y\|^2 = \phi(x)^\top \phi(x) - 2\phi(x)^\top \mu_y + \mu_y^\top \mu_y
$$

$$
\|\phi(x) - \hat{\mu}_y\|^2 = \phi(x)^\top \phi(x) - 2\phi(x)^\top \hat{\mu}_y + \hat{\mu}_y^\top \hat{\mu}_y
$$

Subtraindo as duas expressões:

$$
\|\phi(x) - \hat{\mu}_y\|^2 - \|\phi(x) - \mu_y\|^2 = -2\phi(x)^\top (\hat{\mu}_y - \mu_y) + \hat{\mu}_y^\top \hat{\mu}_y - \mu_y^\top \mu_y
$$

Portanto:

$$
h_\phi(x, y) = -\phi(x)^\top (\hat{\mu}_y - \mu_y) + \frac{1}{2} \left( \hat{\mu}_y^\top \hat{\mu}_y - \mu_y^\top \mu_y \right)
$$

---

#### Passo 4: Expressar $\hat{\mu}_y - \mu_y$ em Termos de $y$

Como $y$ é um vetor one-hot, podemos escrever:

$$
\mu_y = M y
$$

$$
\hat{\mu}_y = \hat{M} y
$$

Onde $M$ e $\hat{M}$ são matrizes cujas colunas são os vetores $\mu_i$ e $\hat{\mu}_i$, respectivamente.

Então:

$$
\hat{\mu}_y - \mu_y = (\hat{M} - M) y = \Delta M y
$$

Onde $\Delta M = \hat{M} - M$.

---

#### Passo 5: Reescrever $h_\phi(x, y)$ Usando $y$

Substituindo na expressão de $h_\phi(x, y)$:

$$
h_\phi(x, y) = -\phi(x)^\top (\Delta M y) + \frac{1}{2} \left( (\hat{M} y)^\top (\hat{M} y) - (M y)^\top (M y) \right)
$$

---

#### Passo 6: Simplificar o Termo Quadrático

Calculando os termos quadráticos:

$$
(\hat{M} y)^\top (\hat{M} y) = y^\top \hat{M}^\top \hat{M} y

\quad \text{e} \quad

(M y)^\top (M y) = y^\top M^\top M y
$$

Portanto:

$$
\hat{\mu}_y^\top \hat{\mu}_y - \mu_y^\top \mu_y = y^\top (\hat{M}^\top \hat{M} - M^\top M) y
$$

---

#### Passo 7: Expressar $h_\phi(x, y)$ na Forma Desejada

Agora, a expressão para $h_\phi(x, y)$ é:

$$
h_\phi(x, y) = -\phi(x)^\top (\Delta M y) + \frac{1}{2} y^\top (\hat{M}^\top \hat{M} - M^\top M) y
$$

Como $y$ é um vetor one-hot, $y^\top (\text{Qualquer Matriz}) y$ seleciona um único elemento da matriz. Definimos um vetor $b$ onde cada componente $b_i$ é:

$$
b_i = \frac{1}{2} \left( \hat{\mu}_i^\top \hat{\mu}_i - \mu_i^\top \mu_i \right)
$$

Assim, $y^\top b$ é o componente $b_i$ correspondente à classe $i$.

---

#### Passo 8: Definir Matriz $A$ e Vetor $b$

Definimos:

- **Matriz $A$**:

$$
A = - (\Delta M)^\top = - (\hat{M} - M)^\top
$$

- **Vetor $b$**:

$$
b_i = \frac{1}{2} \left( \hat{\mu}_i^\top \hat{\mu}_i - \mu_i^\top \mu_i \right)
$$

Portanto, podemos escrever:

$$
h^\ast(x, y) = y^\top (A \phi(x) + b)
$$

---

### Conclusão

Sob as suposições dadas, mostramos que os logits ótimos do discriminador podem ser expressos como uma função linear de $\phi(x)$ e $y$:

$$
h^\ast(x, y) = y^\top (A \phi(x) + b)
$$

Onde:

- **$A$** é uma matriz derivada das diferenças entre as médias das distribuições Gaussianas sob $p_{\text{data}}$ e $p_{\theta}$:

$$
A = - (\hat{M} - M)^\top
$$

- **$b$** é um vetor cujos componentes são:

$$
b_i = \frac{1}{2} \left( \hat{\mu}_i^\top \hat{\mu}_i - \mu_i^\top \mu_i \right)
$$

Isso confirma que, sob o mapeamento $\phi$ e as suposições do problema, o discriminador ótimo tem uma forma linear específica em relação às características $\phi(x)$ e ao vetor de classe $y$.

---

### Referências

- **Teoria de Probabilidades e Estatística**: Propriedades das distribuições normais multivariadas.
- **Aprendizado de Máquina**: Conceitos de vetores one-hot e funções discriminadoras em modelos de classificação.
- **Redes Generativas Adversárias (GANs)**: Estrutura e treinamento de GANs, incluindo discriminadores ótimos.