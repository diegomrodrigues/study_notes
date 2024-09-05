## Revisão de Variáveis Aleatórias Contínuas: CDF e PDF

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902084733466.png" alt="image-20240902084733466" style="zoom:80%;" />

### Introdução

As variáveis aleatórias contínuas são fundamentais na teoria da probabilidade e estatística, formando a base para muitos conceitos avançados em aprendizado de máquina e modelos generativos profundos. Esta revisão se concentra nos conceitos essenciais de função de densidade acumulada (CDF) e função de densidade de probabilidade (PDF) para variáveis aleatórias contínuas, fornecendo o ==alicerce necessário para compreender a fórmula de mudança de variáveis, crucial em fluxos normalizadores e outros modelos generativos [1].==

### Conceitos Fundamentais

| Conceito                                       | Explicação                                                   |
| ---------------------------------------------- | ------------------------------------------------------------ |
| **Variável Aleatória Contínua**                | ==Uma variável aleatória $X$ que pode assumir qualquer valor real dentro de um intervalo contínuo. [1]== |
| **Função de Distribuição Acumulada (CDF)**     | A função $F_X(a) = P(X \leq a)$ que representa a probabilidade de $X$ ser menor ou igual a $a$. [1] |
| **Função de Densidade de Probabilidade (PDF)** | A função $p_X(a) = F'_X(a) = \frac{dF_X(a)}{da}$, que é a ==derivada da CDF. [1]== |

> ⚠️ **Nota Importante**: A PDF não representa diretamente uma probabilidade, mas sim uma ==densidade de probabilidade==. ==A probabilidade é obtida integrando a PDF sobre um intervalo.==

### Função de Distribuição Acumulada (CDF)

![image-20240902084904257](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902084904257.png)

==A CDF, denotada por $F_X(a)$, é uma função fundamental que descreve completamente a distribuição de probabilidade de uma variável aleatória contínua $X$ [1].==

Propriedades principais:

1. Monotonicidade: $F_X(a)$ é não decrescente.
2. Limites: $\lim_{a \to -\infty} F_X(a) = 0$ e $\lim_{a \to \infty} F_X(a) = 1$.
3. Continuidade à direita: $\lim_{h \to 0^+} F_X(a+h) = F_X(a)$.

A CDF permite calcular probabilidades para intervalos:

$$P(a < X \leq b) = F_X(b) - F_X(a)$$

#### Questões Técnicas/Teóricas

1. Como você interpretaria graficamente a probabilidade $P(a < X \leq b)$ usando a CDF?
2. Explique por que a CDF é sempre não decrescente. Que implicação isso tem para a PDF?

### Função de Densidade de Probabilidade (PDF)

![image-20240902085036182](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902085036182.png)

A PDF, denotada por $p_X(a)$, é a derivada da CDF e fornece uma descrição mais intuitiva da distribuição de probabilidade [1]. 

Propriedades principais:

1. Não-negatividade: $p_X(a) \geq 0$ para todo $a$.
2. Área total unitária: $\int_{-\infty}^{\infty} p_X(a) da = 1$.
3. Relação com a CDF: $F_X(b) - F_X(a) = \int_a^b p_X(x) dx$.

> ✔️ **Ponto de Destaque**: ==A área sob a curva da PDF entre dois pontos representa a probabilidade de a variável aleatória cair nesse intervalo.==

#### Demonstração da Relação entre CDF e PDF

Partindo da definição da PDF como a derivada da CDF:

$$p_X(a) = \frac{dF_X(a)}{da}$$

Integrando ambos os lados de $a$ a $b$:

$$\int_a^b p_X(x) dx = \int_a^b \frac{dF_X(x)}{dx} dx$$

Pelo Teorema Fundamental do Cálculo:

$$\int_a^b p_X(x) dx = F_X(b) - F_X(a)$$

==Esta relação é fundamental para entender como a PDF e a CDF se conectam e como elas são usadas em conjunto para descrever completamente uma distribuição de probabilidade contínua.==

#### Questões Técnicas/Teóricas

1. Como você explicaria a diferença entre a interpretação de $p_X(a)$ e $F_X(a)$ para um valor específico $a$?
2. Se você tiver apenas a PDF de uma distribuição, como você calcularia a probabilidade de $X$ estar entre $a$ e $b$?

### Exemplos de Distribuições Paramétricas Comuns

1. **Distribuição Gaussiana (Normal)**:
   
   PDF: $p_X(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
   
   CDF: $F_X(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$

   Onde $\mu$ é a média e $\sigma$ é o desvio padrão [1].

2. **Distribuição Uniforme**:
   
   PDF: $p_X(x) = \frac{1}{b-a}$ para $a \leq x \leq b$, 0 caso contrário.
   
   CDF: $F_X(x) = \begin{cases} 0 & \text{para } x < a \\ \frac{x-a}{b-a} & \text{para } a \leq x < b \\ 1 & \text{para } x \geq b \end{cases}$

   Onde $a$ e $b$ são os limites inferior e superior, respectivamente [1].

> ❗ **Ponto de Atenção**: A escolha da distribuição paramétrica adequada é crucial em modelagem estatística e aprendizado de máquina, pois afeta diretamente a performance e interpretabilidade dos modelos.

### Variáveis Aleatórias Multidimensionais

![image-20240902085226975](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240902085226975.png)

Para variáveis aleatórias contínuas multidimensionais $\mathbf{X} = (X_1, \ldots, X_n)$, a PDF conjunta $p_\mathbf{X}(\mathbf{x})$ descreve a distribuição de probabilidade no espaço n-dimensional [1].

Exemplo: ==PDF da distribuição Gaussiana multivariada:==

$$p_\mathbf{X}(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n|\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

Onde $\boldsymbol{\mu}$ é o vetor de médias e $\Sigma$ é a matriz de covariância.

#### Implementação em Python

Aqui está um exemplo de como gerar e visualizar uma distribuição Gaussiana bivariada usando Python:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parâmetros da distribuição
mean = [0, 0]
cov = [[1, 0.5], [0.5, 2]]

# Cria a grade de pontos
x, y = np.mgrid[-3:3:.1, -3:3:.1]
pos = np.dstack((x, y))

# Cria a distribuição
rv = multivariate_normal(mean, cov)

# Calcula a PDF
z = rv.pdf(pos)

# Visualização
plt.contourf(x, y, z, cmap='viridis')
plt.colorbar(label='PDF')
plt.title('Distribuição Gaussiana Bivariada')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
```

Este código gera um gráfico de contorno da PDF de uma distribuição Gaussiana bivariada, ilustrando como a densidade de probabilidade varia no espaço bidimensional.

### Conclusão

A compreensão profunda das funções de distribuição acumulada (CDF) e de densidade de probabilidade (PDF) para variáveis aleatórias contínuas é fundamental para diversos campos da ciência de dados e aprendizado de máquina. Estes conceitos formam a base para o entendimento de distribuições de probabilidade mais complexas e são essenciais para o desenvolvimento e análise de modelos estatísticos avançados, incluindo fluxos normalizadores e outros modelos generativos profundos [1].

A relação entre CDF e PDF, expressa matematicamente e visualmente, oferece insights valiosos sobre o comportamento de variáveis aleatórias. A capacidade de manipular e interpretar essas funções é crucial para tarefas como estimação de parâmetros, inferência estatística e modelagem de incertezas em aprendizado de máquina.

Ao dominar esses conceitos fundamentais, cientistas de dados e engenheiros de machine learning estão melhor equipados para abordar problemas complexos envolvendo distribuições de probabilidade, desde a modelagem de fenômenos naturais até o desenvolvimento de algoritmos de aprendizado profundo de última geração.

### Questões Avançadas

1. Como você usaria o conceito de CDF para gerar amostras de uma distribuição arbitrária usando o método da transformação inversa? Explique o processo e discuta possíveis limitações.

2. Considerando uma mistura de distribuições Gaussianas, como você derivaria a PDF e a CDF resultantes? Discuta as implicações computacionais e práticas de trabalhar com tais misturas em modelos de aprendizado de máquina.

3. Explique como o conceito de entropia diferencial se relaciona com a PDF de uma variável aleatória contínua. Como isso pode ser usado para medir a incerteza em modelos probabilísticos contínuos?

4. Em fluxos normalizadores, como a relação entre PDF e CDF é explorada para transformar distribuições simples em distribuições mais complexas? Discuta as vantagens e desafios desta abordagem.

5. Considere uma transformação não-linear $Y = g(X)$ de uma variável aleatória contínua $X$. Como você expressaria a PDF de $Y$ em termos da PDF de $X$ e da função $g$? Discuta as implicações desta transformação para modelos generativos.

### Referências

[1] "Let X be a continuous random variable. The cumulative density function (CDF) of X is F_X(a) = P(X ≤ a). The probability density function (pdf) of X is p_X(a) = F'_X(a) = dF_X(a)/da" (Trecho de Normalizing Flow Models - Lecture Notes)

[2] "Typically consider parameterized densities: Gaussian: X ~ N(μ, σ) if p_X(x) = (1/(σ√(2π))) exp(-(x-μ)²/(2σ²)). Uniform: X ~ U(a, b) if p_X(x) = 1/(b-a) 1[a ≤ x ≤ b]" (Trecho de Normalizing Flow Models - Lecture Notes)

[3] "If X is a continuous random vector, we can usually represent it using its joint probability density function: Gaussian: if p_X(x) = (1/√((2π)^n|Σ|)) exp(-(1/2)(x-μ)^T Σ^(-1)(x-μ))" (Trecho de Normalizing Flow Models - Lecture Notes)