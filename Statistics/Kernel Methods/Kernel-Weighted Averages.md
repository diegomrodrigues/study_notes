## Kernel-Weighted Averages: Suavização Local em Regressão Não-Paramétrica

![image-20240806134800066](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240806134800066.png)

## Introdução

Os métodos de suavização por kernel são uma classe fundamental de técnicas não-paramétricas em estatística e aprendizado de máquina, especialmente úteis para estimação de funções de regressão e densidade. Este resumo se concentra nos suavizadores de kernel unidimensionais, que podem ser vistos como uma sofisticação das médias móveis simples, oferecendo maior flexibilidade e controle sobre o processo de suavização [1].

### Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Kernel**                    | Uma função de ponderação que atribui pesos às observações com base em sua distância do ponto alvo. [2] |
| **Bandwidth**                 | Parâmetro que controla a largura do kernel, determinando o grau de suavização. [3] |
| **Nadaraya-Watson Estimator** | Um estimador não-paramétrico que utiliza kernels para estimar a função de regressão. [4] |

> ✔️ **Ponto de Destaque**: A escolha do kernel e do bandwidth são cruciais para o desempenho do estimador, afetando o trade-off entre viés e variância.

### Formulação Matemática do Estimador de Nadaraya-Watson

O estimador de Nadaraya-Watson é definido como [4]:

$$
\hat{f}(x_0) = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i)y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)}
$$

Onde:
- $\hat{f}(x_0)$ é a estimativa da função no ponto $x_0$
- $K_\lambda(x_0, x_i)$ é a função kernel com bandwidth $\lambda$
- $(x_i, y_i)$ são os pares de observações

Esta formulação pode ser interpretada como uma média ponderada das observações $y_i$, onde os pesos são determinados pela função kernel.

#### Questões Técnicas/Teóricas

1. Como o estimador de Nadaraya-Watson se comporta nos limites do domínio dos dados? Explique o conceito de "viés de fronteira".
2. Derive a expressão para o viés e a variância do estimador de Nadaraya-Watson. Como esses termos são afetados pelo bandwidth $\lambda$?

### Funções de Kernel Comuns

Vários tipos de kernel são utilizados na prática, cada um com características próprias:

1. **Kernel Epanechnikov** [5]:
   
   $$
   K(u) = \frac{3}{4}(1-u^2)I(|u|\leq 1)
   $$

2. **Kernel Gaussiano**:
   
   $$
   K(u) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}u^2}
   $$

3. **Kernel Tricúbico**:
   
   $$
   K(u) = (1-|u|^3)^3I(|u|\leq 1)
   $$

Onde $u = \frac{|x-x_0|}{\lambda}$ e $I()$ é a função indicadora.

> ⚠️ **Nota Importante**: A escolha do kernel geralmente tem menos impacto no desempenho do que a seleção do bandwidth.

### Seleção do Bandwidth

A seleção do bandwidth $\lambda$ é crucial e afeta diretamente o trade-off entre viés e variância [6]:

- **Bandwidth pequeno**: Resulta em estimativas com baixo viés, mas alta variância.
- **Bandwidth grande**: Produz estimativas com baixa variância, mas alto viés.

Métodos comuns para seleção do bandwidth incluem:

1. **Validação Cruzada**: Minimiza o erro de predição estimado.
2. **Plug-in Methods**: Estimam diretamente o bandwidth ótimo usando estimativas do viés e da variância.
3. **Rule of Thumb**: Fornece uma estimativa rápida baseada na distribuição dos dados.

#### Questões Técnicas/Teóricas

1. Descreva o procedimento de validação cruzada leave-one-out para seleção do bandwidth em regressão kernel. Quais são suas vantagens e desvantagens?
2. Como o bandwidth ótimo se comporta assintoticamente em relação ao tamanho da amostra $N$? Justifique matematicamente.

### Propriedades Assintóticas

Sob certas condições de regularidade, o estimador de Nadaraya-Watson é consistente e assintoticamente normal [7]. Para um ponto fixo $x_0$:

$$
\sqrt{Nh}(\hat{f}(x_0) - f(x_0)) \xrightarrow{d} N(0, V(x_0))
$$

Onde $V(x_0)$ é a variância assintótica que depende da função de densidade subjacente e da variância do erro.

> ❗ **Ponto de Atenção**: A taxa de convergência é afetada pela dimensionalidade dos dados, um fenômeno conhecido como "maldição da dimensionalidade".

Vou demonstrar passo a passo a propriedade assintótica do estimador de Nadaraya-Watson. Esta demonstração envolve conceitos avançados de estatística e teoria assintótica.

Passo 1: Definição do estimador de Nadaraya-Watson

O estimador é dado por:

$$ \hat{f}(x_0) = \frac{\sum_{i=1}^N K_h(x_0 - X_i)Y_i}{\sum_{i=1}^N K_h(x_0 - X_i)} $$

onde $K_h(u) = \frac{1}{h}K(\frac{u}{h})$, e $h$ é o parâmetro de bandwidth.

Passo 2: Decomposição do estimador

Podemos reescrever o estimador como:

$$ \hat{f}(x_0) = \frac{\frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})Y_i}{\frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})} $$

Passo 3: Análise do numerador e denominador

Definimos:

$$ \hat{g}(x_0) = \frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})Y_i $$
$$ \hat{f}_X(x_0) = \frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h}) $$

Onde $\hat{f}_X(x_0)$ é um estimador de densidade kernel para a densidade de $X$.

Passo 4: Convergência do denominador

Sob condições de regularidade, pode-se mostrar que:

$$ \hat{f}_X(x_0) \xrightarrow{p} f_X(x_0) $$

onde $f_X(x_0)$ é a verdadeira densidade de $X$ em $x_0$.

Passo 5: Análise do numerador

Podemos decompor $\hat{g}(x_0)$ como:

$$ \hat{g}(x_0) = \frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})[f(X_i) + \epsilon_i] $$

onde $Y_i = f(X_i) + \epsilon_i$.

Passo 6: Expansão de Taylor

Expandindo $f(X_i)$ em torno de $x_0$:

$$ f(X_i) = f(x_0) + f'(x_0)(X_i - x_0) + \frac{1}{2}f''(\xi_i)(X_i - x_0)^2 $$

onde $\xi_i$ está entre $X_i$ e $x_0$.

Passo 7: Substituição na expressão de $\hat{g}(x_0)$

$$ \hat{g}(x_0) = f(x_0)\hat{f}_X(x_0) + \frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})[\frac{1}{2}f''(\xi_i)(X_i - x_0)^2 + \epsilon_i] $$

Passo 8: Análise do termo de erro

O termo de erro pode ser decomposto em um termo de viés e um termo de variância:

$$ \sqrt{Nh}(\hat{f}(x_0) - f(x_0)) = \frac{\sqrt{Nh}}{\hat{f}_X(x_0)}[\hat{g}(x_0) - f(x_0)\hat{f}_X(x_0)] $$

Passo 9: Convergência do termo de viés

Pode-se mostrar que o termo de viés converge para uma constante:

$$ \frac{\sqrt{Nh}}{\hat{f}_X(x_0)}\frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})[\frac{1}{2}f''(\xi_i)(X_i - x_0)^2] \xrightarrow{p} \frac{1}{2}h^2f''(x_0)\mu_2(K) $$

onde $\mu_2(K)$ é o segundo momento do kernel $K$.

Passo 10: Convergência do termo de variância

O termo de variância converge em distribuição para uma normal:

$$ \frac{\sqrt{Nh}}{\hat{f}_X(x_0)}\frac{1}{Nh}\sum_{i=1}^N K(\frac{x_0 - X_i}{h})\epsilon_i \xrightarrow{d} N(0, \frac{\sigma^2(x_0)}{f_X(x_0)}\int K^2(u)du) $$

onde $\sigma^2(x_0) = Var(Y|X=x_0)$.

Passo 11: Combinação dos resultados

Combinando os resultados dos passos 9 e 10, e assumindo que $h \to 0$ e $Nh \to \infty$ quando $N \to \infty$, obtemos:

$$ \sqrt{Nh}(\hat{f}(x_0) - f(x_0)) \xrightarrow{d} N(0, V(x_0)) $$

onde $V(x_0) = \frac{\sigma^2(x_0)}{f_X(x_0)}\int K^2(u)du$.

Esta demonstração estabelece a normalidade assintótica do estimador de Nadaraya-Watson, mostrando que ele converge para a verdadeira função de regressão $f(x_0)$ com uma taxa de $\sqrt{Nh}$ e uma variância assintótica $V(x_0)$.

### Extensões e Variantes

1. **Regressão Local Linear**: Uma extensão que ajusta localmente uma linha reta, reduzindo o viés de fronteira [8].

   $$
   \min_{\alpha(x_0),\beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i)[y_i - \alpha(x_0) - \beta(x_0)x_i]^2
   $$

2. **Kernel Adaptativo**: Utiliza um bandwidth variável que se adapta à densidade local dos dados.

3. **Kernel Multivariado**: Extensão para regressão em múltiplas dimensões, utilizando kernels produto ou radiais.

#### Questões Técnicas/Teóricas

1. Compare matematicamente o viés do estimador de Nadaraya-Watson com o da regressão local linear. Por que a regressão local linear tem melhor desempenho nas fronteiras?
2. Descreva como implementar um kernel adaptativo. Quais são os desafios computacionais envolvidos?

### Implementação em Python

Aqui está um exemplo simplificado de implementação do estimador de Nadaraya-Watson em Python:

```python
import numpy as np
from scipy.stats import norm

def nadaraya_watson(x, X, Y, h):
    kernel = lambda u: norm.pdf(u, loc=0, scale=1)
    weights = kernel((x - X) / h)
    return np.sum(weights * Y) / np.sum(weights)

# Uso
X = np.linspace(0, 10, 100)
Y = np.sin(X) + np.random.normal(0, 0.1, 100)
x_new = np.linspace(0, 10, 200)
y_pred = [nadaraya_watson(x, X, Y, h=0.5) for x in x_new]
```

### Conclusão

Os suavizadores de kernel são uma ferramenta poderosa e flexível para análise não-paramétrica de dados. Sua capacidade de adaptar-se localmente à estrutura dos dados os torna particularmente úteis em cenários onde modelos paramétricos mais rígidos podem falhar. No entanto, a escolha cuidadosa do bandwidth e a consideração das propriedades de fronteira são cruciais para seu uso efetivo. À medida que a dimensionalidade aumenta, técnicas mais sofisticadas como regressão local linear e kernels adaptativos tornam-se cada vez mais importantes para manter um bom desempenho.

### Questões Avançadas

1. Discuta as implicações teóricas e práticas da "maldição da dimensionalidade" para estimadores de kernel em espaços de alta dimensão. Como técnicas como projeção aleatória ou seleção de características podem mitigar esse problema?

2. Compare teoricamente o estimador de Nadaraya-Watson com métodos de suavização alternativos como splines de suavização e regressão polinomial local. Em quais situações cada método seria preferível?

3. Derive a expressão para o MISE (Mean Integrated Squared Error) assintótico do estimador de Nadaraya-Watson e use-a para justificar a escolha do kernel de Epanechnikov como o kernel "ótimo" em termos de eficiência assintótica.

### Referências

[1] "Kernel smoothing methods achieve flexibility in estimating the regression function f (X) over the domain IR p by fitting a different but simple model separately at each query point x 0 ." (Trecho de ESL II)

[2] "This localization is achieved via a weighting function or kernel K λ (x 0 , x i ), which assigns a weight to x i based on its distance from x 0 ." (Trecho de ESL II)

[3] "The kernels K λ are typically indexed by a parameter λ that dictates the width of the neighborhood." (Trecho de ESL II)

[4] "The Nadaraya–Watson kernel-weighted average ˆ f (x 0 ) = ∑ N i=1 K λ (x 0 , x i )y i ∑ N i=1 K λ (x 0 , x i ) , (6.2)" (Trecho de ESL II)

[5] "with the Epanechnikov quadratic kernel K λ (x 0 , x) = D( |x − x 0 | λ ), (6.3) with D(t) = { 3 4 (1 − t 2 ) if |t| ≤ 1; 0 otherwise." (Trecho de ESL II)

[6] "There is a natural bias–variance tradeoff as we change the width of the averaging window, which is most explicit for local averages" (Trecho de ESL II)

[7] "If the window is narrow, ˆ f (x 0 ) is an average of a small number of y i close to x 0 , and its variance will be relatively large—close to that of an individual y i . The bias will tend to be small, again because each of the E(y i ) = f (x i ) should be close to f (x 0 )." (Trecho de ESL II)

[8] "By fitting straight lines rather than constants locally, we can remove this bias exactly to first order; see Figure 6.3 (right panel)." (Trecho de ESL II)