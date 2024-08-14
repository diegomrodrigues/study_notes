## Nadaraya-Watson Kernel Smoother: Fundamentos e Aplicações

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240806143330058.png" alt="image-20240806143330058" style="zoom: 67%;" />

O suavizador de kernel Nadaraya-Watson é uma técnica fundamental em estatística não paramétrica e aprendizado de máquina, utilizada para estimar funções de regressão de forma flexível e adaptativa. Este método combina a simplicidade conceitual com a eficácia prática, tornando-o uma ferramenta valiosa para análise de dados e modelagem preditiva.

### Conceitos Fundamentais

| Conceito             | Explicação                                                   |
| -------------------- | ------------------------------------------------------------ |
| **Kernel**           | Uma função não negativa que determina os pesos para os pontos de dados próximos ao ponto de interesse. [1] |
| **Suavização Local** | Processo de estimar o valor da função em um ponto usando informações de pontos vizinhos. [1] |
| **Largura de Banda** | Parâmetro que controla o grau de suavização, determinando a extensão da influência local. [1] |

> ✔️ **Ponto de Destaque**: O suavizador Nadaraya-Watson é um estimador não paramétrico que não assume uma forma funcional específica para a relação entre variáveis, permitindo grande flexibilidade na modelagem de dados.

### Formulação Matemática

O estimador Nadaraya-Watson para a função de regressão $f(x)$ em um ponto $x_0$ é dado por [2]:

$$
\hat{f}(x_0) = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i)y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)}
$$

Onde:
- $K_\lambda(x_0, x_i)$ é a função kernel que atribui pesos aos pontos $x_i$ baseados em sua distância de $x_0$
- $y_i$ são os valores da variável resposta
- $\lambda$ é o parâmetro de largura de banda

Esta fórmula pode ser interpretada como uma média ponderada dos $y_i$, onde os pesos são determinados pela função kernel.

> ⚠️ **Nota Importante**: A escolha da função kernel e da largura de banda $\lambda$ é crucial para o desempenho do estimador Nadaraya-Watson.

### Funções Kernel Comuns

1. **Kernel Epanechnikov**:
   $$
   K_\lambda(x_0, x) = D\left(\frac{|x - x_0|}{\lambda}\right)
   $$
   onde $D(t) = \frac{3}{4}(1-t^2)$ se $|t| \leq 1$, e 0 caso contrário. [3]

2. **Kernel Gaussiano**:
   $$
   K_\lambda(x_0, x) = \frac{1}{\sqrt{2\pi}\lambda}\exp\left(-\frac{(x-x_0)^2}{2\lambda^2}\right)
   $$

3. **Kernel Tri-cúbico**:
   $$
   D(t) = (1-|t|^3)^3 \text{ se } |t| \leq 1, \text{ e 0 caso contrário}
   $$

#### Questões Técnicas/Teóricas

1. Como a escolha da função kernel afeta as propriedades estatísticas do estimador Nadaraya-Watson?
2. Derive a expressão para o viés do estimador Nadaraya-Watson assumindo que $f(x)$ é duas vezes diferenciável.

### Propriedades Estatísticas

O estimador Nadaraya-Watson possui várias propriedades estatísticas importantes:

1. **Consistência**: Sob condições adequadas, o estimador converge em probabilidade para a verdadeira função de regressão à medida que o tamanho da amostra aumenta e a largura de banda diminui apropriadamente.

2. **Viés**: O estimador tende a ter um viés nas fronteiras do domínio dos dados e em regiões de alta curvatura da função verdadeira. [4]

3. **Variância**: A variância do estimador é inversamente proporcional à densidade local dos pontos de dados e à largura de banda.

> ❗ **Ponto de Atenção**: O trade-off entre viés e variância é fundamental na escolha da largura de banda ótima.

### Seleção da Largura de Banda

A seleção da largura de banda $\lambda$ é crucial para o desempenho do estimador. Métodos comuns incluem:

1. **Validação Cruzada**: Minimiza o erro de previsão estimado.
2. **Plug-in**: Estima a largura de banda ótima baseada em estimativas do viés e variância.
3. **Rule of Thumb**: Usa uma fórmula simples baseada na variabilidade dos dados.

A largura de banda ótima teoricamente minimiza o Erro Quadrático Médio Integrado (MISE):

$$
MISE(\lambda) = E\left[\int (\hat{f}(x) - f(x))^2 dx\right]
$$

#### Questões Técnicas/Teóricas

1. Como você implementaria um procedimento de validação cruzada para selecionar a largura de banda ótima?
2. Discuta as vantagens e desvantagens de usar uma largura de banda fixa versus uma largura de banda adaptativa.

### Implementação em Python

Aqui está um exemplo simplificado de implementação do suavizador Nadaraya-Watson em Python:

```python
import numpy as np
from scipy.stats import norm

def nadaraya_watson(x, X, Y, h):
    kernel = lambda u: norm.pdf(u, loc=0, scale=h)
    weights = kernel(x - X)
    return np.sum(weights * Y) / np.sum(weights)

# Uso
X = np.random.rand(100)
Y = np.sin(2*np.pi*X) + 0.1*np.random.randn(100)
x_grid = np.linspace(0, 1, 1000)
y_smooth = [nadaraya_watson(x, X, Y, h=0.1) for x in x_grid]
```

Este código implementa o estimador Nadaraya-Watson com um kernel gaussiano e largura de banda fixa.

### Extensões e Variantes

1. **Regressão Local Linear**: Ajusta uma linha localmente em vez de uma constante, reduzindo o viés nas fronteiras. [5]

2. **Kernel Adaptativo**: Usa uma largura de banda que varia com a densidade local dos dados.

3. **Kernel Multivariado**: Estende o método para múltiplas dimensões usando kernels multivariados.

### Aplicações e Limitações

**Aplicações**:
- Suavização de séries temporais
- Estimação de densidade não paramétrica
- Análise exploratória de dados

**Limitações**:
- Sensibilidade à escolha da largura de banda
- Desempenho pobre em altas dimensões (maldição da dimensionalidade)
- Computacionalmente intensivo para grandes conjuntos de dados

### Conclusão

O suavizador de kernel Nadaraya-Watson é uma técnica poderosa e flexível para estimação não paramétrica de funções de regressão. Sua simplicidade conceitual, combinada com sua eficácia prática, o torna uma ferramenta valiosa no arsenal de qualquer cientista de dados ou estatístico. No entanto, seu uso eficaz requer uma compreensão profunda de suas propriedades estatísticas e cuidado na seleção de parâmetros, particularmente a largura de banda.

### Questões Avançadas

1. Como você modificaria o estimador Nadaraya-Watson para lidar com dados heteroscedásticos?

2. Derive a taxa de convergência assintótica do estimador Nadaraya-Watson sob condições de suavidade adequadas para $f(x)$.

3. Compare teoricamente e empiricamente o desempenho do suavizador Nadaraya-Watson com o de splines de suavização em termos de viés, variância e complexidade computacional.

### Referências

[1] "Kernel smoothing methods achieve flexibility in estimating the regression function f(X) over the domain IR p by fitting a different but simple model separately at each query point x 0 ." (Trecho de ESL II)

[2] "The Nadaraya–Watson kernel-weighted average ˆ f (x 0 ) = ∑ N i=1 K λ (x 0 , x i )y i ∑ N i=1 K λ (x 0 , x i )" (Trecho de ESL II)

[3] "with the Epanechnikov quadratic kernel K λ (x 0 , x) = D( |x − x 0 | λ ), with D(t) = { 3 4 (1 − t 2 ) if |t| ≤ 1; 0 otherwise." (Trecho de ESL II)

[4] "Locally-weighted averages can be badly biased on the boundaries of the domain, because of the asymmetry of the kernel in that region." (Trecho de ESL II)

[5] "By fitting straight lines rather than constants locally, we can remove this bias exactly to first order" (Trecho de ESL II)