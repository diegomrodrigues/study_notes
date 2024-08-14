## Regra de Classificação em Análise Discriminante Linear

![image-20240802112459738](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240802112459738.png)

A regra de classificação é um componente fundamental na Análise Discriminante Linear (LDA), fornecendo um mecanismo para atribuir novas observações a classes predefinidas com base nos valores ajustados pelo modelo [1]. Este resumo aprofunda-se nos aspectos teóricos e práticos desta regra, explorando sua formulação matemática, implementação e implicações para a classificação em problemas de múltiplas classes.

### Conceitos Fundamentais

| Conceito                        | Explicação                                                   |
| ------------------------------- | ------------------------------------------------------------ |
| **Valores Ajustados**           | São as estimativas das probabilidades posteriores de pertencimento a cada classe, calculadas pelo modelo LDA para uma dada observação. [1] |
| **Regra de Classificação**      | Critério que atribui uma nova observação à classe com o maior valor ajustado, maximizando a probabilidade posterior estimada. [1] |
| **Função Discriminante Linear** | Função linear dos preditores que é utilizada para calcular os valores ajustados e, consequentemente, determinar a classificação. [2] |

> ⚠️ **Nota Importante**: A regra de classificação em LDA assume que as classes são mutuamente exclusivas e coletivamente exaustivas, ou seja, cada observação pertence a exatamente uma classe.

### Formulação Matemática da Regra de Classificação

A regra de classificação em LDA pode ser expressa matematicamente da seguinte forma [3]:

$$
\hat{G}(x) = \arg\max_{k} \hat{f}_k(x)
$$

Onde:
- $\hat{G}(x)$ é a classe prevista para a observação $x$
- $\hat{f}_k(x)$ é o valor ajustado (estimativa da probabilidade posterior) para a classe $k$
- $\arg\max_{k}$ denota o argumento $k$ que maximiza a expressão subsequente

Esta formulação encapsula o princípio fundamental da regra de classificação: atribuir a observação à classe com a maior probabilidade posterior estimada [4].

#### Cálculo dos Valores Ajustados

Os valores ajustados $\hat{f}_k(x)$ são calculados usando as funções discriminantes lineares, que têm a forma geral [5]:

$$
\delta_k(x) = x^T\Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k + \log \pi_k
$$

Onde:
- $\Sigma$ é a matriz de covariância comum estimada
- $\mu_k$ é o vetor de médias estimado para a classe $k$
- $\pi_k$ é a probabilidade a priori estimada para a classe $k$

Os valores ajustados são então obtidos através da transformação softmax [6]:

$$
\hat{f}_k(x) = \frac{e^{\delta_k(x)}}{\sum_{j=1}^K e^{\delta_j(x)}}
$$

Esta transformação garante que os valores ajustados somem 1 e possam ser interpretados como probabilidades.

> ✔️ **Ponto de Destaque**: A transformação softmax preserva a ordem relativa dos valores discriminantes, garantindo que a classe com o maior $\delta_k(x)$ também tenha o maior $\hat{f}_k(x)$.

Vamos criar um exemplo numérico passo a passo para ilustrar o cálculo dos valores ajustados em um problema de classificação com Análise Discriminante Linear (LDA) com três classes. Seguiremos cada etapa detalhadamente.

Exemplo: Classificação de três tipos de flores baseada em duas características: comprimento e largura da pétala.

Passo 1: Dados iniciais

Suponha que temos os seguintes dados estimados:

1) Matriz de covariância comum estimada (Σ):
   $$ \Sigma = \begin{bmatrix} 0.3 & 0.1 \\ 0.1 & 0.2 \end{bmatrix} $$

2) Vetores de médias estimados para cada classe (μₖ):
   $$ \mu_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad \mu_2 = \begin{bmatrix} 3 \\ 2 \end{bmatrix}, \quad \mu_3 = \begin{bmatrix} 4 \\ 3 \end{bmatrix} $$

3) Probabilidades a priori estimadas (πₖ):
   $$ \pi_1 = 0.3, \quad \pi_2 = 0.5, \quad \pi_3 = 0.2 $$

4) Nova observação a ser classificada:
   $$ x = \begin{bmatrix} 3.5 \\ 2.5 \end{bmatrix} $$

Passo 2: Calcular Σ⁻¹

$$ \Sigma^{-1} = \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} $$

Passo 3: Calcular δₖ(x) para cada classe

Para a classe 1:
$$ \begin{aligned}
\delta_1(x) &= x^T\Sigma^{-1}\mu_1 - \frac{1}{2}\mu_1^T\Sigma^{-1}\mu_1 + \log \pi_1 \\
&= [3.5 \quad 2.5] \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} - \frac{1}{2}[2 \quad 1] \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} + \log 0.3 \\
&= 15.3846 - 5.7692 - 1.2040 \\
&= 8.4114
\end{aligned} $$

Para a classe 2:
$$ \begin{aligned}
\delta_2(x) &= x^T\Sigma^{-1}\mu_2 - \frac{1}{2}\mu_2^T\Sigma^{-1}\mu_2 + \log \pi_2 \\
&= [3.5 \quad 2.5] \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix} - \frac{1}{2}[3 \quad 2] \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} \begin{bmatrix} 3 \\ 2 \end{bmatrix} + \log 0.5 \\
&= 26.9231 - 13.4615 - 0.6931 \\
&= 12.7685
\end{aligned} $$

Para a classe 3:
$$ \begin{aligned}
\delta_3(x) &= x^T\Sigma^{-1}\mu_3 - \frac{1}{2}\mu_3^T\Sigma^{-1}\mu_3 + \log \pi_3 \\
&= [3.5 \quad 2.5] \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} \begin{bmatrix} 4 \\ 3 \end{bmatrix} - \frac{1}{2}[4 \quad 3] \begin{bmatrix} 3.8462 & -1.9231 \\ -1.9231 & 5.7692 \end{bmatrix} \begin{bmatrix} 4 \\ 3 \end{bmatrix} + \log 0.2 \\
&= 38.4615 - 24.2308 - 1.6094 \\
&= 12.6213
\end{aligned} $$

Passo 4: Aplicar a transformação softmax para obter $\hat{f}_k(x)$

$$ \begin{aligned}
\hat{f}_1(x) &= \frac{e^{8.4114}}{e^{8.4114} + e^{12.7685} + e^{12.6213}} = 0.0041 \\
\hat{f}_2(x) &= \frac{e^{12.7685}}{e^{8.4114} + e^{12.7685} + e^{12.6213}} = 0.5306 \\
\hat{f}_3(x) &= \frac{e^{12.6213}}{e^{8.4114} + e^{12.7685} + e^{12.6213}} = 0.4653
\end{aligned} $$

Passo 5: Interpretar os resultados

Os valores ajustados $\hat{f}_k(x)$ representam as probabilidades posteriores estimadas de que a nova observação x pertença a cada uma das três classes:

- Classe 1: 0.41% de probabilidade
- Classe 2: 53.06% de probabilidade
- Classe 3: 46.53% de probabilidade

Conclusão: Seguindo a regra de classificação LDA, classificaríamos esta nova observação na Classe 2, pois ela tem o maior valor ajustado (maior probabilidade posterior estimada).

Este exemplo numérico demonstra como os cálculos são realizados na prática, desde as funções discriminantes lineares até a obtenção das probabilidades posteriores através da transformação softmax. Ele ilustra como o LDA utiliza as informações das médias das classes, da matriz de covariância comum e das probabilidades a priori para fazer previsões sobre novas observações.

### Implementação da Regra de Classificação

A implementação prática da regra de classificação em um ambiente de programação como Python pode ser realizada de forma eficiente usando operações vetorizadas. Aqui está um exemplo conciso de como isso poderia ser feito:

```python
import numpy as np

def lda_classify(X, means, cov_inv, priors):
    K = means.shape[0]  # Número de classes
    N = X.shape[0]  # Número de observações
    
    # Calcula os valores discriminantes para todas as classes
    discriminants = np.dot(X, np.dot(cov_inv, means.T)) - \
                    0.5 * np.sum(np.dot(means, cov_inv) * means, axis=1) + \
                    np.log(priors)
    
    # Aplica softmax para obter probabilidades
    exp_disc = np.exp(discriminants - np.max(discriminants, axis=1, keepdims=True))
    probs = exp_disc / np.sum(exp_disc, axis=1, keepdims=True)
    
    # Classifica cada observação na classe com maior probabilidade
    predictions = np.argmax(probs, axis=1)
    
    return predictions, probs
```

Esta implementação assume que `X` é uma matriz de observações, `means` é uma matriz de médias das classes, `cov_inv` é a inversa da matriz de covariância comum, e `priors` é um vetor de probabilidades a priori das classes.

#### Questões Técnicas/Teóricas

1. Como a regra de classificação LDA se comportaria se duas classes tivessem exatamente o mesmo valor ajustado máximo para uma determinada observação? Como você modificaria a implementação para lidar com esse caso?

2. Explique como a escolha das probabilidades a priori $\pi_k$ afeta a regra de classificação e em quais cenários práticos você consideraria modificá-las a partir de suas estimativas empíricas.

### Propriedades e Implicações da Regra de Classificação

A regra de classificação LDA possui várias propriedades importantes que influenciam seu desempenho e aplicabilidade [7]:

1. **Linearidade das Fronteiras de Decisão**: As fronteiras entre as regiões de classificação são hiperplanos no espaço de características, o que pode ser uma limitação em dados com relações não-lineares complexas entre as classes.

2. **Sensibilidade à Escala**: A LDA é invariante à escala das variáveis preditoras, desde que a transformação de escala seja aplicada consistentemente a todas as classes.

3. **Robustez a Outliers**: Comparada a métodos como QDA (Análise Discriminante Quadrática), a LDA tende a ser mais robusta a outliers devido à suposição de covariância comum entre as classes.

4. **Eficiência Computacional**: A regra de classificação LDA é computacionalmente eficiente, especialmente para problemas com muitas classes, pois requer apenas o cálculo de funções lineares.

> ❗ **Ponto de Atenção**: Embora eficiente, a regra de classificação LDA assume normalidade multivariada e homoscedasticidade (igualdade de matrizes de covariância entre classes). Violações dessas suposições podem impactar o desempenho do classificador.

### Comparação com Outras Técnicas de Classificação

| 👍 Vantagens da Regra LDA                               | 👎 Desvantagens da Regra LDA                                  |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| Simplicidade e interpretabilidade [8]                  | Suposições restritivas sobre a distribuição dos dados [9]    |
| Eficiência computacional em problemas multiclasse [10] | Incapacidade de capturar relações não-lineares complexas [11] |
| Bom desempenho quando as suposições são atendidas [12] | Sensibilidade a classes altamente desequilibradas [13]       |

### Extensões e Variações

1. **Regularização**: Incorporação de termos de regularização na estimação da matriz de covariância para melhorar a estabilidade e o desempenho em dimensões elevadas [14].

2. **LDA Esparsa**: Modificações da regra de classificação para promover esparsidade nos coeficientes discriminantes, facilitando a interpretação e potencialmente melhorando a generalização [15].

3. **LDA Kernel**: Extensão não-linear da LDA usando técnicas de kernel para lidar com fronteiras de decisão não-lineares [16].

A formulação matemática para LDA Kernel pode ser expressa como:

$$
\delta_k(x) = \sum_{i=1}^N \alpha_{ki} K(x, x_i) + \beta_k
$$

Onde $K(x, x_i)$ é a função kernel, $\alpha_{ki}$ são os coeficientes aprendidos, e $\beta_k$ é o termo de viés para a classe $k$.

#### Questões Técnicas/Teóricas

1. Como você modificaria a regra de classificação LDA para incorporar custos diferentes para diferentes tipos de erros de classificação? Forneça uma formulação matemática para esta modificação.

2. Descreva um cenário em que a LDA Kernel seria preferível à LDA padrão e explique como você escolheria e otimizaria a função kernel apropriada.

### Conclusão

A regra de classificação em Análise Discriminante Linear é um componente crucial que traduz os resultados do modelo em decisões de classificação práticas. Sua simplicidade, eficiência computacional e base teórica sólida a tornam uma escolha popular em muitas aplicações de aprendizado de máquina e estatística [17]. No entanto, é fundamental que os praticantes estejam cientes das suposições subjacentes e das limitações potenciais ao aplicar esta técnica em problemas do mundo real. A compreensão profunda da regra de classificação LDA, incluindo suas propriedades matemáticas e implicações práticas, é essencial para sua aplicação eficaz e para o desenvolvimento de extensões e melhorias futuras.

### Questões Avançadas

1. Considere um problema de classificação com três classes em um espaço bidimensional. Dado que a LDA produziu as seguintes funções discriminantes:

   $\delta_1(x) = 2x_1 + 3x_2 - 1$
   $\delta_2(x) = -x_1 + 2x_2 + 2$
   $\delta_3(x) = x_1 - x_2 + 1$

   Descreva geometricamente as regiões de decisão resultantes e derive as equações das fronteiras de decisão entre as classes.

2. Em um cenário de alta dimensionalidade (p >> n), como você modificaria a regra de classificação LDA para lidar com o problema de singularidade da matriz de covariância? Discuta as implicações teóricas e práticas de sua abordagem.

3. Proponha e justifique uma métrica de avaliação apropriada para um classificador LDA em um problema de detecção de fraude bancária, onde as classes são altamente desequilibradas (99.9% transações legítimas, 0.1% fraudulentas) e o custo de falsos negativos é significativamente maior que o de falsos positivos.
