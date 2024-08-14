## Bootstrap para Avaliação de Incerteza

![image-20240810111512243](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240810111512243.png)

O bootstrap é uma técnica estatística poderosa e versátil para avaliar a incerteza em estimativas de parâmetros e previsões. Introduzido por Bradley Efron em 1979, o bootstrap revolucionou a inferência estatística ao fornecer uma abordagem computacional direta para quantificar a variabilidade em estimativas estatísticas [1].

### Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Bootstrap**              | Método de reamostragem que envolve a criação de múltiplas amostras a partir dos dados originais para estimar a distribuição amostral de uma estatística. [1] |
| **Amostra Bootstrap**      | Uma amostra gerada pelo bootstrap, obtida através da amostragem com reposição dos dados originais. [2] |
| **Distribuição Bootstrap** | A distribuição empírica das estatísticas calculadas a partir das amostras bootstrap. [3] |

> ✔️ **Ponto de Destaque**: O bootstrap permite a avaliação da incerteza sem fazer suposições distributivas, tornando-o uma ferramenta robusta e amplamente aplicável.

### Metodologia Bootstrap

![image-20240810111824635](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240810111824635.png)

O processo bootstrap segue os seguintes passos [2]:

1. Dado um conjunto de dados original $Z = \{z_1, z_2, ..., z_N\}$, gere B amostras bootstrap $Z^{*1}, Z^{*2}, ..., Z^{*B}$, cada uma contendo N observações amostradas com reposição de Z.

2. Para cada amostra bootstrap $Z^{*b}$, calcule a estatística de interesse $\hat{\theta}^{*b}$.

3. Use a distribuição empírica de $\{\hat{\theta}^{*1}, \hat{\theta}^{*2}, ..., \hat{\theta}^{*B}\}$ para inferir sobre a variabilidade de $\hat{\theta}$.

A estimativa bootstrap do erro padrão é dada por [3]:

$$
\hat{se}_B = \sqrt{\frac{1}{B-1} \sum_{b=1}^B (\hat{\theta}^{*b} - \bar{\theta}^*)^2}
$$

onde $\bar{\theta}^* = \frac{1}{B} \sum_{b=1}^B \hat{\theta}^{*b}$.

#### Questões Técnicas/Teóricas

1. Como o número de amostras bootstrap (B) afeta a precisão da estimativa do erro padrão?
2. Explique a diferença entre o bootstrap paramétrico e não-paramétrico.

### Aplicação em Suavização por Splines

O bootstrap pode ser aplicado para avaliar a incerteza em modelos de suavização, como splines cúbicos [4]. Considere um modelo de spline cúbico:

$$
\mu(x) = \sum_{j=1}^7 \beta_j h_j(x)
$$

onde $h_j(x)$ são funções de base B-spline.

Para construir intervalos de confiança pontual, podemos:

1. Gerar B amostras bootstrap dos dados originais.
2. Ajustar o modelo de spline para cada amostra bootstrap.
3. Calcular os percentis 2.5% e 97.5% das curvas ajustadas em cada ponto x.

> ⚠️ **Nota Importante**: O bootstrap captura tanto a variabilidade devido ao ruído nos dados quanto a incerteza na escolha dos nós do spline.

### Bootstrap vs. Inferência Bayesiana

O bootstrap tem uma conexão interessante com a inferência Bayesiana [5]:

1. A distribuição bootstrap pode ser vista como uma aproximação da distribuição posterior não-paramétrica.
2. A média bootstrap é análoga à média posterior Bayesiana.

Formalmente, para uma estatística $S(\hat{w})$, onde $\hat{w}$ são as proporções observadas em categorias discretas, temos:

$$
\hat{w}^* \sim \text{Mult}(N, \hat{w})
$$

Esta distribuição é similar à distribuição posterior Dirichlet:

$$
w \sim \text{Di}_L(N\hat{w})
$$

> ❗ **Ponto de Atenção**: Embora similares, o bootstrap e a inferência Bayesiana têm diferenças fundamentais em sua interpretação e aplicação.

#### Questões Técnicas/Teóricas

1. Como o bootstrap difere da inferência Bayesiana na interpretação dos intervalos de confiança?
2. Em quais situações o bootstrap pode ser preferível à inferência Bayesiana clássica?

### Implementação em Python

Aqui está um exemplo simplificado de como implementar o bootstrap para estimar o erro padrão de uma média:

```python
import numpy as np

def bootstrap_mean(data, B=1000):
    n = len(data)
    means = np.zeros(B)
    for i in range(B):
        sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    return np.std(means)

# Exemplo de uso
data = np.random.normal(0, 1, 100)
se = bootstrap_mean(data)
print(f"Erro padrão bootstrap: {se}")
```

### Conclusão

O bootstrap é uma ferramenta poderosa para avaliar a incerteza em estimativas estatísticas, oferecendo uma alternativa flexível e robusta aos métodos paramétricos tradicionais [6]. Sua aplicabilidade em uma ampla gama de problemas, desde regressão até classificação, o torna uma técnica essencial no arsenal de qualquer cientista de dados ou estatístico.

### Questões Avançadas

1. Como o bootstrap pode ser adaptado para lidar com dados dependentes, como séries temporais?
2. Discuta as limitações do bootstrap em amostras pequenas e como elas podem ser mitigadas.
3. Compare o desempenho computacional e estatístico do bootstrap com métodos de Monte Carlo para Cadeias de Markov (MCMC) em um cenário de inferência bayesiana.

### Referências

[1] "O bootstrap método fornece uma maneira computacional direta de avaliar a incerteza, amostrando dos dados de treinamento." (Trecho de ESL II)

[2] "Aqui está como poderíamos aplicar o bootstrap neste exemplo. Desenhamos B conjuntos de dados cada um de tamanho N = 50 com substituição de nossos dados de treinamento, sendo a unidade de amostragem o par zi = (xi, yi)." (Trecho de ESL II)

[3] "Para cada conjunto de dados bootstrap Z∗ ajustamos uma spline cúbica μˆ∗(x); os ajustes de dez dessas amostras são mostrados no painel inferior esquerdo da Figura 8.2." (Trecho de ESL II)

[4] "Usando B = 200 amostras bootstrap, podemos formar uma banda de confiança pontual de 95% a partir dos percentis em cada x: encontramos os valores 2,5% × 200 = quinto maiores e menores em cada x." (Trecho de ESL II)

[5] "Neste sentido, a distribuição bootstrap representa uma distribuição posterior não-paramétrica e não-informativa (aproximada) para nosso parâmetro." (Trecho de ESL II)

[6] "Mas esta distribuição bootstrap é obtida sem dor — sem ter que especificar formalmente uma priori e sem ter que amostrar da distribuição posterior." (Trecho de ESL II)