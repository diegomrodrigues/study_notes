## Função de Erro em Redes Neurais

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816094925578.png" alt="image-20240816094925578" style="zoom:67%;" />

````
MSE Final (Treino): 0.0912
MSE Final (Validação): 0.1099
MSE Final (Teste): 0.0962
MAE Final (Teste): 0.2712
````

A função de erro desempenha um papel crucial no ajuste de redes neurais, fornecendo uma medida quantitativa da discrepância entre as previsões do modelo e os valores reais. Este resumo explora em profundidade as funções de erro comumente utilizadas em redes neurais, focando na soma dos erros quadrados para problemas de regressão e na entropia cruzada para tarefas de classificação.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Função de Erro**           | Métrica que quantifica a diferença entre as previsões do modelo e os valores reais, guiando o processo de otimização. [1] |
| **Soma dos Erros Quadrados** | Função de erro padrão para problemas de regressão, calculando a soma das diferenças quadráticas entre previsões e valores reais. [2] |
| **Entropia Cruzada**         | Função de erro preferida para problemas de classificação, especialmente eficaz com ativações sigmóides ou softmax. [3] |

> ⚠️ **Nota Importante**: A escolha da função de erro deve ser alinhada com a natureza do problema (regressão ou classificação) e a função de ativação da camada de saída para otimizar o desempenho do modelo.

### Soma dos Erros Quadrados (SSE)

![image-20240816095518791](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816095518791.png)

A Soma dos Erros Quadrados (SSE) é amplamente utilizada em problemas de regressão. Para uma rede neural com K unidades de saída e N observações de treinamento, a SSE é definida como [2]:
$$
R(\theta) = \sum_{k=1}^K \sum_{i=1}^N (y_{ik} - f_k(x_i))^2
$$

Onde:
- $y_{ik}$ é o valor real da k-ésima saída para a i-ésima observação
- $f_k(x_i)$ é a previsão do modelo para a k-ésima saída dado o input $x_i$
- $\theta$ representa os parâmetros do modelo (pesos e vieses)

#### Propriedades Matemáticas da SSE:

1. **Convexidade**: A SSE é uma função convexa em relação às saídas do modelo, garantindo um mínimo global único quando os parâmetros são lineares. [4]

2. **Diferenciabilidade**: É continuamente diferenciável, facilitando a otimização por métodos de gradiente. [5]

3. **Sensibilidade a Outliers**: O termo quadrático amplifica o efeito de erros grandes, tornando a SSE sensível a outliers. [6]

A derivada parcial da SSE em relação a um parâmetro $\theta_j$ é dada por:

$$
\frac{\partial R}{\partial \theta_j} = 2 \sum_{k=1}^K \sum_{i=1}^N (y_{ik} - f_k(x_i)) \frac{\partial f_k(x_i)}{\partial \theta_j}
$$

Esta derivada é fundamental para algoritmos de otimização baseados em gradiente, como o backpropagation.

#### [Questões Técnicas/Teóricas]

1. Como a sensibilidade da SSE a outliers pode afetar o treinamento de uma rede neural em um conjunto de dados com ruído?
2. Compare matematicamente a eficiência computacional da SSE com uma função de erro baseada no erro absoluto médio.

### Entropia Cruzada

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816095643387.png" alt="image-20240816095643387" style="zoom:80%;" />

A entropia cruzada é a função de erro preferida para problemas de classificação, especialmente quando combinada com ativações sigmóides ou softmax na camada de saída. Para um problema de classificação com K classes, a entropia cruzada é definida como [3]:
$$
R(\theta) = -\sum_{i=1}^N \sum_{k=1}^K y_{ik} \log f_k(x_i)
$$

Onde:
- $y_{ik}$ é 1 se a i-ésima observação pertence à classe k, e 0 caso contrário
- $f_k(x_i)$ é a probabilidade prevista pelo modelo de que a i-ésima observação pertença à classe k

#### Propriedades Matemáticas da Entropia Cruzada:

1. **Não-negatividade**: A entropia cruzada é sempre não-negativa, atingindo zero apenas quando as previsões são perfeitas. [7]

2. **Assimetria**: Não é simétrica em relação a $y_{ik}$ e $f_k(x_i)$, o que a torna particularmente adequada para problemas de classificação. [8]

3. **Gradientes Fortes**: Produz gradientes mais fortes para erros pequenos em comparação com a SSE, acelerando o aprendizado em regiões de saturação das funções sigmóides. [9]

A derivada parcial da entropia cruzada em relação à saída da rede $z_k = \log f_k(x_i)$ (antes da ativação softmax) é dada por:

$$
\frac{\partial R}{\partial z_k} = f_k(x_i) - y_{ik}
$$

Esta forma simples da derivada contribui para a eficiência computacional do treinamento.

> ❗ **Ponto de Atenção**: A entropia cruzada combinada com a ativação softmax na camada de saída resulta em um gradiente particularmente simples e eficaz para o treinamento de redes neurais em problemas de classificação multiclasse.

#### [Questões Técnicas/Teóricas]

1. Explique matematicamente por que a entropia cruzada é preferível à SSE para problemas de classificação com ativações sigmóides.
2. Como a escolha entre SSE e entropia cruzada afeta a convergência do treinamento em uma rede neural profunda?

### Implementação Prática

A implementação da função de erro é um componente crítico no treinamento de redes neurais. Aqui está um exemplo conciso de como implementar tanto a SSE quanto a entropia cruzada em Python usando NumPy:

```python
import numpy as np

def sse(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

def cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Previne log(0)
    return -np.sum(y_true * np.log(y_pred))

# Exemplo de uso
y_true = np.array([[1, 0], [0, 1]])
y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])

print(f"SSE: {sse(y_true, y_pred)}")
print(f"Cross-entropy: {cross_entropy(y_true, y_pred)}")
```

> ✔️ **Ponto de Destaque**: A implementação da entropia cruzada inclui um termo epsilon para evitar problemas numéricos com log(0), uma consideração importante na prática.

### Conclusão

A escolha da função de erro é fundamental para o desempenho e a estabilidade do treinamento de redes neurais. A Soma dos Erros Quadrados (SSE) é amplamente utilizada em problemas de regressão devido à sua convexidade e diferenciabilidade [2]. Por outro lado, a entropia cruzada emergiu como a escolha preferida para problemas de classificação, especialmente quando combinada com ativações sigmóides ou softmax, devido à sua capacidade de produzir gradientes mais fortes e acelerar o aprendizado [3][9].

A compreensão profunda das propriedades matemáticas e das implicações práticas dessas funções de erro é essencial para os cientistas de dados e engenheiros de machine learning. Essa compreensão permite a seleção informada da função de erro mais apropriada para cada problema específico, otimizando assim o processo de treinamento e o desempenho final do modelo.

### Questões Avançadas

1. Desenvolva uma análise comparativa das propriedades de convergência da SSE e da entropia cruzada em um cenário de classificação binária com dados desbalanceados. Como cada função de erro afeta o equilíbrio entre precisão e recall?

2. Considere uma rede neural com função de ativação ReLU nas camadas ocultas e softmax na camada de saída. Derive matematicamente o gradiente da função de erro (usando entropia cruzada) em relação aos pesos da penúltima camada. Como esse gradiente se compara ao de uma rede com ativações sigmóides?

3. Proponha e justifique matematicamente uma função de erro híbrida que combine aspectos da SSE e da entropia cruzada para um problema de regressão com saídas restritas ao intervalo [0, 1]. Quais seriam as vantagens potenciais dessa abordagem?

### Referências

[1] "A função de erro fornece uma medida quantitativa da discrepância entre as previsões do modelo e os valores reais" (Trecho de ESL II)

[2] "Para regressão, usamos a soma dos erros quadrados como nossa medida de ajuste (função de erro)" (Trecho de ESL II)

[3] "Para classificação usamos entropia cruzada (deviance)" (Trecho de ESL II)

[4] "A SSE é uma função convexa em relação às saídas do modelo" (Trecho de ESL II)

[5] "É continuamente diferenciável, facilitando a otimização por métodos de gradiente" (Trecho de ESL II)

[6] "O termo quadrático amplifica o efeito de erros grandes, tornando a SSE sensível a outliers" (Trecho de ESL II)

[7] "A entropia cruzada é sempre não-negativa, atingindo zero apenas quando as previsões são perfeitas" (Trecho de ESL II)

[8] "Não é simétrica em relação a y_ik e f_k(x_i), o que a torna particularmente adequada para problemas de classificação" (Trecho de ESL II)

[9] "Produz gradientes mais fortes para erros pequenos em comparação com a SSE, acelerando o aprendizado em regiões de saturação das funções sigmóides" (Trecho de ESL II)