## Votação por Maioria Ponderada em Boosting

![image-20240813083445310](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240813083445310.png)

O conceito de votação por maioria ponderada é fundamental no contexto de boosting, especialmente no algoritmo AdaBoost. Este método combina as previsões de múltiplos classificadores fracos para produzir uma previsão final mais robusta e precisa [1].

### Conceitos Fundamentais

| Conceito                | Explicação                                                   |
| ----------------------- | ------------------------------------------------------------ |
| **Classificador Fraco** | Um algoritmo de aprendizado que produz previsões apenas ligeiramente melhores que escolhas aleatórias. No contexto de boosting, geralmente são árvores de decisão simples [1]. |
| **Votação Ponderada**   | Processo de combinar previsões de múltiplos classificadores, atribuindo pesos diferentes a cada um com base em seu desempenho [1]. |
| **Função de Agregação** | A função matemática que combina as previsões ponderadas dos classificadores fracos para produzir a previsão final [1]. |

> ✔️ **Ponto de Destaque**: A votação por maioria ponderada permite que o modelo final aproveite as forças de cada classificador fraco, mitigando suas fraquezas individuais.

### Formulação Matemática da Votação por Maioria Ponderada

A essência da votação por maioria ponderada no AdaBoost é capturada pela seguinte equação [1]:

$$
G(x) = \text{sign}\left(\sum_{m=1}^M \alpha_m G_m(x)\right)
$$

Onde:
- $G(x)$ é a previsão final do ensemble
- $G_m(x)$ são as previsões dos classificadores fracos individuais
- $\alpha_m$ são os pesos atribuídos a cada classificador
- $M$ é o número total de classificadores fracos
- $\text{sign}()$ é a função sinal, que retorna +1 para valores positivos e -1 para negativos

Esta formulação demonstra como as previsões individuais $G_m(x)$ são combinadas linearmente, ponderadas pelos coeficientes $\alpha_m$, para produzir uma soma ponderada. A função sinal então converte esta soma em uma classificação binária final [1].

> ❗ **Ponto de Atenção**: Os pesos $\alpha_m$ são cruciais, pois determinam a influência relativa de cada classificador na decisão final.

### Determinação dos Pesos $\alpha_m$

No AdaBoost, os pesos $\alpha_m$ são calculados durante o processo de treinamento. A fórmula para $\alpha_m$ é [1]:

$$
\alpha_m = \log\left(\frac{1 - \text{err}_m}{\text{err}_m}\right)
$$

Onde $\text{err}_m$ é a taxa de erro ponderada do m-ésimo classificador fraco.

Esta fórmula atribui maiores pesos aos classificadores com menores taxas de erro, garantindo que classificadores mais precisos tenham maior influência na decisão final [1].

#### Questões Técnicas/Teóricas

1. Como a escolha da função sign() na equação de votação por maioria ponderada afeta a interpretabilidade do modelo final? Quais seriam as implicações de usar uma função suave em vez da função sign()?

2. Considerando a fórmula para $\alpha_m$, como o peso de um classificador se comporta quando sua taxa de erro se aproxima de 0.5? Justifique matematicamente e discuta as implicações para o ensemble.

### Vantagens e Desvantagens da Votação por Maioria Ponderada

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Melhora a precisão geral do modelo através da combinação de múltiplos classificadores [1] | Pode ser computacionalmente intensivo, especialmente com um grande número de classificadores |
| Reduz o overfitting ao combinar múltiplos modelos simples [1] | A interpretabilidade do modelo final pode ser reduzida devido à complexidade do ensemble |
| Permite que o modelo se adapte a diferentes aspectos dos dados através de diferentes classificadores | Requer uma cuidadosa calibração dos pesos para garantir um desempenho ótimo |

### Implementação Prática

Embora o AdaBoost.M1 seja um algoritmo complexo, podemos ilustrar o conceito de votação por maioria ponderada com um exemplo simplificado:

```python
import numpy as np

def weighted_majority_vote(classifiers, weights, X):
    predictions = np.array([clf.predict(X) for clf in classifiers])
    weighted_sum = np.dot(predictions.T, weights)
    return np.sign(weighted_sum)

# Assumindo que temos classificadores treinados e seus pesos
classifiers = [clf1, clf2, clf3]  # Lista de classificadores fracos
weights = [0.2, 0.5, 0.3]  # Pesos correspondentes
X_new = np.array([[1.5, 2.0]])  # Novo ponto de dados para classificação

final_prediction = weighted_majority_vote(classifiers, weights, X_new)
```

Este exemplo demonstra como as previsões de múltiplos classificadores são combinadas usando seus pesos correspondentes para produzir uma previsão final [1].

### Conclusão

A votação por maioria ponderada é um componente crucial do algoritmo AdaBoost e de muitas outras técnicas de ensemble learning. Ao combinar as previsões de múltiplos classificadores fracos de forma ponderada, este método permite a criação de um classificador forte que supera significativamente o desempenho de qualquer um dos classificadores individuais [1].

A eficácia deste método reside na sua capacidade de atribuir maior importância aos classificadores mais precisos, enquanto ainda leva em conta as contribuições de todos os membros do ensemble. Isso resulta em um modelo final que é mais robusto, menos propenso a overfitting e geralmente mais preciso do que modelos individuais [1].

### Questões Avançadas

1. Como o conceito de votação por maioria ponderada poderia ser estendido para problemas de classificação multiclasse? Proponha uma formulação matemática e discuta os desafios potenciais.

2. Considere um cenário onde alguns classificadores fracos têm alta precisão em certas regiões do espaço de características, mas baixa precisão em outras. Como você modificaria o esquema de votação por maioria ponderada para aproveitar essa informação local? Descreva uma abordagem e suas potenciais vantagens e desvantagens.

3. O AdaBoost utiliza uma função de perda exponencial para derivar os pesos dos classificadores. Como a escolha de diferentes funções de perda (por exemplo, perda logística) afetaria o esquema de votação por maioria ponderada? Analise as implicações teóricas e práticas.

### Referências

[1] "The predictions from all of them are then combined through a weighted majority vote to produce the final prediction:

G(x) = sign(∑^M_m=1 α_m G_m(x)).

Here α_1, α_2, . . . , α_M are computed by the boosting algorithm, and weight the contribution of each respective G_m(x). Their effect is to give higher influence to the more accurate classifiers in the sequence." (Trecho de ESL II)