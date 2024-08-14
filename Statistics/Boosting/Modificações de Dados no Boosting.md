## Modificações de Dados no Boosting

![image-20240813084743078](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240813084743078.png)

O boosting é uma técnica poderosa de aprendizado de máquina que visa melhorar o desempenho de modelos "fracos" combinando-os sequencialmente. Um aspecto crucial deste método é a modificação sistemática dos dados de treinamento entre as iterações, permitindo que o algoritmo se concentre progressivamente nas observações mais desafiadoras [1].

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Pesos das Observações** | Valores numéricos atribuídos a cada amostra de treinamento, influenciando sua importância durante o aprendizado [2] |
| **Classificador Fraco**   | Um modelo com desempenho ligeiramente melhor que a aleatoriedade, usado como base para o boosting [3] |
| **Erro Ponderado**        | Medida de erro que leva em conta os pesos das observações ao avaliar o desempenho do classificador [4] |

> ⚠️ **Nota Importante**: A modificação dos pesos das observações é o mecanismo central pelo qual o boosting adapta seu foco ao longo das iterações.

### Processo de Modificação de Dados

O processo de modificação de dados no boosting segue um padrão iterativo bem definido:

1. **Inicialização**: Todos os pesos são inicialmente definidos como $w_i = 1/N$, onde $N$ é o número total de observações [5].

2. **Iteração**: Para cada etapa $m = 2, 3, ..., M$:
   
   a) Os pesos das observações são individualmente modificados [6].
   
   b) O algoritmo de classificação é reaplicado aos dados ponderados [7].

3. **Atualização de Pesos**: Após cada iteração, os pesos são atualizados da seguinte forma:
   
   $$w_i \leftarrow w_i \cdot \exp[\alpha_m \cdot I(y_i \neq G_m(x_i))]$$
   
   Onde:
   - $w_i$ é o peso da observação $i$
   - $\alpha_m$ é o peso atribuído ao classificador $m$
   - $I(y_i \neq G_m(x_i))$ é uma função indicadora que retorna 1 se a classificação estiver incorreta e 0 caso contrário [8]

> ✔️ **Ponto de Destaque**: A atualização exponencial dos pesos garante que observações mal classificadas recebam maior ênfase nas iterações subsequentes.

### Implicações da Modificação de Dados

1. **Foco Adaptativo**: À medida que as iterações progridem, o algoritmo concentra-se cada vez mais nas observações que são difíceis de classificar corretamente [9].

2. **Regularização Implícita**: A modificação dos pesos atua como uma forma de regularização, ajudando a prevenir o overfitting [10].

3. **Robustez**: Ao dar mais importância às observações mal classificadas, o boosting torna-se mais robusto a ruídos e outliers nos dados [11].

#### Questões Técnicas

1. Como a modificação dos pesos das observações afeta a convergência do algoritmo de boosting?
2. Explique por que a atualização exponencial dos pesos é preferível a uma atualização linear no contexto do boosting.

### Análise Matemática da Modificação de Pesos

A atualização dos pesos pode ser vista como uma otimização do critério de perda exponencial:

$$L(y, f(x)) = \exp(-y f(x))$$

Onde $y \in \{-1, 1\}$ é a classe verdadeira e $f(x)$ é a previsão do modelo [12].

A minimização deste critério leva naturalmente à regra de atualização de pesos usada no AdaBoost:

$$w_i^{(m+1)} = w_i^{(m)} \cdot \exp(-y_i f_m(x_i))$$

Esta formulação garante que:

1. Observações corretamente classificadas tenham seus pesos reduzidos.
2. Observações incorretamente classificadas tenham seus pesos aumentados exponencialmente.

> ❗ **Ponto de Atenção**: A escolha da função de perda exponencial não é arbitrária. Ela possui propriedades matemáticas que facilitam a derivação do algoritmo AdaBoost e garantem sua convergência [13].

### Implementação Prática

Aqui está um exemplo simplificado de como a modificação de dados pode ser implementada em Python:

```python
import numpy as np

def update_weights(y_true, y_pred, weights, alpha):
    incorrect = (y_true != y_pred).astype(int)
    weights *= np.exp(alpha * incorrect)
    return weights / np.sum(weights)  # Normalização

# Exemplo de uso
y_true = np.array([1, -1, 1, -1, 1])
y_pred = np.array([1, 1, -1, -1, 1])
weights = np.ones(5) / 5
alpha = 0.5

new_weights = update_weights(y_true, y_pred, weights, alpha)
print(new_weights)
```

Este código demonstra como os pesos são atualizados com base nas previsões corretas e incorretas, seguindo o princípio do AdaBoost [14].

### Conclusão

A modificação sistemática dos dados através da atualização de pesos é um componente crucial do boosting. Este processo permite que o algoritmo adapte seu foco iterativamente, concentrando-se nas observações mais desafiadoras e, consequentemente, melhorando o desempenho geral do modelo. A fundamentação matemática por trás dessa abordagem, baseada na minimização da perda exponencial, fornece uma estrutura teórica sólida que explica o sucesso do boosting em uma ampla gama de aplicações de aprendizado de máquina [15].

### Questões Avançadas

1. Como a escolha da função de perda afeta o processo de modificação de dados no boosting? Compare o comportamento do AdaBoost (perda exponencial) com uma variante que use a perda logística.

2. Discuta as implicações teóricas e práticas de usar diferentes esquemas de inicialização de pesos no boosting. Como isso poderia afetar a convergência e o desempenho do modelo?

3. Proponha e justifique uma estratégia de modificação de dados para um cenário de boosting em aprendizado semi-supervisionado, onde apenas uma parte dos dados de treinamento é rotulada.

### Referências

[1] "The data modifications at each boosting step consist of applying weights w1, w2, . . . , wN to each of the training observations (xi, yi), i = 1, 2, . . . , N." (Trecho de ESL II)

[2] "Initially all of the weights are set to wi = 1/N, so that the first step simply trains the classifier on the data in the usual manner." (Trecho de ESL II)

[3] "A weak classifier is one whose error rate is only slightly better than random guessing." (Trecho de ESL II)

[4] "The resulting weighted error rate is computed at line 2b." (Trecho de ESL II)

[5] "Initially all of the weights are set to wi = 1/N" (Trecho de ESL II)

[6] "For each successive iteration m = 2, 3, . . . , M the observation weights are individually modified" (Trecho de ESL II)

[7] "and the classification algorithm is reapplied to the weighted observations." (Trecho de ESL II)

[8] "Set wi ← wi · exp[αm · I(yi 6= Gm(xi))], i = 1, 2, . . . , N." (Trecho de ESL II)

[9] "Thus as iterations proceed, observations that are difficult to classify correctly receive ever-increasing influence." (Trecho de ESL II)

[10] "Each successive classifier is thereby forced to concentrate on those training observations that are missed by previous ones in the sequence." (Trecho de ESL II)

[11] "Those observations that were misclassified by the classifier Gm−1(x) induced at the previous step have their weights increased, whereas the weights are decreased for those that were classified correctly." (Trecho de ESL II)

[12] "L(y, f(x)) = exp(−yf(x))" (Trecho de ESL II)

[13] "The principal attraction of exponential loss in the context of additive modeling is computational; it leads to the simple modular reweighting AdaBoost algorithm." (Trecho de ESL II)

[14] "Algorithm 10.1 AdaBoost.M1." (Trecho de ESL II)

[15] "Boosting is one of the most powerful learning ideas introduced in the last twenty years." (Trecho de ESL II)