## Boosting: Potencializando Classificadores Fracos

![image-20240812090603478](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240812090603478.png)

O boosting é uma das ideias mais poderosas introduzidas no aprendizado de máquina nas últimas duas décadas [1]. Originalmente concebido para problemas de classificação, o boosting pode ser estendido com grande eficácia para problemas de regressão. Este método revolucionário combina a saída de muitos classificadores "fracos" para produzir um poderoso "comitê" de decisão [1].

### Conceitos Fundamentais

| Conceito                | Explicação                                                   |
| ----------------------- | ------------------------------------------------------------ |
| **Classificador Fraco** | Um algoritmo cuja performance é apenas ligeiramente melhor que a adivinhação aleatória [2] |
| **Boosting**            | Processo de aplicação sequencial do algoritmo de classificação fraca a versões modificadas dos dados, produzindo uma sequência de classificadores fracos [2] |
| **Comitê**              | Combinação ponderada dos classificadores fracos para produzir a previsão final [3] |

> ⚠️ **Nota Importante**: O boosting difere fundamentalmente de outras abordagens baseadas em comitês, como o bagging, apesar de superficialmente parecerem similares [1].

### O Algoritmo AdaBoost.M1

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240812090822780.png" alt="image-20240812090822780" style="zoom: 67%;" />

O AdaBoost.M1, desenvolvido por Freund e Schapire (1997), é o algoritmo de boosting mais popular [2]. Considere um problema de classificação binária com a variável de saída Y ∈ {-1, 1}.

1. **Inicialização**: Atribua pesos iguais a todas as observações de treinamento.

2. **Iteração**: Para m = 1 até M:
   a) Ajuste um classificador $G_m(x)$ aos dados de treinamento usando os pesos atuais.
   b) Calcule o erro ponderado:
      $$ err_m = \frac{\sum_{i=1}^N w_i I(y_i \neq G_m(x_i))}{\sum_{i=1}^N w_i} $$
   c) Compute $\alpha_m = \log((1 - err_m)/err_m)$.
   d) Atualize os pesos: $w_i \leftarrow w_i \cdot \exp[\alpha_m \cdot I(y_i \neq G_m(x_i))]$

3. **Saída**: O classificador final é dado por:
   $$ G(x) = sign\left[\sum_{m=1}^M \alpha_m G_m(x)\right] $$

> ✔️ **Ponto de Destaque**: O AdaBoost aumenta iterativamente a influência das observações mal classificadas, forçando o próximo classificador a se concentrar nelas [4].

#### Questões Técnicas/Teóricas

1. Como o AdaBoost lida com o problema de overfitting à medida que o número de iterações aumenta?
2. Explique por que o AdaBoost é considerado um algoritmo de "margem grande" e como isso se relaciona com sua capacidade de generalização.

### Boosting como um Modelo Aditivo

O sucesso do boosting pode ser entendido através da perspectiva de modelos aditivos [5]. O boosting ajusta uma expansão aditiva na forma:

$$ f(x) = \sum_{m=1}^M \beta_m b(x; \gamma_m) $$

onde $\beta_m$ são os coeficientes de expansão e $b(x; \gamma)$ são funções base simples caracterizadas por um conjunto de parâmetros $\gamma$.

> ❗ **Ponto de Atenção**: A escolha das funções base $b(x; \gamma)$ é crucial e varia dependendo do tipo de boosting utilizado [5].

### Modelagem Aditiva Progressiva Direta (Forward Stagewise)

O boosting aproxima a solução do problema de otimização:

$$ \min_{\{\beta_m, \gamma_m\}_1^M} \frac{1}{N} \sum_{i=1}^N L\left(y_i, \sum_{m=1}^M \beta_m b(x_i; \gamma_m)\right) $$

através de um processo iterativo [6]:

1. Inicialize $f_0(x) = 0$.
2. Para m = 1 até M:
   a) Compute $(\beta_m, \gamma_m) = \arg\min_{\beta, \gamma} \sum_{i=1}^N L(y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma))$
   b) Atualize $f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)$

Este processo adiciona novas funções base sem ajustar os parâmetros das funções já adicionadas [6].

#### Questões Técnicas/Teóricas

1. Como o processo de modelagem aditiva progressiva direta difere da otimização conjunta de todos os parâmetros? Quais são as vantagens computacionais?
2. Descreva um cenário em que a abordagem progressiva direta pode ser preferível à otimização global.

### Perda Exponencial e AdaBoost

Uma descoberta fundamental é que o AdaBoost é equivalente à modelagem aditiva progressiva direta usando a função de perda exponencial [7]:

$$ L(y, f(x)) = \exp(-yf(x)) $$

Esta equivalência fornece insights valiosos sobre o funcionamento do AdaBoost:

1. **Minimizador Populacional**: O minimizador da perda exponencial é:
   
   $$ f^*(x) = \frac{1}{2} \log \frac{Pr(Y=1|x)}{Pr(Y=-1|x)} $$

   Isto justifica o uso do sinal de $f(x)$ como regra de classificação [8].

2. **Comparação com Verossimilhança Binomial**: A perda exponencial é similar à negativa da log-verossimilhança binomial (deviance):

   $$ -l(Y, f(x)) = \log(1 + e^{-2Yf(x)}) $$

   Ambas têm o mesmo minimizador populacional, mas diferem no tratamento de observações mal classificadas [9].

> ✔️ **Ponto de Destaque**: A perda exponencial penaliza erros de classificação muito mais severamente que a deviance binomial, tornando o AdaBoost menos robusto a ruído e outliers [9].

### Funções de Perda e Robustez

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240812091152110.png" alt="image-20240812091152110" style="zoom:67%;" />

A escolha da função de perda tem implicações significativas na robustez e desempenho do modelo:

| Função de Perda   | Robustez | Aplicação                  |
| ----------------- | -------- | -------------------------- |
| Exponencial       | Baixa    | Classificação (AdaBoost)   |
| Deviance Binomial | Média    | Classificação (LogitBoost) |
| Erro Quadrático   | Baixa    | Regressão                  |
| Erro Absoluto     | Alta     | Regressão Robusta          |
| Huber             | Alta     | Regressão Robusta          |

A função de perda de Huber, por exemplo, combina as vantagens do erro quadrático próximo a zero com o erro absoluto para valores grandes:

$$ L(y, f(x)) = \begin{cases} 
[y - f(x)]^2 & \text{para } |y - f(x)| \leq \delta \\
2\delta|y - f(x)| - \delta^2 & \text{caso contrário}
\end{cases} $$

> ❗ **Ponto de Atenção**: Em aplicações de mineração de dados, onde o ruído e outliers são comuns, funções de perda robustas como Huber ou erro absoluto são preferíveis [10].

#### Questões Técnicas/Teóricas

1. Como a escolha da função de perda afeta a sensibilidade do modelo a outliers? Explique usando as propriedades matemáticas das funções discutidas.
2. Proponha uma estratégia para escolher entre perda exponencial e deviance binomial em um problema de classificação, considerando as características dos dados.

### Boosting com Árvores

As árvores de decisão são particularmente adequadas como base learners para boosting devido à sua flexibilidade e interpretabilidade [11]. O modelo boosted de árvores é uma soma de árvores:

$$ f_M(x) = \sum_{m=1}^M T(x; \Theta_m) $$

onde $T(x; \Theta_m)$ é uma árvore com parâmetros $\Theta_m$.

O processo de boosting para árvores envolve:

1. Inicialização com uma árvore simples.
2. Iterativamente adicionar novas árvores que melhor se ajustam aos resíduos atuais.
3. Atualizar o modelo combinando a nova árvore com as anteriores.

> ✔️ **Ponto de Destaque**: O boosting com árvores pode capturar interações complexas entre variáveis, tornando-o extremamente poderoso para problemas de alta dimensionalidade [11].

### Regularização em Boosting

Para evitar overfitting, várias técnicas de regularização são empregadas:

1. **Shrinkage**: Introduz um parâmetro de taxa de aprendizado $\nu$:
   
   $$ f_m(x) = f_{m-1}(x) + \nu \cdot \sum_{j=1}^J \gamma_{jm} I(x \in R_{jm}) $$

2. **Subamostragem**: Utiliza apenas uma fração $\eta$ das observações de treinamento em cada iteração.

3. **Early Stopping**: Interrompe o processo de boosting quando o erro de validação começa a aumentar.

> ⚠️ **Nota Importante**: A combinação de shrinkage e subamostragem frequentemente leva a melhores resultados, mas requer um ajuste cuidadoso dos hiperparâmetros [12].

### Conclusão

O boosting representa um avanço significativo em aprendizado de máquina, oferecendo um framework poderoso para combinar modelos fracos em um ensemble robusto. Sua flexibilidade, poder preditivo e capacidade de lidar com problemas complexos o tornaram uma ferramenta indispensável em data mining e análise preditiva.

Entretanto, o uso eficaz do boosting requer uma compreensão profunda de seus princípios subjacentes, desde a escolha apropriada da função de perda até as estratégias de regularização. A contínua pesquisa nesta área promete expandir ainda mais as capacidades e aplicações desta técnica revolucionária.

### Questões Avançadas

1. Compare e contraste o comportamento assintótico do AdaBoost com o de outros métodos de ensemble, como Random Forests. Como suas propriedades de convergência diferem e quais são as implicações práticas?

2. Desenvolva um argumento teórico para explicar por que o boosting tende a ser mais eficaz em problemas de alta dimensionalidade comparado a métodos tradicionais de regressão/classificação.

3. Proponha uma modificação no algoritmo AdaBoost para lidar especificamente com dados desbalanceados em problemas de classificação binária. Justifique sua abordagem matematicamente.

4. Analise o trade-off entre viés e variância no contexto do boosting. Como o número de iterações, a complexidade dos base learners e as técnicas de regularização afetam este trade-off?

5. Discuta as implicações computacionais e estatísticas de usar boosting em um contexto de aprendizado online ou streaming de dados. Quais modificações seriam necessárias no algoritmo padrão?

### Referências

[1] "Boosting is one of the most powerful learning ideas introduced in the last twenty years. It was originally designed for classification problems, but as will be seen in this chapter, it can profitably be extended to regression as well." (Trecho de ESL II)

[2] "A weak classifier is one whose error rate is only slightly better than random guessing. The purpose of boosting is to sequentially apply the weak classification algorithm to repeatedly modified versions of the data, thereby producing a sequence of weak classifiers G m (x), m = 1, 2, . . . , M ." (Trecho de ESL II)

[3] "The predictions from all of them are then combined through a weighted majority vote to produce the final prediction" (Trecho de ESL II)

[4] "Thus as iterations proceed, observations that are difficult to classify correctly receive ever-increasing influence. Each successive classifier is thereby forced to concentrate on those training observations that are missed by previous ones in the sequence." (Trecho de ESL II)

[5] "Boosting is a way of fitting an additive expansion in a set of elementary "basis" functions." (Trecho de ESL II)

[6] "Forward stagewise modeling approximates the solution to (10.4) by sequentially adding new basis functions to the expansion without adjusting the parameters and coefficients of those that have already been added." (Trecho de ESL II)

[7] "We now show that AdaBoost.M1 (Algorithm 10.1) is equivalent to forward stagewise additive modeling (Algorithm 10.2) using the loss function L(y, f (x)) = exp(−y f (x))." (Trecho de ESL II)

[8] "Thus, the additive expansion produced by AdaBoost is estimating one-half the log-odds of P (Y = 1|x). This justifies using its sign as the classification rule in (10.1)." (Trecho de ESL II)

[9] "Both criteria are monotone decreasing functions of the "margin" yf (x). In classification (with a −1/1 response) the margin plays a role analogous to the residuals y−f (x) in regression." (Trecho de ESL II)

[10] "These considerations suggest than when robustness is a concern, as is especially the case in data mining applications (see Section 10.7), squared-error loss for regression and exponential loss for classification are not the best criteria from a statistical perspective." (Trecho de ESL II)

[11] "Regression and classification trees are discussed in detail in Section 9.2. They partition the space of all joint predictor variable values into disjoint regions R j , j = 1, 2, . . . , J, as represented by the terminal nodes of the tree." (Trecho de ESL II)

[12] "Experience so far indicates that 4 ≤ J ≤ 8 works well in the context of boosting, with results being fairly insensitive to particular choices in this range." (Trecho de ESL II)