# Estimação de Parâmetros em Classificação de Texto Linear

<imagem: Um gráfico ilustrando a estimação dos parâmetros $\mu$ e $\phi$ em distribuições categóricas e multinomiais no contexto de classificação de texto>

## Introdução

A estimação de parâmetros é um componente crucial na classificação de texto linear, pois determina a capacidade do modelo de capturar padrões relevantes nos dados [1]. Em particular, ao lidarmos com distribuições categóricas e multinomiais, a precisão na estimação dos parâmetros afeta diretamente o desempenho de algoritmos como o classificador **Naive Bayes** [2]. ==Este resumo concentra-se na estimação dos parâmetros $\mu$ e $\phi$ dessas distribuições, utilizando a estimativa de frequência relativa, que, neste contexto, é equivalente à **Estimativa de Máxima Verossimilhança** (Maximum Likelihood Estimation - MLE) [3]==. Compreender o fundamento teórico por trás desse processo é fundamental para implementar classificadores eficazes em tarefas de Processamento de Linguagem Natural (PLN).

## Conceitos Fundamentais

| Conceito                                                     | Explicação                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Distribuição Categórica (Categorical Distribution)**       | ==Uma distribuição de probabilidade discreta que descreve o resultado de um experimento aleatório com $K$ categorias mutuamente exclusivas==, onde cada categoria tem uma probabilidade associada $\mu_k$. Em classificação de texto, é ==usada para modelar a distribuição de rótulos de classe [4].== |
| **Distribuição Multinomial (Multinomial Distribution)**      | Uma generalização da distribuição binomial para casos com $K > 2$ categorias e múltiplas contagens. ==Modela a probabilidade de um vetor de contagens $(x_1, x_2, ..., x_K)$, onde cada $x_k$ representa o número de ocorrências da categoria $k$.== No contexto de classificação de texto, ==modela a distribuição de contagens de palavras em um documento [5].== |
| **Estimativa de Máxima Verossimilhança (Maximum Likelihood Estimation - MLE)** | ==Método estatístico de estimação de parâmetros que maximiza a probabilidade (verossimilhança) de observar os dados do conjunto de treinamento dado o modelo paramétrico.== Para distribuições categóricas e multinomiais, a ==MLE produz estimativas que correspondem às frequências relativas observadas nos dados [6].== |

> ⚠️ **Nota Importante**: ==A Estimativa de Máxima Verossimilhança pode levar a problemas de overfitting, especialmente em conjuntos de dados pequenos ou esparsos.== Técnicas de regularização, como smoothing (suavização), são frequentemente necessárias para melhorar a generalização do modelo [7].

### Estimação de Parâmetros para Distribuição Categórica

<imagem: Um diagrama mostrando a contagem de ocorrências de cada categoria em um conjunto de dados, com setas apontando para as estimativas dos parâmetros correspondentes>

A distribuição categórica é essencial para modelar a probabilidade de rótulos em problemas de classificação. ==Seja $Y$ o conjunto de rótulos possíveis, a distribuição de probabilidade para um rótulo $y$ é parametrizada por $\mu_y$, onde $\sum_{y \in Y} \mu_y = 1$.==

A função de verossimilhança para um conjunto de dados de treinamento $\{y^{(i)}\}_{i=1}^N$ é dada por:

$$
L(\mu) = \prod_{i=1}^N \mu_{y^{(i)}}
$$

Tomando o logaritmo da verossimilhança, obtemos a log-verossimilhança:

$$
\ell(\mu) = \sum_{i=1}^N \log \mu_{y^{(i)}}
$$

Para maximizar $\ell(\mu)$ sujeito à restrição $\sum_{y \in Y} \mu_y = 1$, utilizamos multiplicadores de Lagrange. O Lagrangiano é:

$$
\mathcal{L}(\mu, \lambda) = \sum_{i=1}^N \log \mu_{y^{(i)}} - \lambda \left( \sum_{y \in Y} \mu_y - 1 \right)
$$

Derivando em relação a $\mu_y$ e igualando a zero:

$$
\frac{\partial \mathcal{L}}{\partial \mu_y} = \frac{\text{count}(y)}{\mu_y} - \lambda = 0
$$

Onde $\text{count}(y)$ é o número de ocorrências do rótulo $y$. Resolvendo para $\mu_y$:

$$
\mu_y = \frac{\text{count}(y)}{\lambda}
$$

Aplicando a restrição $\sum_{y \in Y} \mu_y = 1$:

$$
\sum_{y \in Y} \mu_y = \frac{N}{\lambda} = 1 \implies \lambda = N
$$

Portanto, a ==estimativa de máxima verossimilhança para $\mu_y$ é:==

$$
\mu_y = \frac{\text{count}(y)}{N}
$$

Esta estimativa corresponde à frequência relativa do rótulo $y$ no conjunto de treinamento [8].

#### Perguntas Teóricas

1. **Prove que a estimativa de máxima verossimilhança para $\mu$ na distribuição categórica é equivalente à estimativa de frequência relativa apresentada acima.**
2. **Como a estimativa de $\mu$ seria afetada se tivéssemos um conjunto de dados extremamente desbalanceado? Discuta as implicações teóricas e práticas da estimativa de máxima verossimilhança neste cenário.**
3. **Derive a expressão para o erro padrão da estimativa $\hat{\mu}_y$ e explique como isso pode ser usado para construir intervalos de confiança para os parâmetros estimados.**

### Estimação de Parâmetros para Distribuição Multinomial

<imagem: Um gráfico de barras mostrando as contagens de palavras em documentos de diferentes classes, com linhas pontilhadas indicando as estimativas de $\phi$ para cada classe>

==No modelo **Naive Bayes** para classificação de texto, assumimos que, dado um rótulo $y$, as palavras em um documento são geradas a partir de uma distribuição multinomial parametrizada por $\phi_y = (\phi_{y,1}, \phi_{y,2}, ..., \phi_{y,V})$, onde $V$ é o tamanho do vocabulário e $\sum_{j=1}^V \phi_{y,j} = 1$ para cada $y$.==

Para um documento representado como um vetor de contagens de palavras $\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, ..., x_V^{(i)})$, a probabilidade de observar $\mathbf{x}^{(i)}$ dado o rótulo $y^{(i)}$ é:

$$
p_{\text{mult}}(\mathbf{x}^{(i)}; \phi_{y^{(i)}}) = \frac{(\sum_{j=1}^V x_j^{(i)})!}{\prod_{j=1}^V x_j^{(i)}!} \prod_{j=1}^V \phi_{y^{(i)},j}^{x_j^{(i)}}
$$

A log-verossimilhança total para o conjunto de treinamento $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ é:

$$
\ell(\phi) = \sum_{i=1}^N \left( \log \frac{(\sum_{j=1}^V x_j^{(i)})!}{\prod_{j=1}^V x_j^{(i)}!} + \sum_{j=1}^V x_j^{(i)} \log \phi_{y^{(i)},j} \right)
$$

Como o primeiro termo não depende de $\phi$, podemos focar em maximizar o segundo termo. Para cada classe $y$, a função a ser maximizada é:

$$
\ell_y(\phi_y) = \sum_{i:y^{(i)}=y} \sum_{j=1}^V x_j^{(i)} \log \phi_{y,j}
$$

Sujeito à restrição $\sum_{j=1}^V \phi_{y,j} = 1$.

Utilizando multiplicadores de Lagrange, o Lagrangiano é:

$$
\mathcal{L}(\phi_y, \lambda_y) = \sum_{i:y^{(i)}=y} \sum_{j=1}^V x_j^{(i)} \log \phi_{y,j} - \lambda_y \left( \sum_{j=1}^V \phi_{y,j} - 1 \right)
$$

Derivando em relação a $\phi_{y,j}$ e igualando a zero:

$$
\frac{\partial \mathcal{L}}{\partial \phi_{y,j}} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{\phi_{y,j}} - \lambda_y = 0
$$

Resolvendo para $\phi_{y,j}$:

$$
\phi_{y,j} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{\lambda_y}
$$

Aplicando a restrição $\sum_{j=1}^V \phi_{y,j} = 1$:

$$
\sum_{j=1}^V \phi_{y,j} = \frac{\sum_{j=1}^V \sum_{i:y^{(i)}=y} x_j^{(i)}}{\lambda_y} = \frac{T_y}{\lambda_y} = 1 \implies \lambda_y = T_y
$$

Onde $T_y = \sum_{j=1}^V \sum_{i:y^{(i)}=y} x_j^{(i)}$ é o total de contagens de palavras em documentos com rótulo $y$.

==Portanto, a estimativa de máxima verossimilhança para $\phi_{y,j}$ é:==
$$
\phi_{y,j} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{T_y} = \frac{\text{count}(y, j)}{\sum_{j'=1}^V \text{count}(y, j')}
$$

Esta estimativa corresponde à frequência relativa da palavra $j$ em documentos com rótulo $y$ [9].

> ❗ **Ponto de Atenção**: A estimativa de máxima verossimilhança para $\phi$ pode resultar em probabilidades zero para palavras que não aparecem em nenhum documento de uma determinada classe no conjunto de treinamento. Isso pode levar a problemas na classificação, pois uma única palavra ausente pode zerar a probabilidade de uma classe inteira durante a inferência [10].

Para mitigar este problema, é comum utilizar técnicas de suavização, como a **suavização de Laplace** (Laplace smoothing):

$$
\phi_{y,j} = \frac{\alpha + \sum_{i:y^{(i)}=y} x_j^{(i)}}{V\alpha + \sum_{j'=1}^V \sum_{i:y^{(i)}=y} x_{j'}^{(i)}}
$$

Onde $\alpha > 0$ é um hiperparâmetro que controla o grau de suavização [11].

#### Perguntas Teóricas

1. **Demonstre matematicamente por que a estimativa de máxima verossimilhança para $\phi$ na distribuição multinomial é equivalente à estimativa de frequência relativa apresentada.**
2. **Derive a expressão para o gradiente da log-verossimilhança com respeito a $\phi_{y,j}$ e mostre que o ponto onde este gradiente é zero corresponde à estimativa de frequência relativa.**
3. **Analise teoricamente o impacto da suavização de Laplace na variância das estimativas de $\phi$. Como isso afeta o trade-off entre viés e variância no modelo?**

### Justificativa Teórica para Estimação de Máxima Verossimilhança

==A **Estimativa de Máxima Verossimilhança** (MLE) é um método fundamental em estatística para a estimação de parâmetros de modelos probabilísticos.== A ideia central é selecionar os parâmetros que ==tornam os dados observados mais prováveis sob o modelo.== No contexto de distribuições categóricas e multinomiais, a MLE oferece estimativas intuitivas que correspondem às frequências observadas nos dados.

**Propriedades da MLE:**

- **Não-viesada**: As estimativas de MLE são, em geral, não-viesadas, ou seja, o valor esperado das estimativas é igual ao verdadeiro valor do parâmetro.
- **Consistência**: ==À medida que o tamanho da amostra aumenta, as estimativas de MLE convergem em probabilidade para o verdadeiro valor do parâmetro.==
- **Eficiência Assintótica**: As estimativas de MLE atingem a variância mínima possível (limite de Cramér-Rao) quando o tamanho da amostra tende ao infinito [12].

No caso da distribuição multinomial, mostramos que a MLE para $\phi_{y,j}$ corresponde à frequência relativa da palavra $j$ nos documentos de classe $y$. Esta estimativa maximiza a probabilidade de observar as contagens de palavras nos documentos dado o modelo multinomial.

**Derivação via Informação de Fisher:**

A matriz de **Informação de Fisher** $I(\theta)$ para um conjunto de parâmetros $\theta$ é definida como:

$$
I(\theta) = -E \left[ \frac{\partial^2 \ell(\theta)}{\partial \theta \partial \theta^\top} \right]
$$

Onde $\ell(\theta)$ é a log-verossimilhança. A variância das estimativas de MLE é dada pelo inverso da matriz de Informação de Fisher, ou seja, $\text{Var}(\hat{\theta}) \approx I(\theta)^{-1}$.

Para a distribuição multinomial, podemos calcular $I(\phi_{y})$ e verificar que a MLE atinge o limite inferior de Cramér-Rao, evidenciando sua eficiência assintótica [13].

> ✔️ **Destaque**: A MLE é particularmente adequada quando temos um grande conjunto de dados, pois suas propriedades assintóticas garantem estimativas precisas. No entanto, em conjuntos de dados pequenos ou esparsos, a MLE pode apresentar variância elevada, justificando o uso de técnicas de regularização [14].

#### Perguntas Teóricas

1. **Derive a matriz de Informação de Fisher para a distribuição multinomial e use-a para calcular o limite inferior de Cramér-Rao para as estimativas de $\phi$.**
2. **Prove que a estimativa de máxima verossimilhança para $\phi$ é consistente, ou seja, converge em probabilidade para o verdadeiro valor do parâmetro à medida que o tamanho da amostra aumenta.**
3. **Analise teoricamente como a adição de regularização L2 na log-verossimilhança afetaria as estimativas de $\phi$. Derive as novas equações para as estimativas regularizadas e discuta o impacto no viés e na variância.**

## Conclusão

A estimação de parâmetros para distribuições categóricas e multinomiais é fundamental na classificação de texto linear, especialmente no contexto do classificador **Naive Bayes** [15]. A utilização da **Estimativa de Máxima Verossimilhança** fornece uma base teórica sólida para a estimação dos parâmetros $\mu$ e $\phi$, correspondendo às frequências observadas nos dados e garantindo propriedades estatísticas desejáveis, como consistência e eficiência assintótica [16].

No entanto, é crucial considerar técnicas de regularização, como a suavização de Laplace, para lidar com problemas de esparsidade e evitar probabilidades zero, que podem comprometer o desempenho do modelo em dados não vistos [17]. A compreensão profunda dos fundamentos teóricos da MLE permite não apenas a implementação eficaz de algoritmos de classificação, mas também a capacidade de aprimorá-los e adaptá-los a diferentes cenários e desafios em **Processamento de Linguagem Natural** [18].

À medida que avançamos para modelos mais complexos, como redes neurais profundas, os princípios fundamentais de estimação de parâmetros continuam relevantes, formando a base para compreender e melhorar algoritmos mais avançados de aprendizado de máquina em processamento de texto [19].

## Perguntas Teóricas Avançadas

1. **Derive a expressão para a informação mútua entre as features e os rótulos em um classificador Naive Bayes multinomial. Como essa medida poderia ser usada para selecionar features de forma teoricamente fundamentada?**
2. **Considere um cenário onde os documentos têm comprimentos muito variados. Proponha e analise teoricamente uma modificação na estimativa de $\phi$ que leve em conta essa variação, discutindo suas propriedades estatísticas.**
3. **Compare teoricamente a variância das estimativas de $\phi$ obtidas por máxima verossimilhança com as obtidas por inferência bayesiana usando uma prior Dirichlet. Em que condições a abordagem bayesiana seria preferível?**
4. **Prove que, para um conjunto de dados fixo, à medida que o parâmetro de suavização $\alpha$ aumenta, as estimativas de $\phi$ para todas as classes convergem para uma distribuição uniforme. Discuta as implicações desse resultado para a escolha de $\alpha$.**

## Referências

[1] "To predict a label from a bag-of-words, we can assign a score to each word in the vocabulary, measuring the compatibility with the label." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[2] "Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong independence assumptions between the features." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[3] "Maximum Likelihood Estimation (MLE) is a method of estimating the parameters of a statistical model that maximizes the likelihood function." *(Trecho de um texto sobre MLE)*

[4] "The categorical distribution is a generalization of the Bernoulli distribution for a random variable that can take on one of K possible categories, with the probability of each category separately specified." *(Definição comum em estatística)*

[5] "The multinomial distribution is a generalization of the binomial distribution. It models the probability of counts for each side of a K-sided die rolled N times." *(Definição comum em estatística)*

[6] "In the context of the multinomial distribution, the maximum likelihood estimates of the parameters are the observed proportions in each category." *(Conceito padrão em estatística)*

[7] "Regularization techniques are used to prevent overfitting by adding additional information or constraints to the model." *(Conceito geral em aprendizado de máquina)*

[8] "The MLE for the parameters of a categorical distribution is the relative frequency of each category in the data." *(Resultado clássico em estatística)*

[9] "For the multinomial distribution, the MLE of the parameter vector is given by the vector of observed proportions." *(Resultado clássico em estatística)*

[10] "Zero probabilities in Naive Bayes can be problematic; smoothing methods like Laplace smoothing adjust estimates to avoid zero probabilities." *(Discussão comum em PLN)*

[11] "Laplace smoothing (additive smoothing) adds a small constant to each count to ensure that all probabilities are non-zero." *(Conceito padrão em PLN)*

[12] "The Fisher Information measures the amount of information that an observable random variable carries about an unknown parameter." *(Definição em estatística matemática)*

[13] "The Cramér-Rao lower bound provides a lower bound on the variance of estimators of a parameter." *(Resultado fundamental em teoria da estimação)*

[14] "Regularization introduces bias into parameter estimates but can reduce variance, leading to better model performance on new data." *(Conceito chave em aprendizado de máquina)*

[15] "Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of features and classes." *(Propriedade dos classificadores Naive Bayes)*

[16] "MLE estimators have desirable properties under certain conditions, such as consistency, efficiency, and asymptotic normality." *(Propriedades estatísticas da MLE)*

[17] "Data sparsity is a common issue in NLP, making smoothing techniques essential for robust probability estimation." *(Desafio comum em PLN)*

[18] "Understanding the theoretical foundations of algorithms allows for better adaptation and improvement in practical applications." *(Importância da teoria em prática)*

[19] "Even with the rise of deep learning, foundational statistical methods remain relevant for understanding and interpreting models." *(Conexão entre métodos tradicionais e modernos)*