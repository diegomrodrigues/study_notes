# Classificadores Lineares para Texto

<imagem: Um diagrama mostrando um espaço vetorial de alta dimensão com vetores de palavras e um hiperplano separador representando um classificador linear>

## Introdução

Os **classificadores lineares** são uma classe fundamental de algoritmos de aprendizado de máquina amplamente utilizados para a classificação de texto. ==Eles operam com base em uma função de pontuação linear, calculada como o produto interno entre um vetor de pesos e um vetor de características== [1]. Este resumo explora em profundidade os conceitos, teorias e aplicações dos classificadores lineares no contexto da classificação de texto, fornecendo uma visão abrangente e avançada para cientistas de dados e pesquisadores na área de processamento de linguagem natural.

A classificação de texto é uma tarefa crucial em muitas aplicações, como ==filtragem de spam, categorização de notícias e análise de sentimentos==. Os classificadores lineares oferecem uma abordagem eficiente e interpretável para essas tarefas, ==baseando-se na representação do texto como um vetor de características e na aprendizagem de um conjunto de pesos que determinam a importância relativa de cada característica para a classificação [2].==

> ⚠️ **Nota Importante**: A eficácia dos classificadores lineares ==depende fortemente da escolha adequada das características e da capacidade de capturar relações lineares nos dados.== Em cenários onde as relações são altamente não-lineares, podem ser necessárias abordagens mais complexas [3].

## Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Vetor de Características** | ==Representação numérica de um documento de texto, geralmente usando a abordagem bag-of-words.== Cada dimensão corresponde a uma palavra do vocabulário, e o valor indica a contagem ou presença da palavra no documento [4]. |
| **Vetor de Pesos**           | ==Um vetor θ que atribui um peso a cada característica, indicando sua importância para a classificação==. Estes pesos são aprendidos durante o treinamento do modelo [5]. |
| **Função de Pontuação**      | ==Uma função linear que calcula a compatibilidade entre um documento e uma classe==, definida como o produto interno entre o vetor de características e o vetor de pesos: Ψ(x, y) = θ · f(x, y) [6]. |
| **Margem**                   | A diferença entre a pontuação da classe correta e a pontuação da classe incorreta mais próxima. Maximizar a margem é um objetivo comum em muitos algoritmos de classificação linear [7]. |

### Representação Matemática

A função de pontuação para um classificador linear é definida como:

$$
\Psi(x, y) = \theta \cdot f(x, y) = \sum_{j} \theta_j f_j(x, y)
$$

Onde:
- $x$ é o vetor de características do documento
- $y$ é a classe
- $\theta$ é o vetor de pesos
- $f(x, y)$ é a função de características que mapeia o par (documento, classe) para um vetor [8]

> 💡 **Destaque**: A função de características $f(x, y)$ permite incorporar informações específicas da classe na representação do documento, o que é crucial para classificação multiclasse eficiente [9].

### Perguntas Teóricas

1. Derive a expressão para a margem em um classificador linear binário e explique como ela se relaciona com a noção de separabilidade linear.

2. Demonstre matematicamente por que um classificador linear não pode resolver o problema do XOR. Como isso se relaciona com a limitação dos perceptrons discutida por Minsky e Papert?

3. Analise teoricamente o impacto da dimensionalidade do espaço de características na capacidade de generalização de um classificador linear. Como isso se relaciona com o fenômeno conhecido como "maldição da dimensionalidade"?

## Algoritmos de Classificação Linear

### Naive Bayes

==O classificador Naive Bayes é um modelo probabilístico que, apesar de sua simplicidade, pode ser interpretado como um classificador linear no espaço de log-probabilidades [10].==

A probabilidade condicional de uma classe y dado um documento x é calculada como:

$$
p(y|x; \phi) = \frac{\prod_{j=1}^V \phi_{y,j}^{x_j}}{\sum_{y'\in Y} \prod_{j=1}^V \phi_{y',j}^{x_j}}
$$

Onde:
- $\phi_{y,j}$ é a probabilidade da palavra j na classe y
- $x_j$ é a contagem da palavra j no documento
- $V$ é o tamanho do vocabulário [11]

> ❗ **Ponto de Atenção**: ==A suposição de independência condicional entre as características, conhecida como a suposição "naive"==, é crucial para a eficiência computacional do Naive Bayes, mas pode limitar sua precisão em alguns cenários [12].

### Perceptron

O algoritmo Perceptron é um dos classificadores lineares mais antigos e fundamentais. Ele atualiza iterativamente os pesos com base nos erros de classificação [13].

O algoritmo de aprendizagem do Perceptron é definido como:

```python
def perceptron(x, y, max_iterations):
    theta = np.zeros(len(x[0]))
    for _ in range(max_iterations):
        for i in range(len(x)):
            y_pred = np.sign(np.dot(theta, x[i]))
            if y_pred != y[i]:
                theta += y[i] * x[i]
    return theta
```

> ✔️ **Destaque**: O Perceptron é garantido convergir para uma solução se os dados forem linearmente separáveis. Esta propriedade é conhecida como o Teorema de Convergência do Perceptron [14].

### Support Vector Machine (SVM)

O SVM busca encontrar o hiperplano que maximiza a margem entre as classes. Para dados não linearmente separáveis, introduz-se variáveis de folga ξᵢ [15].

A formulação do problema de otimização para o SVM é:

$$
\min_{\theta, \xi} \frac{1}{2} ||\theta||^2 + C \sum_{i=1}^N \xi_i
$$

Sujeito a:
$$
y^{(i)}(\theta \cdot x^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i
$$

Onde C é um hiperparâmetro que controla o trade-off entre maximizar a margem e minimizar o erro de treinamento [16].

### Regressão Logística

A regressão logística modela a probabilidade de uma classe usando a função logística (sigmoid) [17]:

$$
p(y|x; \theta) = \frac{1}{1 + e^{-\theta \cdot x}}
$$

O treinamento é realizado maximizando a log-verossimilhança:

$$
\mathcal{L}(\theta) = \sum_{i=1}^N [y^{(i)} \log p(y^{(i)}|x^{(i)}; \theta) + (1-y^{(i)}) \log (1-p(y^{(i)}|x^{(i)}; \theta))]
$$

> 💡 **Destaque**: A regressão logística pode ser vista como uma versão probabilística do SVM, oferecendo não apenas classificações, mas também probabilidades bem calibradas [18].

### Perguntas Teóricas

1. Derive a atualização de gradiente para o algoritmo Perceptron e mostre como ela se relaciona com a minimização do erro de classificação.

2. Prove que a função de perda da regressão logística é convexa em relação aos parâmetros θ. Como isso afeta o processo de otimização?

3. Compare teoricamente a capacidade de generalização do SVM com margens rígidas versus margens suaves. Em que condições cada variante seria preferível?

## Otimização e Regularização

A otimização dos parâmetros em classificadores lineares geralmente envolve a minimização de uma função de perda, frequentemente com um termo de regularização para evitar overfitting [19].

### Gradiente Descendente Estocástico (SGD)

O SGD é um método de otimização amplamente utilizado devido à sua eficiência em grandes conjuntos de dados. A atualização dos pesos é dada por:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta^{(t)} \nabla_\theta \ell(\theta^{(t)}; x^{(i)}, y^{(i)})
$$

Onde $\eta^{(t)}$ é a taxa de aprendizado na iteração t [20].

### Regularização L2

A regularização L2 adiciona um termo de penalidade à função objetivo:

$$
\mathcal{L}(\theta) = \sum_{i=1}^N \ell(\theta; x^{(i)}, y^{(i)}) + \frac{\lambda}{2} ||\theta||_2^2
$$

Onde $\lambda$ é o parâmetro de regularização que controla a força da penalidade [21].

> ⚠️ **Nota Importante**: A escolha do parâmetro de regularização λ é crucial. Um valor muito alto pode levar a underfitting, enquanto um valor muito baixo pode resultar em overfitting [22].

### Perguntas Teóricas

1. Derive a expressão para o gradiente da função de perda regularizada L2 para a regressão logística. Como a regularização afeta a atualização dos pesos?

2. Analise teoricamente o comportamento assintótico do SGD em um problema de otimização convexa. Sob quais condições a convergência é garantida?

3. Compare matematicamente os efeitos da regularização L1 e L2 na esparsidade dos pesos aprendidos. Como isso se relaciona com a seleção de características?

## Avaliação e Interpretação

A avaliação de classificadores lineares envolve métricas como acurácia, precisão, recall e F1-score. A interpretação dos pesos aprendidos pode fornecer insights valiosos sobre a importância relativa das características [23].

### Análise de Pesos

Os pesos θ_j podem ser interpretados como a importância relativa da característica j para a classificação. Características com pesos positivos altos são indicativas da classe positiva, enquanto pesos negativos altos são indicativos da classe negativa [24].

### Calibração de Probabilidades

Para modelos que produzem probabilidades (como regressão logística), a calibração é importante. Técnicas como Platt Scaling podem ser usadas para melhorar a calibração [25].

> 💡 **Destaque**: A interpretabilidade dos classificadores lineares é uma de suas principais vantagens, permitindo insights diretos sobre o processo de decisão do modelo [26].

## Conclusão

Os classificadores lineares são ferramentas poderosas e versáteis para a classificação de texto, oferecendo um equilíbrio entre simplicidade, eficiência computacional e interpretabilidade. Sua formulação matemática elegante permite análises teóricas profundas, enquanto sua aplicabilidade prática os torna indispensáveis em muitas tarefas de processamento de linguagem natural [27].

Embora tenham limitações em capturar relações não-lineares complexas, os avanços em técnicas de engenharia de características e métodos de kernel expandiram significativamente sua aplicabilidade. A compreensão profunda dos fundamentos teóricos dos classificadores lineares é essencial para o desenvolvimento e aplicação eficaz de técnicas mais avançadas em aprendizado de máquina e processamento de linguagem natural [28].

## Perguntas Teóricas Avançadas

1. Derive a expressão para o dual do problema de otimização do SVM e explique como isso leva à formulação do "kernel trick". Como isso se relaciona com o Teorema de Representer?

2. Analise teoricamente o trade-off entre viés e variância em classificadores lineares. Como a complexidade do modelo (por exemplo, através da regularização) afeta este trade-off?

3. Desenvolva uma prova formal para o Teorema de Convergência do Perceptron. Quais são as implicações deste teorema para dados não linearmente separáveis?

4. Compare teoricamente a capacidade expressiva de classificadores lineares com redes neurais de uma camada oculta. Em que condições um classificador linear pode aproximar arbitrariamente bem uma função de decisão não-linear?

5. Analise o impacto da maldição da dimensionalidade na performance de classificadores lineares. Como técnicas de redução de dimensionalidade, como PCA, afetam teoricamente a capacidade de generalização destes modelos?

## Referências

[1] "To predict a label from a bag-of-words, we can assign a score to each word in the vocabulary, measuring the compatibility with the label." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[2] "For example, for the label FICTION, we might assign a positive score to the word whale, and a negative score to the word molybdenum. These scores are called weights, and they are arranged in a column vector θ." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[3] "In a linear bag-of-words classifier, this score is the vector inner product between the weights θ and the output of a feature function f(x, y)." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[4] "As the notation suggests, f is a function of two arguments, the word counts x and the label y, and it returns a vector output." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[5] "The corresponding weight θ_j then scores the compatibility of the word whale with the label FICTION." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[6] "Ψ(x, y) = θ · f(x, y) = ∑ θ_j f_j(x, y)." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[7] "The output of the feature function can be formalized as a vector:" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[8] "This is identical to the "naïve" independence assumption implied by the multinomial distribution, and as a result, the optimal parameters for this model are identical to those in multinomial Naïve Bayes." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[9] "For example, man bites dog and dog bites man correspond to an identical count vector, {bites : 1, dog : 1, man : 1}, and B(x) is equal to the total number of possible word orderings for count vector x." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[10] "The Naïve Bayes prediction rule is to choose the label y which maximizes log p(x, y; μ, ϕ):" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[11] "= log B(x) + ∑(j=1 to V) x_j log ϕ_y,j + log μ_y" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[12] "This is called Laplace smoothing." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[13] "Algorithm 3 Perceptron learning algorithm" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[14] "Linear separability is important because of the following guarantee: if your data is linearly separable, then the perceptron algorithm will find a separator (Novikoff, 1962)." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[15] "In the support vector machine, the objective is the regularized margin loss," *(Tr