# Representação Bag-of-Words: Modelagem de Texto como Vetor de Contagens de Palavras

<imagem: Um diagrama mostrando um texto sendo transformado em um vetor esparso, onde cada dimensão representa uma palavra do vocabulário e os valores são as contagens das palavras no texto.>

## Introdução

A representação **Bag-of-Words** (BoW) é uma técnica fundamental na área de processamento de linguagem natural (NLP) e classificação de texto [1]. Esta abordagem simplifica a representação de documentos de texto, convertendo-os em vetores numéricos que capturam a frequência das palavras, ignorando a ordem em que elas aparecem [2]. Apesar de sua simplicidade, o modelo BoW é surpreendentemente eficaz em muitas tarefas de NLP, fornecendo uma base sólida para algoritmos mais complexos [3].

## Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Bag-of-Words**             | Representação vetorial de um documento onde cada elemento corresponde à contagem de uma palavra específica no vocabulário, ignorando a ordem das palavras [4]. |
| **Vocabulário**              | Conjunto de todas as palavras únicas presentes no corpus de documentos [5]. |
| **Vetor de Características** | Representação numérica de um documento, onde cada dimensão corresponde a uma palavra no vocabulário [6]. |

> ⚠️ **Nota Importante**: A representação BoW trata cada palavra como um token independente, perdendo informações sobre a estrutura gramatical e a ordem das palavras no texto original [7].

> ❗ **Ponto de Atenção**: Embora simples, o BoW é a base para muitos modelos mais avançados de NLP e pode ser surpreendentemente eficaz em tarefas de classificação de texto [8].

> ✔️ **Destaque**: A eficácia do BoW reside em sua capacidade de capturar a essência temática de um documento através da frequência de palavras, permitindo comparações e análises estatísticas robustas [9].

## Formalização Matemática do Modelo Bag-of-Words

<imagem: Um diagrama mostrando a transformação de um texto em um vetor esparso, com setas indicando como as palavras são mapeadas para índices no vetor.>

A representação matemática do modelo Bag-of-Words é fundamental para entender sua implementação e aplicações em aprendizado de máquina [10]. Vamos formalizar o conceito:

Seja $D = \{d_1, d_2, ..., d_N\}$ um conjunto de $N$ documentos e $V = \{w_1, w_2, ..., w_M\}$ o vocabulário de $M$ palavras únicas em $D$ [11]. A representação BoW de um documento $d_i$ é um vetor $x_i \in \mathbb{R}^M$, onde:

$$
x_i = [x_{i1}, x_{i2}, ..., x_{iM}]
$$

E cada elemento $x_{ij}$ representa a contagem da palavra $w_j$ no documento $d_i$ [12].

Para um classificador linear, podemos definir uma função de pontuação $\Psi(x, y)$ que mede a compatibilidade entre um vetor de características $x$ e uma classe $y$ [13]:

$$
\Psi(x, y) = \theta \cdot f(x, y) = \sum_j \theta_j f_j(x, y)
$$

Onde $\theta$ é um vetor de pesos e $f(x, y)$ é uma função de características que mapeia o par $(x, y)$ para um vetor [14].

### Exemplo de Função de Características

Uma função de características simples pode ser definida como [15]:

$$
f_j(x, y) = \begin{cases}
    x_\text{whale}, & \text{se } y = \text{FICTION} \\
    0, & \text{caso contrário}
\end{cases}
$$

Esta função retorna a contagem da palavra "whale" se a classe for FICTION, e zero caso contrário [16].

### Perguntas Teóricas

1. Derive a expressão para a probabilidade condicional $p(y|x;\theta)$ em um classificador logístico usando a representação BoW.

2. Demonstre como a função de características $f(x, y)$ pode ser formalizada como um vetor para um problema de classificação multiclasse com $K$ classes.

3. Analise teoricamente o impacto da esparsidade do vetor BoW na complexidade computacional e na eficácia de um classificador linear.

## Estimação e Aprendizado

O processo de aprendizado em modelos que utilizam a representação BoW geralmente envolve a estimação de parâmetros que maximizam alguma função objetivo [17]. Um exemplo clássico é o classificador Naïve Bayes, que estima as probabilidades $\phi_{y,j}$ da seguinte forma [18]:

$$
\phi_{y,j} = \frac{\text{count}(y, j)}{\sum_{j'=1}^V \text{count}(y, j')} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{\sum_{j'=1}^V \sum_{i:y^{(i)}=y} x_{j'}^{(i)}}
$$

Onde $\text{count}(y, j)$ é a contagem da palavra $j$ em documentos com rótulo $y$ [19].

Esta estimativa de máxima verossimilhança pode ser justificada maximizando a log-verossimilhança [20]:

$$
\mathcal{L}(\phi, \mu) = \sum_{i=1}^N \log p_\text{mult}(x^{(i)}; \phi_{y(i)}) + \log p_\text{cat}(y^{(i)}; \mu)
$$

Onde $p_\text{mult}$ é a distribuição multinomial e $p_\text{cat}$ é a distribuição categórica [21].

> ⚠️ **Nota Importante**: A estimação de máxima verossimilhança pode levar a problemas de overfitting, especialmente para palavras raras. Técnicas de suavização, como a suavização de Laplace, são frequentemente aplicadas para mitigar este problema [22].

### Suavização de Laplace

A suavização de Laplace adiciona um pseudocontagem $\alpha$ a cada contagem antes da normalização [23]:

$$
\phi_{y,j} = \frac{\alpha + \text{count}(y, j)}{V\alpha + \sum_{j'=1}^V \text{count}(y, j')}
$$

Esta técnica ajuda a evitar probabilidades zero e melhora a generalização do modelo [24].

### Perguntas Teóricas

1. Derive a expressão para o gradiente da log-verossimilhança em relação a $\phi_{y,j}$ no modelo Naïve Bayes com representação BoW.

2. Analise teoricamente o impacto da suavização de Laplace na variância e viés do estimador de $\phi_{y,j}$.

3. Demonstre como a suavização de Laplace pode ser interpretada como uma forma de regularização bayesiana.

## Otimização para Classificadores BoW

A otimização de classificadores que utilizam a representação BoW geralmente envolve a minimização de uma função de perda sobre o conjunto de treinamento [25]. Para um classificador logístico, por exemplo, a função objetivo regularizada é dada por [26]:

$$
L_\text{LOGREG} = \frac{\lambda}{2} ||\theta||_2^2 - \sum_{i=1}^N \left(\theta \cdot f(x^{(i)},y^{(i)}) - \log \sum_{y\in Y} \exp(\theta \cdot f(x^{(i)}, y))\right)
$$

Onde $\lambda$ é o parâmetro de regularização L2 [27].

O gradiente desta função de perda é dado por [28]:

$$
\nabla_\theta L_\text{LOGREG} = \lambda\theta - \sum_{i=1}^N \left(f(x^{(i)}, y^{(i)}) - E_{Y|X}[f(x^{(i)}, y)]\right)
$$

Este gradiente pode ser utilizado em algoritmos de otimização como o Gradiente Descendente Estocástico (SGD) [29].

### Algoritmo: Gradiente Descendente Generalizado

```python
def gradient_descent(x, y, L, eta, batcher, T_max):
    theta = np.zeros(len(x[0]))
    t = 0
    while t < T_max:
        batches = batcher(len(x))
        for batch in batches:
            t += 1
            gradient = compute_gradient(L, theta, x[batch], y[batch])
            theta -= eta[t] * gradient
            if converged(theta):
                return theta
    return theta
```

Este algoritmo genérico pode ser adaptado para diferentes esquemas de batchingpara otimização em lote, SGD ou mini-batch SGD [30].

### Perguntas Teóricas

1. Derive a expressão para o gradiente da função de perda do SVM (Support Vector Machine) utilizando a representação BoW.

2. Analise teoricamente a convergência do algoritmo SGD para a função de perda logística com regularização L2.

3. Demonstre como o algoritmo de Perceptron pode ser visto como um caso especial de SGD aplicado a uma função de perda específica.

## Conclusão

A representação Bag-of-Words, apesar de sua simplicidade, continua sendo uma técnica fundamental em NLP e classificação de texto [31]. Sua eficácia reside na capacidade de capturar informações essenciais sobre o conteúdo do documento, permitindo a aplicação de uma ampla gama de algoritmos de aprendizado de máquina [32]. Embora tenha limitações, como a perda de informações sobre a ordem das palavras, o BoW serve como base para muitos modelos mais avançados e continua sendo uma ferramenta valiosa no arsenal de qualquer cientista de dados trabalhando com texto [33].

## Perguntas Teóricas Avançadas

1. Derive a expressão para a entropia condicional H(Y|X) em um modelo de classificação que utiliza a representação BoW, e analise como esta quantidade se relaciona com a performance do classificador.

2. Considerando um problema de classificação multiclasse com K classes, demonstre matematicamente como a representação BoW pode ser estendida para incorporar n-gramas, e analise o impacto desta extensão na complexidade computacional e na capacidade expressiva do modelo.

3. Derive a forma dual do problema de otimização para um SVM com kernel utilizando a representação BoW, e discuta as implicações teóricas de usar diferentes funções de kernel neste contexto.

4. Analise teoricamente o trade-off entre viés e variância em modelos que utilizam a representação BoW, considerando o tamanho do vocabulário como um hiperparâmetro. Como este trade-off se relaciona com o fenômeno de "maldição da dimensionalidade"?

5. Desenvolva uma prova formal para mostrar que, sob certas condições, a representação BoW é suficiente para garantir a separabilidade linear de classes em um espaço de alta dimensão. Quais são as implicações práticas desta propriedade?

## Referências

[1] "To predict a label from a bag-of-words, we can assign a score to each word in the vocabulary, measuring the compatibility with the label." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[2] "Suppose that you want a multiclass classifier, where K ≜ |Y| > 2. For example, you might want to classify news stories about sports, celebrities, music, and business. The goal is to predict a label y, given the bag of words x, using the weights θ." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[3] "In a linear bag-of-words classifier, this score is the vector inner product between the weights θ and the output of a feature function f(x, y)." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[4] "Ψ(x, y) = θ · f(x, y) = ∑ θ_j f_j(x, y)." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[5] "As the notation suggests, f is a function of two arguments, the word counts x and the label y, and it returns a vector output." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[6] "For example, given arguments x and y, element j of this feature vector might be, f_j(x, y) = {x_whale, if y = FICTION  [2.2] {0,      otherwise" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[7] "This function returns the count of the word whale if the label is FICTION, and it returns zero otherwise." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[8] "The index j depends on the position of whale in the vocabulary, and of FICTION in the set of possible labels." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[9] "The corresponding weight θ_j then scores the compatibility of the word whale with the label FICTION." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[10] "A positive score means that this word makes the label more likely." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[11] "The output of the feature function can be formalized as a vector: f(x, y = 1) = [x; 0; 0; . . . ; 0]  [2.3] (K−1)×V" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[12] "f(x, y = 2) = [0; 0; . . . ; 0; x; 0; 0; . . . ; 0]  [2.4] V        (K−2)×V" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[13] "f(x, y = K) = [0; 0; . . . ; 0; x],  [2.5] (K−1)×V" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[14] "where [0; 0; . . . ; 0] is a column vector of (K − 1) × V zeros, and the semicolon indicates (K−1)×V vertical concatenation." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[15] "For each of the K possible labels, the feature function returns a associated dictionary." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[16] "For example,³ θ(E,bicycle) = 1           θ(S,bicycle) = 0 θ(E,bicicleta) = 0         θ(S,bicicleta) = 1 θ(E,car) = 1               θ(S,car) = 1 θ(E,ordinateur) = 0        θ(S,ordinateur) = 0." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[17] "But it is usually not easy to set classification weights by hand, due to the large number of words and the difficulty of selecting exact numerical weights." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[18] "Instead, we will learn the weights from data. Email users manually label messages as SPAM; newspapers label their own articles as BUSINESS or STYLE." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[19] "Using such instance labels