## Classificação: Fundamentos e Técnicas

![image-20240807095431485](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240807095431485.png)

### Introdução

A classificação é uma tarefa fundamental em aprendizado de máquina e estatística, onde o objetivo é prever uma variável resposta categórica com base em variáveis preditoras [1][2]. Enquanto a regressão lida com respostas quantitativas contínuas, a classificação aborda respostas qualitativas discretas, sendo frequentemente mais comum em aplicações práticas [1][2].

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Variável Categórica**      | Variável que assume um número limitado de valores discretos, como cores de olhos ou tipos de e-mail (spam/ham) [1][2] |
| **Classificador**            | Função que mapeia um vetor de características X para uma classe em um conjunto C de valores discretos [2][3] |
| **Probabilidades de Classe** | Estimativas da probabilidade de pertencer a cada categoria, frequentemente mais úteis que classificações rígidas [3][4] |

> ⚠️ **Nota Importante**: Embora a classificação seja frequentemente apresentada como uma tarefa de atribuição direta a classes, estimar as probabilidades de pertencer a cada classe é muitas vezes mais valioso em aplicações práticas [3][4].

### Visualização de Dados de Classificação

![image-20240807095543622](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240807095543622.png)

A visualização é crucial para entender dados de classificação [5]:

1. **Gráficos de Dispersão**: Úteis para visualizar a separação entre classes em diferentes variáveis [5].
2. **Box Plots**: Fornecem um resumo da distribuição dos dados para cada classe [5][6].

#### Box Plot Explicado
- Linha central: mediana
- Limites da caixa: 1º e 3º quartis
- Whiskers: Indicam a faixa de dados, geralmente 1.5 * IQR (Intervalo Interquartil) [6]

> 💡 **Dica**: Box plots, inventados por John Tukey, são ferramentas poderosas para visualização inicial de dados [7].

### Regressão Linear para Classificação

Embora a regressão linear possa ser usada para problemas de classificação binária, ela apresenta limitações [8][9]:

1. Codificação da resposta como 0/1
2. Classificação baseada em um limiar (geralmente 0.5)

**Vantagens**:
- Simplicidade
- Equivalência à análise discriminante linear para classificação binária [9]

**Desvantagens**:
- Pode produzir probabilidades fora do intervalo [0,1] [10]
- Não adequada para problemas multiclasse [11]

$$
P(Y=1|X) = X^T\beta
$$

Onde $X$ é o vetor de características e $\beta$ são os coeficientes da regressão.

#### Questões Técnicas
1. Por que a regressão linear pode produzir probabilidades fora do intervalo [0,1] em problemas de classificação?
2. Como a escolha da codificação (ex: 0/1 vs -1/+1) afeta a interpretação dos coeficientes na regressão linear para classificação?

### Regressão Logística

A regressão logística supera as limitações da regressão linear para classificação [10]:

1. Modela diretamente a probabilidade de pertencer a uma classe
2. Garante que as probabilidades estimadas estejam sempre no intervalo [0,1]

$$
P(Y=1|X) = \frac{1}{1 + e^{-(β_0 + β^T X)}}
$$

> ✔️ **Ponto de Destaque**: A regressão logística é especialmente adequada para estimar probabilidades de classe, crucial em muitas aplicações práticas [10].

### Classificação Multiclasse

Para problemas com mais de duas classes, métodos específicos são necessários [11]:

1. **Regressão Logística Multinomial**: Extensão da regressão logística para múltiplas classes
2. **Análise Discriminante**: Modela a distribuição dos preditores dentro de cada classe

> ❗ **Ponto de Atenção**: A codificação arbitrária de classes multiclasse (ex: 1, 2, 3) para uso em regressão linear pode implicar em uma ordenação indesejada entre as classes [11].

### Conclusão

A classificação é uma tarefa fundamental em aprendizado de máquina, com aplicações diversas desde detecção de spam até diagnósticos médicos. Enquanto métodos simples como regressão linear podem ser aplicados em certos casos, técnicas mais sofisticadas como regressão logística e análise discriminante são geralmente preferidas, especialmente para problemas multiclasse e quando a estimação de probabilidades é crucial.

### Questões Avançadas

1. Compare e contraste as abordagens de regressão logística e análise discriminante linear para classificação binária. Em que situações uma pode ser preferível à outra?

2. Discuta as implicações de usar regressão linear vs. regressão logística para estimar probabilidades de classe em um problema de classificação binária. Como isso afeta a interpretação e a qualidade das previsões?

3. Em um cenário de classificação multiclasse, proponha e justifique uma estratégia para combinar múltiplos classificadores binários (ex: one-vs-all, one-vs-one). Quais são as vantagens e desvantagens dessa abordagem em comparação com métodos nativamente multiclasse?

### Referências

[1] "In this section, we're going to talk about classification where the response variable has got two or more values." (Trecho do vídeo)

[2] "The classification task is to build a function that takes X as input and delivers one of the elements of the set C." (Trecho do vídeo)

[3] "Now, although classification problems are always couched in this form, we're often more interested in estimating the probabilities that X belongs to each category C." (Trecho do vídeo)

[4] "So estimating the probabilities is also key." (Trecho do vídeo)

[5] "Two variables-- this is the credit card default data set that we're going to use in this section. And the part on the left here is a scatter plot of balance against income." (Trecho do vídeo)

[6] "OK, well, a box plot, what's indicated there-- Trevor, you can point-- the black line is the median." (Trecho do vídeo)

[7] "John Tukey, one of the most famous statisticians-- he's no longer with us, but he's left a big legacy behind." (Trecho do vídeo)

[8] "OK, well, one question we can ask is, can we use linear regression to solve classification problems?" (Trecho do vídeo)

[9] "For a binary outcome, linear regression does a pretty good job and is equivalent to linear discriminant analysis." (Trecho do vídeo)

[10] "What we're going to see, however, is that linear regression might actually produce probabilities that could be less than 0, or even bigger than 1. And for this reason, we're going to introduce you to logistic regression, which is more appropriate." (Trecho do vídeo)

[11] "So when you have more than two categories, assigning numbers to the categories just arbitrarily seems a little dangerous, and especially if you're going to use it in linear regression." (Trecho do vídeo)