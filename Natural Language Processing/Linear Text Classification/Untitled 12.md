# Estimação de Parâmetros em Classificação de Texto Linear

<imagem: Um gráfico mostrando curvas de distribuição categórica e multinomial, com setas apontando para os parâmetros μ e φ sendo estimados a partir de um conjunto de dados de texto>

## Introdução

A estimação de parâmetros é um componente crucial na classificação de texto linear, especialmente quando lidamos com distribuições categóricas e multinomiais [1]. Este resumo se concentra na estimação dos parâmetros μ e φ dessas distribuições usando a estimativa de frequência relativa, que é equivalente à estimativa de máxima verossimilhança neste contexto [2]. Compreender esse processo é fundamental para implementar classificadores eficazes em tarefas de processamento de linguagem natural.

## Conceitos Fundamentais

| Conceito                                 | Explicação                                                   |
| ---------------------------------------- | ------------------------------------------------------------ |
| **Distribuição Categórica**              | Uma distribuição de probabilidade que descreve o resultado de um experimento aleatório com K categorias mutuamente exclusivas. Em classificação de texto, é frequentemente usada para modelar a distribuição de rótulos [3]. |
| **Distribuição Multinomial**             | Uma generalização da distribuição binomial para K > 2 categorias. No contexto de classificação de texto, modela a distribuição de contagens de palavras em um documento [4]. |
| **Estimativa de Máxima Verossimilhança** | Método de estimação de parâmetros que maximiza a probabilidade de observar os dados do conjunto de treinamento. Para distribuições categóricas e multinomiais, isso equivale à estimativa de frequência relativa [5]. |

> ⚠️ **Nota Importante**: A estimativa de máxima verossimilhança pode levar a problemas de overfitting, especialmente em conjuntos de dados pequenos ou esparsos. Técnicas de regularização, como smoothing, são frequentemente necessárias [6].

### Estimação de Parâmetros para Distribuição Categórica

<imagem: Um diagrama mostrando a contagem de ocorrências de cada categoria em um conjunto de dados, com setas apontando para as estimativas dos parâmetros correspondentes>

A distribuição categórica é fundamental para modelar a probabilidade de rótulos em classificação de texto. Seja Y o conjunto de rótulos possíveis, a estimação do parâmetro μ para esta distribuição é dada por [7]:

$$
\mu_y = \frac{\text{count}(y)}{\sum_{y' \in Y} \text{count}(y')} = \frac{\sum_{i=1}^N \delta(y^{(i)} = y)}{N}
$$

Onde:
- $\mu_y$ é a probabilidade estimada para o rótulo y
- count(y) é o número de ocorrências do rótulo y no conjunto de treinamento
- $\delta(y^{(i)} = y)$ é a função delta de Kronecker, que retorna 1 se $y^{(i)} = y$ e 0 caso contrário
- N é o número total de instâncias no conjunto de treinamento

Esta estimativa é intuitiva: a probabilidade de cada rótulo é simplesmente a proporção de vezes que ele aparece no conjunto de treinamento [8].

#### Perguntas Teóricas

1. Prove que a estimativa de máxima verossimilhança para μ na distribuição categórica é equivalente à estimativa de frequência relativa apresentada acima.
2. Como a estimativa de μ seria afetada se tivéssemos um conjunto de dados extremamente desbalanceado? Discuta as implicações teóricas e práticas.
3. Derive a expressão para o erro padrão da estimativa de μ_y e explique como isso poderia ser usado para construir intervalos de confiança para os parâmetros estimados.

### Estimação de Parâmetros para Distribuição Multinomial

<imagem: Um gráfico de barras mostrando as contagens de palavras em documentos de diferentes classes, com linhas pontilhadas indicando as estimativas de φ para cada classe>

Na classificação de texto usando o modelo Naive Bayes, a distribuição multinomial é usada para modelar a probabilidade de ocorrência de palavras dado um rótulo específico. A estimação do parâmetro φ para esta distribuição é dada por [9]:

$$
\phi_{y,j} = \frac{\text{count}(y, j)}{\sum_{j'=1}^V \text{count}(y, j')} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{\sum_{j'=1}^V \sum_{i:y^{(i)}=y} x_{j'}^{(i)}}
$$

Onde:
- $\phi_{y,j}$ é a probabilidade estimada da palavra j dado o rótulo y
- count(y, j) é o número de ocorrências da palavra j em documentos com rótulo y
- $x_j^{(i)}$ é a contagem da palavra j no documento i
- V é o tamanho do vocabulário

Esta estimativa representa a proporção de vezes que a palavra j aparece em documentos com o rótulo y, em relação ao total de palavras em documentos com esse rótulo [10].

> ❗ **Ponto de Atenção**: A estimativa de máxima verossimilhança para φ pode resultar em probabilidades zero para palavras que não aparecem em nenhum documento de uma determinada classe no conjunto de treinamento. Isso pode levar a problemas na classificação, pois uma única palavra ausente poderia zerar a probabilidade de uma classe inteira [11].

Para mitigar esse problema, é comum usar técnicas de smoothing, como o smoothing de Laplace [12]:

$$
\phi_{y,j} = \frac{\alpha + \text{count}(y, j)}{V\alpha + \sum_{j'=1}^V \text{count}(y, j')}
$$

Onde α é um hiperparâmetro que controla o grau de smoothing. Esta técnica adiciona um pseudocount α a todas as contagens, evitando probabilidades zero [13].

#### Perguntas Teóricas

1. Demonstre matematicamente por que a estimativa de máxima verossimilhança para φ na distribuição multinomial é equivalente à estimativa de frequência relativa apresentada.
2. Derive a expressão para o gradiente da log-verossimilhança com respeito a φ_y,j e mostre que o ponto onde este gradiente é zero corresponde à estimativa de frequência relativa.
3. Analise teoricamente o impacto do smoothing de Laplace na variância das estimativas de φ. Como isso afeta o trade-off entre viés e variância no modelo?

### Justificativa Teórica para Estimação de Máxima Verossimilhança

A estimação de máxima verossimilhança (MLE) para os parâmetros μ e φ pode ser justificada através da maximização da função de verossimilhança [14]. Para a distribuição multinomial, a log-verossimilhança é dada por:

$$
\mathcal{L}(\phi) = \sum_{i=1}^N \log p_{\text{mult}}(x^{(i)}; \phi_{y(i)}) = \sum_{i=1}^N \log B(x^{(i)}) + \sum_{j=1}^V x_j^{(i)} \log \phi_{y(i),j}
$$

Onde B(x^(i)) é o coeficiente multinomial, que é constante em relação a φ [15].

Para maximizar esta função sujeita à restrição $\sum_{j=1}^V \phi_{y,j} = 1$ para todo y, podemos usar multiplicadores de Lagrange. O Lagrangiano é dado por:

$$
\ell(\phi_y) = \sum_{i:y^{(i)}=y} \sum_{j=1}^V x_j^{(i)} \log \phi_{y,j} - \lambda(\sum_{j=1}^V \phi_{y,j} - 1)
$$

Diferenciando em relação a φ_y,j e igualando a zero, obtemos:

$$
\frac{\partial \ell(\phi_y)}{\partial \phi_{y,j}} = \sum_{i:y^{(i)}=y} x_j^{(i)} / \phi_{y,j} - \lambda = 0
$$

Resolvendo esta equação, chegamos à estimativa de frequência relativa para φ_y,j [16].

> ✔️ **Destaque**: A estimativa de máxima verossimilhança para distribuições categóricas e multinomiais tem a propriedade desejável de ser não-viesada e assintoticamente eficiente, ou seja, atinge o limite inferior de Cramér-Rao à medida que o tamanho da amostra aumenta [17].

#### Perguntas Teóricas

1. Derive a matriz de informação de Fisher para a distribuição multinomial e use-a para calcular o limite inferior de Cramér-Rao para as estimativas de φ.
2. Prove que a estimativa de máxima verossimilhança para φ é consistente, ou seja, converge em probabilidade para o verdadeiro valor do parâmetro à medida que o tamanho da amostra aumenta.
3. Analise teoricamente como a adição de regularização L2 na log-verossimilhança afetaria as estimativas de φ. Derive as novas equações para as estimativas regularizadas.

## Implementação e Considerações Práticas

Na prática, a implementação da estimação de parâmetros para classificação de texto linear pode ser realizada de forma eficiente usando bibliotecas de processamento de linguagem natural. Aqui está um exemplo avançado usando PyTorch para implementar um classificador Naive Bayes multinomial com smoothing de Laplace [18]:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultinomialNaiveBayes(nn.Module):
    def __init__(self, num_classes, vocab_size, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.alpha = alpha
        
        # Inicializa log(φ) e log(μ)
        self.log_phi = nn.Parameter(torch.randn(num_classes, vocab_size))
        self.log_mu = nn.Parameter(torch.randn(num_classes))
    
    def forward(self, x):
        # x: (batch_size, vocab_size)
        return (x @ self.log_phi.T) + self.log_mu
    
    def fit(self, X, y):
        # X: (num_samples, vocab_size), y: (num_samples,)
        counts = torch.zeros(self.num_classes, self.vocab_size)
        for c in range(self.num_classes):
            counts[c] = X[y == c].sum(dim=0)
        
        # Aplica smoothing de Laplace e calcula log(φ)
        smoothed_counts = counts + self.alpha
        log_phi = torch.log(smoothed_counts) - torch.log(smoothed_counts.sum(dim=1, keepdim=True))
        
        # Calcula log(μ)
        class_counts = torch.bincount(y, minlength=self.num_classes)
        log_mu = torch.log(class_counts + self.alpha) - torch.log(len(y) + self.alpha * self.num_classes)
        
        # Atualiza os parâmetros
        self.log_phi.data = log_phi
        self.log_mu.data = log_mu

# Uso
model = MultinomialNaiveBayes(num_classes=3, vocab_size=10000)
X_train = torch.randint(0, 5, (1000, 10000))  # Dados de exemplo
y_train = torch.randint(0, 3, (1000,))  # Rótulos de exemplo
model.fit(X_train, y_train)

# Previsão
X_test = torch.randint(0, 5, (100, 10000))
with torch.no_grad():
    log_probs = model(X_test)
    predictions = log_probs.argmax(dim=1)
```

Este código implementa um classificador Naive Bayes multinomial usando PyTorch, permitindo a estimação eficiente dos parâmetros μ e φ com smoothing de Laplace [19]. A classe `MultinomialNaiveBayes` herda de `nn.Module`, permitindo que seja facilmente integrada em pipelines de deep learning mais complexos, se necessário [20].

> 💡 **Dica**: Trabalhar no espaço logarítmico (log(φ) e log(μ)) melhora a estabilidade numérica e evita underflow em cálculos com probabilidades muito pequenas [21].

## Conclusão

A estimação de parâmetros para distribuições categóricas e multinomiais é fundamental na classificação de texto linear, especialmente no contexto do classificador Naive Bayes [22]. A estimativa de frequência relativa, equivalente à estimativa de máxima verossimilhança neste caso, fornece uma base teórica sólida para a estimação dos parâmetros μ e φ [23]. 

No entanto, é crucial considerar técnicas de regularização, como o smoothing de Laplace, para lidar com problemas de esparsidade e evitar probabilidades zero [24]. A implementação prática desses conceitos, como demonstrado no exemplo em PyTorch, permite a criação de classificadores eficientes e robustos para tarefas de processamento de linguagem natural [25].

À medida que avançamos para modelos mais complexos, como redes neurais profundas, os princípios fundamentais de estimação de parâmetros continuam relevantes, formando a base para compreender e melhorar algoritmos mais avançados de aprendizado de máquina em processamento de texto [26].

## Perguntas Teóricas Avançadas

1. Derive a expressão para a informação mútua entre as features e os rótulos em um classificador Naive Bayes multinomial. Como essa medida poderia ser usada para selecionar features de forma teoricamente fundamentada?

2. Considere um cenário onde os documentos têm comprimentos muito variados. Proponha e analise teoricamente uma modificação na estimativa de φ que leve em conta essa variação, discutindo suas propriedades estatísticas.

3. Compare teoricamente a variância das estimativas de φ obtidas por máxima verossimilhança com as obtidas por inferência bayesiana usando uma prior Dirichlet. Em que condições a abordagem bayesiana seria preferível?

4. Prove que, para um conjunto de dados fixo, à medida que o parâmetro de smoothing α aumenta, as estimativas de φ para todas as classes convergem para uma distribuição uniforme. Discuta as implicações desse resultado para a escolha de α.

5. Desenvolva uma prova formal de que, para um classificador Naive Bayes multinomial com smoothing de Laplace, a decisão de classificação é equivalente a encontrar o hiperplano separador em um espaço de características transformado. Relacione este resultado com o conceito de kernel trick em SVMs.

## Referências

[1] "To predict a label from a bag-of-words, we can assign a score to each word in the vocabulary, measuring the compatibility with the label." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[2] "Maximum likelihood estimation chooses φ to maximize the log-likelihood L." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[3] "The categorical distribution involves a product over words, with each term in the product equal to the probability φj, exponentiated by the count x