# Pesos e Função de Características em Classificação de Texto Linear

<imagem: Um diagrama mostrando um vetor de pesos θ conectado a um vetor de características f(x,y), com setas indicando a interação entre palavras, rótulos e scores>

## Introdução

A classificação de texto linear é uma abordagem fundamental em aprendizado de máquina para processamento de linguagem natural. Neste contexto, dois conceitos cruciais emergem: os **pesos** (θ) e a **função de características** (f(x,y)). Estes elementos formam a base para a predição de rótulos a partir de representações bag-of-words de textos [1]. Este resumo explorará em profundidade como esses componentes são definidos, utilizados e otimizados em modelos de classificação de texto linear.

## Conceitos Fundamentais

| Conceito                               | Explicação                                                   |
| -------------------------------------- | ------------------------------------------------------------ |
| **Pesos (θ)**                          | ==Vetor coluna que atribui um score a cada palavra no vocabulário, medindo sua compatibilidade com um determinado rótulo.== Por exemplo, para o rótulo FICTION, "whale" pode ter um peso positivo, enquanto "molybdenum" pode ter um peso negativo [1]. |
| **Função de Características (f(x,y))** | ==Função que mapeia as contagens de palavras (x) e rótulos (y) para um vetor de características. Esta função produz uma representação vetorial que captura a relação entre as palavras presentes no texto e o rótulo potencial [1][2].== |
| **Score de Compatibilidade (Ψ(x,y))**  | ==Medida escalar da compatibilidade entre o bag-of-words x e o rótulo y==, calculada como o produto interno entre os pesos θ e a saída da função de características f(x,y) [1]. |

> ⚠️ **Nota Importante**: A definição precisa da função de características é crucial para o desempenho do classificador, pois determina como as informações do texto são representadas para a tarefa de classificação [2].

### Formulação Matemática

O score de compatibilidade Ψ(x,y) é definido matematicamente como:

$$
\Psi(x, y) = \theta \cdot f(x, y) = \sum_j \theta_j f_j(x, y)
$$

Onde:
- θ é o vetor de pesos
- f(x,y) é o vetor de características
- j indexa os elementos desses vetores [1]

Esta formulação permite que o classificador compute um score para cada possível rótulo y ∈ Y, dado um bag-of-words x.

> 💡 **Destaque**: A flexibilidade desta abordagem permite modelar uma variedade de tarefas de classificação, desde classificação binária até problemas multiclasse com K > 2 rótulos [1].

### Função de Características Detalhada

A função de características f(x,y) pode ser definida de várias formas. Uma abordagem comum é:

$$
f_j(x, y) = \begin{cases} 
x_\text{whale}, & \text{se } y = \text{FICTION} \\
0, & \text{caso contrário}
\end{cases}
$$

Esta definição retorna a contagem da palavra "whale" se o rótulo for FICTION, e zero caso contrário [2]. O índice j depende da posição de "whale" no vocabulário e de FICTION no conjunto de rótulos possíveis.

Para um problema de classificação com K rótulos, a saída da função de características pode ser formalizada como:

$$
f(x, y=1) = [x; 0; 0; \ldots; 0]_{(K-1) \times V}
$$

$$
f(x, y=2) = [0; 0; \ldots; 0; x; 0; 0; \ldots; 0]_{V \quad (K-2) \times V}
$$

$$
f(x, y=K) = [0; 0; \ldots; 0; x]_{(K-1) \times V}
$$

Onde V é o tamanho do vocabulário e K é o número de rótulos [3][4][5].

#### Perguntas Teóricas

1. Derive a expressão para o gradiente do score de compatibilidade Ψ(x,y) em relação aos pesos θ, considerando a definição da função de características apresentada.

2. Considerando um problema de classificação binária, demonstre matematicamente como a função de características f(x,y) pode ser simplificada em comparação com o caso multiclasse.

3. Analise teoricamente o impacto da esparsidade do vetor de características na eficiência computacional e na capacidade de generalização do modelo de classificação linear.

## Representação e Implementação

<imagem: Diagrama ilustrando a transformação de texto para bag-of-words e então para vetor de características, com pesos associados>

Na prática, tanto f quanto θ podem ser implementados como dicionários em vez de vetores, eliminando a necessidade de identificar explicitamente o índice j. Nessa implementação, a tupla (palavra, RÓTULO) atua como chave em ambos os dicionários [1].

```python
import torch

class LinearTextClassifier:
    def __init__(self, vocab_size, num_labels):
        self.weights = torch.randn(vocab_size, num_labels, requires_grad=True)
    
    def feature_function(self, x, y):
        # x: bag-of-words tensor
        # y: label index
        return torch.cat([x if i == y else torch.zeros_like(x) for i in range(self.weights.shape[1])])
    
    def compatibility_score(self, x, y):
        return torch.dot(self.weights.flatten(), self.feature_function(x, y))
    
    def predict(self, x):
        return torch.argmax(torch.tensor([self.compatibility_score(x, y) for y in range(self.weights.shape[1])]))

# Exemplo de uso
classifier = LinearTextClassifier(vocab_size=10000, num_labels=5)
x = torch.randint(0, 5, (10000,))  # Simulando um bag-of-words
predicted_label = classifier.predict(x)
```

Este código implementa um classificador de texto linear usando PyTorch, demonstrando como os conceitos de pesos e função de características podem ser aplicados na prática.

> ✔️ **Destaque**: A implementação eficiente da função de características e dos pesos é crucial para o desempenho computacional do classificador, especialmente em problemas com grandes vocabulários [1].

### Análise Teórica da Representação

A representação vetorial produzida pela função de características tem propriedades importantes:

1. **Esparsidade**: ==Para cada instância, apenas uma parte do vetor de características será não-zero, correspondendo às palavras presentes no documento e ao rótulo específico [3][4][5].==

2. **Dimensionalidade**: O vetor de características tem dimensão K × V, onde K é o número de rótulos e V é o tamanho do vocabulário [3][4][5].

3. **Informação Mútua**: A estrutura do vetor de características captura implicitamente a informação mútua entre palavras e rótulos [2].

A eficácia desta representação depende da capacidade de capturar padrões relevantes nos dados de treinamento, permitindo que o modelo aprenda a associar certas palavras ou combinações de palavras com rótulos específicos.

#### Perguntas Teóricas

1. Analise teoricamente como a dimensionalidade do vetor de características afeta o risco de overfitting no modelo de classificação linear. Como isso se relaciona com o conceito de "maldição da dimensionalidade"?

2. Derive uma expressão para a complexidade computacional do cálculo do score de compatibilidade em função do tamanho do vocabulário V e do número de rótulos K. Como essa complexidade se compara com outras abordagens de classificação de texto?

3. Considere uma modificação na função de características que incorpora n-gramas além de palavras individuais. Formalize matematicamente esta extensão e discuta seu impacto teórico na capacidade expressiva do modelo.

## Otimização dos Pesos

A otimização dos pesos θ é um aspecto crucial na construção de um classificador de texto linear eficaz. Diferentes abordagens podem ser utilizadas para esta otimização, cada uma com suas próprias características e implicações teóricas.

### Método do Gradiente

O método do gradiente é uma abordagem fundamental para otimizar os pesos. A atualização dos pesos é realizada iterativamente usando a regra:

$$
\theta^{(t+1)} \leftarrow \theta^{(t)} - \eta^{(t)} \nabla_\theta L
$$

Onde:
- $\theta^{(t)}$ é o vetor de pesos na iteração t
- $\eta^{(t)}$ é a taxa de aprendizado na iteração t
- $\nabla_\theta L$ é o gradiente da função de perda L em relação a θ [6]

> ❗ **Ponto de Atenção**: A escolha da taxa de aprendizado $\eta^{(t)}$ é crucial para a convergência do algoritmo. Uma taxa muito alta pode levar a oscilações ou divergência, enquanto uma taxa muito baixa pode resultar em convergência lenta [6].

### Gradiente Estocástico vs. Batch

O gradiente pode ser computado de duas formas principais:

1. **Gradiente Estocástico**: Atualiza os pesos usando um único exemplo de treinamento por vez.
2. **Gradiente em Batch**: Computa o gradiente sobre todo o conjunto de treinamento antes de atualizar os pesos.

A escolha entre estas abordagens afeta a velocidade de convergência e a estabilidade do treinamento [6][7].

```python
import torch
import torch.optim as optim

def train_classifier(classifier, train_data, num_epochs, learning_rate):
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for x, y_true in train_data:
            optimizer.zero_grad()
            y_pred = classifier(x)
            loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            loss.backward()
            optimizer.step()
```

Este código ilustra uma implementação básica de treinamento usando gradiente estocástico em PyTorch.

### Regularização

A regularização é uma técnica crucial para prevenir overfitting. A regularização L2 é comumente usada, adicionando um termo à função objetivo:

$$
L_\text{reg} = L + \frac{\lambda}{2} ||\theta||_2^2
$$

Onde $\lambda$ é o parâmetro de regularização [7].

> 💡 **Destaque**: A regularização L2 pode ser interpretada como impor uma distribuição prévia Gaussiana sobre os pesos, conectando a otimização com princípios bayesianos [7].

#### Perguntas Teóricas

1. Derive a expressão para o gradiente da função de perda regularizada $L_\text{reg}$ em relação aos pesos θ. Como a regularização afeta a direção do gradiente?

2. Analise teoricamente o compromisso entre viés e variância introduzido pela regularização L2. Como o parâmetro λ influencia este compromisso?

3. Considere um cenário onde algumas características são mais informativas que outras. Proponha e analise matematicamente uma estratégia de regularização que leve em conta esta heterogeneidade nas características.

## Conclusão

Os conceitos de pesos e função de características são fundamentais na classificação de texto linear. A função de características transforma os dados de texto em uma representação vetorial, enquanto os pesos determinam a importância de cada característica para a classificação. A interação entre estes elementos, através do score de compatibilidade, permite ao modelo fazer previsões sobre novos textos [1][2][3].

A otimização eficiente dos pesos, considerando aspectos como regularização e escolha apropriada de algoritmos de gradiente, é crucial para o desempenho do classificador [6][7]. Além disso, a flexibilidade na definição da função de características permite adaptar o modelo a diferentes tarefas de classificação de texto [2][3][4][5].

Compreender profundamente estes conceitos e suas implicações teóricas é essencial para desenvolver e aplicar eficazmente modelos de classificação de texto linear em problemas do mundo real.

## Perguntas Teóricas Avançadas

1. Considere um cenário de classificação multiclasse com K classes. Demonstre matematicamente que a formulação do classificador linear com K vetores de peso pode ser reduzida a uma formulação equivalente com K-1 vetores de peso sem perda de expressividade. Quais são as implicações teóricas e práticas desta redução?

2. Analise teoricamente o impacto da esparsidade do vetor de características na convergência do algoritmo de otimização. Como a taxa de convergência é afetada pela proporção de elementos não-zeros no vetor de características? Derive uma expressão para a complexidade de tempo esperada em função desta esparsidade.

3. Proponha e analise matematicamente uma extensão do modelo que incorpora informações de dependência entre palavras (por exemplo, n-gramas ou dependências sintáticas) na função de características. Como isso afeta a complexidade do modelo e sua capacidade de capturar padrões linguísticos mais sofisticados?

4. Derive a forma fechada da solução para os pesos ótimos em um classificador de texto linear com regularização L2, assumindo uma função de perda quadrática. Compare esta solução com a obtida através de métodos iterativos de otimização, discutindo prós e contras de cada abordagem.

5. Analise o comportamento assintótico do classificador de texto linear quando o tamanho do vocabulário V tende ao infinito, mantendo o número de exemplos de treinamento N fixo. Quais são as implicações teóricas para a consistência e a taxa de convergência do estimador? Como isso se relaciona com o fenômeno conhecido como "maldição da dimensionalidade"?

## Referências

[1] "Para prever um rótulo a partir de um bag-of-words, podemos atribuir um score a cada palavra no vocabulário, medindo a compatibilidade com o rótulo. Por exemplo, para o rótulo FICTION, podemos atribuir um score positivo à palavra whale, e um score negativo à palavra molybdenum. Esses scores são chamados de pesos, e são organizados em um vetor coluna θ." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[2] "Suponha que você queira um classificador multiclasse, onde K ≜ |Y| > 2. Por exemplo, você pode querer classificar notícias sobre esportes, celebridades, música e negócios. O objetivo é prever um rótulo y, dado o bag of words x, usando os pesos θ. Para cada rótulo y ∈ Y, computamos um score Ψ(x, y), que é uma medida escalar da compatibilidade entre o bag-of-words x e o rótulo y. Em um classificador linear bag-of-words, este score é o produto interno vetorial entre os pesos θ e a saída de uma função de características f(x, y)." *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[3] "f(x, y = 1) = [x; 0; 0; . . . ; 0]  [2.3]
              (K−1)×V" *(Trecho de CHAPTER 2. LINEAR TEXT CLASSIFICATION)*

[4] "f(x, y =