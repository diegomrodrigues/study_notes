## Unidades Ocultas e Função de Ativação em Redes Neurais

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816093324472.png" alt="image-20240816093324472" style="zoom:80%;" />

As unidades ocultas e funções de ativação são componentes fundamentais das redes neurais, desempenhando um papel crucial na capacidade do modelo de aprender representações complexas e não-lineares dos dados de entrada. Este resumo explora em profundidade esses conceitos, sua importância e implementações práticas.

### Conceitos Fundamentais

| Conceito               | Explicação                                                   |
| ---------------------- | ------------------------------------------------------------ |
| **Unidades Ocultas**   | São os nós intermediários em uma rede neural que computam características derivadas dos dados de entrada. Elas transformam a entrada em representações mais abstratas e de alto nível. [1] |
| **Função de Ativação** | Uma função não-linear aplicada à saída ponderada de cada unidade, introduzindo não-linearidade no modelo. A função sigmóide é uma escolha comum. [1] |
| **Camada Oculta**      | Uma camada de unidades ocultas entre as camadas de entrada e saída. Redes podem ter múltiplas camadas ocultas. [1] |

> ⚠️ **Nota Importante**: As unidades ocultas são chamadas assim porque seus valores não são diretamente observados nos dados de treinamento, diferentemente das unidades de entrada e saída. [1]

### Unidades Ocultas: O Coração da Rede Neural

As unidades ocultas são o componente central que permite às redes neurais aprender representações complexas e hierárquicas dos dados. Elas transformam as entradas em características de alto nível que são mais úteis para a tarefa em questão.

#### Funcionamento das Unidades Ocultas

Cada unidade oculta realiza as seguintes operações:

1. Recebe entradas ponderadas das unidades da camada anterior.
2. Calcula uma soma ponderada dessas entradas.
3. Aplica uma função de ativação não-linear ao resultado.

Matematicamente, podemos expressar isso como:

$$
z_m = \sigma(\alpha_{0m} + \alpha_m^T X)
$$

Onde:
- $z_m$ é a saída da m-ésima unidade oculta
- $\sigma$ é a função de ativação
- $\alpha_{0m}$ é o termo de viés
- $\alpha_m$ é o vetor de pesos
- $X$ é o vetor de entradas [1]

> ✔️ **Ponto de Destaque**: A capacidade das redes neurais de aproximar qualquer função contínua está diretamente relacionada ao uso de unidades ocultas com funções de ativação não-lineares. [1]

#### Questões Técnicas/Teóricas

1. Como o número de unidades ocultas afeta a capacidade de aprendizado de uma rede neural?
2. Explique por que as unidades ocultas são essenciais para a capacidade de uma rede neural aprender representações não-lineares.

### Função de Ativação: Introduzindo Não-Linearidade

A função de ativação é um componente crucial que introduz não-linearidade no modelo, permitindo que a rede neural aprenda relações complexas nos dados.

#### Função Sigmóide

A função sigmóide é uma escolha comum para a função de ativação em redes neurais. Ela é definida como:

$$
\sigma(v) = \frac{1}{1 + e^{-v}}
$$

Características da função sigmóide:
- Intervalo de saída entre 0 e 1
- Suave e diferenciável
- Saturação em valores extremos [1]

<image: Gráfico da função sigmóide, mostrando sua forma em S característica>

> ❗ **Ponto de Atenção**: A escolha da função de ativação pode afetar significativamente o desempenho e a facilidade de treinamento da rede neural. [1]

#### Outras Funções de Ativação

Embora a sigmóide seja comum, outras funções de ativação também são utilizadas:

| Função     | Equação                 | Características                              |
| ---------- | ----------------------- | -------------------------------------------- |
| ReLU       | $f(x) = \max(0, x)$     | Não saturante, eficiente computacionalmente  |
| Tanh       | $f(x) = \tanh(x)$       | Intervalo entre -1 e 1, centrada em zero     |
| Leaky ReLU | $f(x) = \max(0.01x, x)$ | Evita o problema do "neurônio morto" do ReLU |

#### Implementação em Python

Aqui está uma implementação concisa das funções de ativação mais comuns:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

#### Questões Técnicas/Teóricas

1. Quais são as vantagens e desvantagens da função sigmóide em comparação com ReLU?
2. Como a escolha da função de ativação pode afetar o problema do desaparecimento do gradiente em redes neurais profundas?

### Impacto das Unidades Ocultas e Funções de Ativação no Aprendizado

A combinação de unidades ocultas e funções de ativação não-lineares permite que as redes neurais aprendam representações hierárquicas complexas dos dados de entrada.

#### Aproximação Universal

O Teorema da Aproximação Universal estabelece que uma rede neural feedforward com uma única camada oculta contendo um número finito de neurônios pode aproximar qualquer função contínua em um subconjunto compacto de $\mathbb{R}^n$, desde que seja usada uma função de ativação adequada. [1]

> ✔️ **Ponto de Destaque**: A capacidade de aproximação universal das redes neurais é uma das razões fundamentais para seu sucesso em uma ampla gama de tarefas de aprendizado de máquina.

#### Profundidade vs. Largura

A escolha entre redes mais profundas (mais camadas) ou mais largas (mais unidades por camada) é um tópico de pesquisa ativo:

- Redes mais profundas podem aprender hierarquias mais complexas de características
- Redes mais largas podem capturar mais variações nos dados em cada nível de abstração

#### Gradientes e Treinamento

A escolha da função de ativação afeta significativamente o fluxo de gradientes durante o treinamento:

- Sigmóides podem levar ao problema do desaparecimento do gradiente em redes profundas
- ReLU ajuda a mitigar esse problema, permitindo gradientes não-nulos para entradas positivas

#### Implementação de uma Camada Oculta

Aqui está uma implementação simplificada de uma camada oculta em Python:

```python
import numpy as np

class HiddenLayer:
    def __init__(self, input_size, output_size, activation=np.tanh):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.activation = activation
    
    def forward(self, X):
        self.input = X
        self.output = self.activation(np.dot(X, self.weights) + self.bias)
        return self.output
    
    def backward(self, dY):
        dZ = dY * (1 - self.output**2)  # derivada do tanh
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        return dX, dW, db
```

### Conclusão

As unidades ocultas e funções de ativação são componentes essenciais das redes neurais, permitindo que elas aprendam representações complexas e não-lineares dos dados. A escolha adequada do número de unidades ocultas, da arquitetura da rede e das funções de ativação é crucial para o desempenho do modelo em tarefas de aprendizado de máquina. 

A compreensão profunda desses conceitos é fundamental para o desenvolvimento e otimização de modelos de aprendizado profundo eficazes.

### Questões Avançadas

1. Como você projetaria uma arquitetura de rede neural para capturar tanto características locais quanto globais em uma tarefa de processamento de imagem, considerando o papel das unidades ocultas e funções de ativação?

2. Discuta as implicações teóricas e práticas de usar funções de ativação diferentes em diferentes camadas de uma rede neural profunda. Como isso poderia afetar a capacidade de representação e o processo de treinamento?

3. Considerando o teorema da aproximação universal, proponha um experimento para investigar empiricamente quantas unidades ocultas são necessárias para aproximar uma função específica com um determinado nível de precisão. Como você abordaria este problema?

### Referências

[1] "As unidades no meio da rede, computando os recursos derivados Z
m
, são chamadas de unidades ocultas porque os valores Z
m 
não são diretamente observados. Em geral, pode haver mais de uma camada oculta. Podemos pensar nos Z
m 
como uma expansão de base dos dados de entrada originais; a rede neural é então um modelo linear padrão, ou modelo multilogit linear, usando essas transformações como entradas. No entanto, há um aprimoramento importante sobre as técnicas de expansão de base discutidas no Capítulo 5; aqui os parâmetros das funções de base são aprendidos com os dados." (Trecho de ESL II)