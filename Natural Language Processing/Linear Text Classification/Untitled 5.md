# Probabilistic vs. Discriminative Approaches: Distinguishing between probabilistic classifiers and discriminative classifiers

<imagem: Um diagrama comparativo mostrando Naïve Bayes (representado por uma rede bayesiana) de um lado e SVM/Perceptron (representado por um hiperplano separador) do outro, com setas apontando para suas características distintas>

## Introdução

Na classificação de texto e aprendizado de máquina em geral, duas abordagens fundamentais se destacam: os classificadores probabilísticos e os discriminativos. Essa distinção é crucial para entender as diferentes metodologias de aprendizado e suas implicações teóricas e práticas [1]. Este resumo se concentra em distinguir essas duas abordagens, focando especificamente no Naïve Bayes como exemplo de classificador probabilístico, e no perceptron e SVM (Support Vector Machine) como exemplos de classificadores discriminativos.

Os classificadores probabilísticos, como o Naïve Bayes, baseiam-se na modelagem da distribuição de probabilidade conjunta dos dados e rótulos. Por outro lado, os classificadores discriminativos, como o perceptron e o SVM, focam diretamente na tarefa de discriminação entre classes, sem modelar explicitamente a distribuição dos dados [2]. Essa diferença fundamental leva a distintas abordagens de aprendizado, representação de conhecimento e desempenho em várias tarefas de classificação.

## Conceitos Fundamentais

| Conceito                         | Explicação                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Classificador Probabilístico** | Modela a distribuição de probabilidade conjunta p(X,Y) dos dados X e rótulos Y. No caso do Naïve Bayes, isso é feito através da decomposição p(X,Y) = p(Y)p(X |
| **Classificador Discriminativo** | Foca diretamente na modelagem da probabilidade condicional p(Y |
| **Naïve Bayes**                  | Um classificador probabilístico que faz a suposição "ingênua" de independência condicional entre as características dado a classe. Isso simplifica significativamente o cálculo da verossimilhança [5]. |
| **Perceptron**                   | Um classificador linear discriminativo que aprende atualizando pesos iterativamente com base nos erros de classificação. É um dos algoritmos mais simples de aprendizado de máquina [6]. |
| **SVM (Support Vector Machine)** | Um classificador discriminativo que busca encontrar o hiperplano de margem máxima que separa as classes. É conhecido por sua capacidade de generalização e robustez [7]. |

> ⚠️ **Nota Importante**: A escolha entre abordagens probabilísticas e discriminativas pode ter um impacto significativo no desempenho do modelo, dependendo da natureza dos dados e da tarefa em questão [8].

### Modelagem Probabilística vs. Discriminativa

<imagem: Gráfico comparativo mostrando a função de decisão de um classificador Naïve Bayes (curvas de probabilidade) e um SVM (hiperplano) em um espaço bidimensional>

A distinção fundamental entre as abordagens probabilística e discriminativa reside na forma como elas modelam o problema de classificação [9].

#### Modelagem Probabilística (Naïve Bayes)

O Naïve Bayes, como classificador probabilístico, modela a distribuição conjunta p(X,Y) [10]. Para um problema de classificação binária, temos:

$$
p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}
$$

Onde:
- p(Y|X) é a probabilidade posterior
- p(X|Y) é a verossimilhança
- p(Y) é a probabilidade a priori
- p(X) é a evidência (normalizador)

A suposição "ingênua" de independência condicional do Naïve Bayes permite decompor a verossimilhança [11]:

$$
p(X|Y) = \prod_{j=1}^V p(X_j|Y)
$$

Onde V é o número de características.

#### Modelagem Discriminativa (Perceptron e SVM)

Os classificadores discriminativos, como o perceptron e o SVM, modelam diretamente a fronteira de decisão entre as classes [12]. Para um classificador linear, a função de decisão tem a forma:

$$
f(X) = \text{sign}(\theta \cdot X + b)
$$

Onde:
- θ é o vetor de pesos
- b é o viés
- X é o vetor de características

No caso do SVM, busca-se maximizar a margem entre as classes [13]:

$$
\text{max}_{\theta, b} \frac{2}{||\theta||} \text{ sujeito a } y_i(\theta \cdot x_i + b) \geq 1, \forall i
$$

#### Perguntas Teóricas

1. Derive a expressão para a atualização de pesos do perceptron e explique como ela difere conceitualmente da estimativa de máxima verossimilhança no Naïve Bayes.
2. Considerando um conjunto de dados linearmente separável, prove que o SVM de margem rígida sempre encontrará uma solução, enquanto o perceptron pode não convergir em um número finito de iterações.
3. Analise teoricamente como a suposição de independência condicional do Naïve Bayes afeta sua capacidade de modelar interações complexas entre características, em comparação com abordagens discriminativas.

### Estimação de Parâmetros

A estimação de parâmetros é um aspecto crucial que diferencia as abordagens probabilísticas e discriminativas [14].

#### Naïve Bayes

No Naïve Bayes, os parâmetros são estimados usando o princípio da máxima verossimilhança [15]. Para um vocabulário de V palavras e K classes, temos:

$$
\phi_{y,j} = \frac{\text{count}(y,j)}{\sum_{j'=1}^V \text{count}(y,j')} = \frac{\sum_{i:y^{(i)}=y} x_j^{(i)}}{\sum_{j'=1}^V \sum_{i:y^{(i)}=y} x_{j'}^{(i)}}
$$

Onde count(y,j) é a contagem da palavra j em documentos com rótulo y.

Para evitar problemas com palavras não vistas no treinamento, é comum usar suavização de Laplace [16]:

$$
\phi_{y,j} = \frac{\alpha + \text{count}(y,j)}{V\alpha + \sum_{j'=1}^V \text{count}(y,j')}
$$

Onde α é o hiperparâmetro de suavização.

#### Perceptron

O perceptron atualiza seus pesos de forma online, baseando-se nos erros de classificação [17]:

$$
\theta^{(t)} = \theta^{(t-1)} + f(x^{(i)}, y^{(i)}) - f(x^{(i)}, \hat{y})
$$

Onde $\hat{y}$ é a previsão do modelo e y^(i) é o rótulo verdadeiro.

#### SVM

O SVM resolve um problema de otimização quadrática para encontrar o hiperplano de margem máxima [18]:

$$
\min_{\theta, b} \frac{1}{2} ||\theta||^2 + C \sum_{i=1}^N \xi_i
$$

Sujeito a:
$$
y_i(\theta \cdot x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i
$$

Onde $\xi_i$ são variáveis de folga e C é um hiperparâmetro de regularização.

> ❗ **Ponto de Atenção**: A estimação de parâmetros em modelos discriminativos geralmente envolve otimização numérica, enquanto em modelos probabilísticos como Naïve Bayes, muitas vezes é possível obter estimativas de forma fechada [19].

### Funções de Perda e Otimização

As funções de perda e os métodos de otimização utilizados são fundamentalmente diferentes entre as abordagens probabilísticas e discriminativas [20].

#### Naïve Bayes

A função objetivo do Naïve Bayes é a log-verossimilhança [21]:

$$
\mathcal{L}(\phi, \mu) = \sum_{i=1}^N \log p_\text{mult}(x^{(i)}; \phi_{y(i)}) + \log p_\text{cat}(y^{(i)}; \mu)
$$

A otimização desta função leva às estimativas de máxima verossimilhança mencionadas anteriormente.

#### Perceptron

O perceptron minimiza uma aproximação da perda de dobradiça (hinge loss) [22]:

$$
\ell_\text{PERCEPTRON}(\theta; x^{(i)}, y^{(i)}) = \max_{\hat{y} \in Y} \theta \cdot f(x^{(i)}, \hat{y}) - \theta \cdot f(x^{(i)}, y^{(i)})
$$

#### SVM

O SVM minimiza uma combinação de perda de dobradiça e regularização L2 [23]:

$$
L_\text{SVM} = \frac{\lambda}{2} ||\theta||^2_2 + \sum_{i=1}^N (\max_{y \in Y} [\theta \cdot f(x^{(i)}, y) + c(y^{(i)}, y)] - \theta \cdot f(x^{(i)}, y^{(i)}))_+
$$

> ✔️ **Destaque**: A escolha da função de perda e do método de otimização tem implicações significativas na interpretabilidade do modelo, na velocidade de convergência e na capacidade de generalização [24].

### Vantagens e Desvantagens

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Naïve Bayes**: Simples, rápido de treinar, bom com dados esparsos [25] | **Naïve Bayes**: Suposição de independência pode ser irrealista, pode sofrer com o problema de probabilidade zero [26] |
| **Perceptron**: Simples de implementar, online e eficiente [27] | **Perceptron**: Pode não convergir para dados não linearmente separáveis, sensível à ordem dos dados [28] |
| **SVM**: Boa generalização, eficaz em espaços de alta dimensão [29] | **SVM**: Treinamento pode ser computacionalmente intensivo, escolha de kernel pode ser desafiadora [30] |

#### Perguntas Teóricas

1. Derive a expressão para o gradiente da função de perda do SVM e compare-a com o gradiente da log-verossimilhança do Naïve Bayes. Discuta as implicações dessas diferenças na otimização.
2. Analise teoricamente como a suposição de independência condicional do Naïve Bayes afeta sua capacidade de modelar interações complexas entre características, em comparação com SVM e perceptron.
3. Considerando um conjunto de dados não linearmente separável, prove que o SVM com kernel pode encontrar uma solução, enquanto o perceptron linear falhará. Discuta as implicações teóricas desta diferença.

### Implementação Avançada

Aqui está um exemplo avançado de implementação de um classificador SVM usando PyTorch, demonstrando como a abordagem discriminativa pode ser implementada de forma eficiente [31]:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def hinge_loss(outputs, labels):
    return torch.mean(torch.clamp(1 - outputs.t() * labels, min=0))

def train_svm(model, X, y, learning_rate=0.01, num_epochs=1000):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = hinge_loss(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Exemplo de uso
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0]], dtype=torch.float32)
y = torch.tensor([1, 1, -1, -1], dtype=torch.float32)

model = SVM(input_dim=2)
train_svm(model, X, y)
```

Este código implementa um SVM linear usando PyTorch, demonstrando como a otimização baseada em gradiente pode ser aplicada a um classificador discriminativo [32].

## Conclusão

A distinção entre classificadores probabilísticos e discriminativos é fundamental na teoria e prática do aprendizado de máquina. O Naïve Bayes, como representante da abordagem probabilística, oferece uma modelagem explícita da distribuição dos dados, permitindo inferências probabilísticas diretas. Por outro lado, classificadores discriminativos como o perceptron e o SVM focam na fronteira de decisão, muitas vezes alcançando melhor desempenho em tarefas de classificação pura [33].

A escolha entre estas abordagens depende de vários fatores, incluindo a natureza dos dados, o tamanho do conjunto de treinamento, a necessidade de interpretabilidade e os requisitos computacionais. Enquanto o Naïve Bayes pode ser mais adequado para conjuntos de dados pequenos ou quando estimativas de probabilidade são necessárias, SVM e perceptron podem oferecer melhor desempenho em tarefas de classificação de alta dimensionalidade ou quando a suposição de independência do Naïve Bayes é violada [34].

Compreender as diferenças teóricas e práticas entre estas abordagens é crucial para os cientistas de dados, permitindo a escolha informada de modelos e a interpretação adequada dos resultados em diversos cenários de aprendizado de máquina [35].

## Perguntas Teóricas Avançadas

1. Considere um problema de classificação binária com características X e rótulos Y. Derive a expressão para o erro de Bayes e compare-a com o limite inferior do erro de generalização do SVM de margem rígida. Discuta as implicações teóricas desta comparação.

2. Prove que, para um conjunto de dados linearmente separável, o algoritmo do perceptron converge em um número finito de iterações. Compare essa garantia de convergência com a do SVM e discuta as implicações práticas dessas diferenças teóricas.

3. Analise teoricamente como a suposição de independência condicional do Naïve Bayes afeta sua capacidade de modelar interações complexas entre características. Proponha e analise