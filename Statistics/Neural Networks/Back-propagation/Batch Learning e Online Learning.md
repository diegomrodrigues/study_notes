## Batch Learning e Online Learning em Redes Neurais

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816101829816.png" alt="image-20240816101829816" style="zoom:80%;" />

O treinamento de redes neurais é um processo fundamental na aprendizagem de máquina, e duas abordagens principais se destacam: batch learning (aprendizagem em lote) e online learning (aprendizagem online). Este resumo explorará em profundidade essas duas técnicas, suas implementações no contexto do algoritmo de retropropagação, vantagens, desvantagens e aplicações práticas [1].

### Conceitos Fundamentais

| Conceito            | Explicação                                                   |
| ------------------- | ------------------------------------------------------------ |
| **Batch Learning**  | Método de treinamento onde os parâmetros da rede são atualizados após processar todo o conjunto de dados de treinamento [1]. |
| **Online Learning** | Técnica de treinamento onde os parâmetros são atualizados após processar cada exemplo individual do conjunto de dados [1]. |
| **Retropropagação** | Algoritmo fundamental para o treinamento de redes neurais, que calcula o gradiente da função de erro em relação aos pesos da rede [2]. |

> ✔️ **Ponto de Destaque**: A escolha entre batch learning e online learning pode impactar significativamente a velocidade de convergência e a qualidade do modelo final.

### Batch Learning

![image-20240816102424437](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816102424437.png)

O batch learning, também conhecido como aprendizagem em lote, é uma abordagem onde a rede neural processa todo o conjunto de dados de treinamento antes de realizar uma atualização dos pesos [1]. 

#### Implementação Matemática

Para um conjunto de treinamento com N observações, a atualização dos pesos no batch learning é dada por:

$$
\beta_{km}^{(r+1)} = \beta_{km}^{(r)} - \gamma_r \sum_{i=1}^N \frac{\partial R_i}{\partial \beta_{km}^{(r)}}
$$

$$
\alpha_{ml}^{(r+1)} = \alpha_{ml}^{(r)} - \gamma_r \sum_{i=1}^N \frac{\partial R_i}{\partial \alpha_{ml}^{(r)}}
$$

Onde:
- $\beta_{km}$ e $\alpha_{ml}$ são os pesos da rede
- $r$ é o número da iteração (época)
- $\gamma_r$ é a taxa de aprendizagem
- $R_i$ é a função de erro para a i-ésima observação [3]

#### 👍Vantagens
* Estimativas de gradiente mais estáveis
* Potencial para paralelização eficiente

#### 👎Desvantagens
* Requer mais memória para armazenar todo o conjunto de dados
* Pode ser lento para conjuntos de dados muito grandes

#### [Questões Técnicas/Teóricas]

1. Como o tamanho do lote afeta a convergência do algoritmo de batch learning?
2. Em que situações o batch learning pode ser preferível ao online learning?

### Online Learning

O online learning atualiza os pesos da rede após processar cada exemplo individual do conjunto de dados [1]. Esta abordagem é particularmente útil para conjuntos de dados muito grandes ou streams de dados contínuos.

#### Implementação Matemática

No online learning, as atualizações de peso ocorrem para cada observação:

$$
\beta_{km}^{(r+1)} = \beta_{km}^{(r)} - \gamma_r \frac{\partial R_i}{\partial \beta_{km}^{(r)}}
$$

$$
\alpha_{ml}^{(r+1)} = \alpha_{ml}^{(r)} - \gamma_r \frac{\partial R_i}{\partial \alpha_{ml}^{(r)}}
$$

Onde $i$ representa o índice da observação atual [3].

> ⚠️ **Nota Importante**: A taxa de aprendizagem $\gamma_r$ no online learning deve geralmente diminuir com o tempo para garantir a convergência.

#### Taxa de Aprendizagem Adaptativa

Para garantir a convergência no online learning, a taxa de aprendizagem deve satisfazer as seguintes condições:

1. $\gamma_r \rightarrow 0$ conforme $r \rightarrow \infty$
2. $\sum_r \gamma_r = \infty$
3. $\sum_r \gamma_r^2 < \infty$

Uma escolha comum que satisfaz essas condições é $\gamma_r = \frac{1}{r}$ [4].

#### 👍Vantagens
* Adaptação rápida a novos dados
* Menor requisito de memória

#### 👎Desvantagens
* Atualizações de peso mais ruidosas
* Pode ser mais suscetível a mínimos locais

#### [Questões Técnicas/Teóricas]

1. Como você implementaria uma estratégia de decaimento da taxa de aprendizagem em online learning?
2. Quais são as implicações do online learning para a estacionariedade dos dados?

### Comparação entre Batch e Online Learning

| Aspecto                         | Batch Learning | Online Learning |
| ------------------------------- | -------------- | --------------- |
| Uso de Memória                  | Alto           | Baixo           |
| Velocidade de Adaptação         | Lenta          | Rápida          |
| Estabilidade do Gradiente       | Alta           | Baixa           |
| Adequação para Grandes Datasets | Menor          | Maior           |

### Implementação Prática

Aqui está um exemplo simplificado de como implementar batch e online learning em Python usando NumPy:

````python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
    
    def forward(self, X):
        self.z = np.dot(X, self.weights1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.weights2)
        return self.sigmoid(self.z3)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.z2_error = self.output_delta.dot(self.weights2.T)
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.z2)
        
        self.weights2 += self.z2.T.dot(self.output_delta)
        self.weights1 += X.T.dot(self.z2_delta)
    
    def train_batch(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def train_online(self, X, y, epochs):
        for _ in range(epochs):
            for i in range(X.shape[0]):
                output = self.forward(X[i:i+1])
                self.backward(X[i:i+1], y[i:i+1], output)
````

Este exemplo demonstra a implementação básica de uma rede neural com uma camada oculta, utilizando tanto batch learning quanto online learning [5].

### Conclusão

Batch learning e online learning são duas abordagens fundamentais para o treinamento de redes neurais, cada uma com suas próprias vantagens e desvantagens. A escolha entre elas depende de fatores como o tamanho do conjunto de dados, requisitos de memória, velocidade de adaptação necessária e a natureza do problema em questão. Compreender as nuances de cada método é crucial para desenvolver modelos de aprendizagem de máquina eficientes e eficazes [1][3][4].

### Questões Avançadas

1. Como você projetaria um sistema híbrido que combina batch e online learning para otimizar o treinamento em um cenário de streaming de dados?

2. Discuta as implicações teóricas e práticas de usar mini-batch learning como um meio-termo entre batch e online learning. Como isso afeta a convergência e a eficiência computacional?

3. Em um cenário de aprendizado por reforço, como você adaptaria os princípios de online learning para lidar com a natureza sequencial e potencialmente não-estacionária do ambiente?

### Referências

[1] "The updates in (11.13) are a kind of batch learning, with the parameter updates being a sum over all of the training cases. Learning can also be carried out online—processing each observation one at a time, updating the gradient after each training case, and cycling through the training cases many times." (Trecho de ESL II)

[2] "The generic approach to minimizing R(θ) is by gradient descent, called back-propagation in this setting. Because of the compositional form of the model, the gradient can be easily derived using the chain rule for differentiation." (Trecho de ESL II)

[3] "Given these derivatives, a gradient descent update at the (r + 1)st iteration has the form
β(r+1)km = β(r)km − γr ∑Ni=1 ∂Ri/∂β(r)km,
α(r+1)ml = α(r)ml − γr ∑Ni=1 ∂Ri/∂α(r)ml,
where γr is the learning rate, discussed below." (Trecho de ESL II)

[4] "With online learning γr should decrease to zero as the iteration r → ∞. This learning is a form of stochastic approximation (Robbins and Munro, 1951); results in this field ensure convergence if γr → 0, ∑r γr = ∞, and ∑r γ2r < ∞ (satisfied, for example, by γr = 1/r)." (Trecho de ESL II)

[5] "Back-propagation can be very slow, and for that reason is usually not the method of choice. Second-order techniques such as Newton's method are not attractive here, because the second derivative matrix of R (the Hessian) can be very large." (Trecho de ESL II)