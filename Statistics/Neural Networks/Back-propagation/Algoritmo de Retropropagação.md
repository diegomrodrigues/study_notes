## Algoritmo de Retropropagação (Back-propagation)

![image-20240816100749672](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240816100749672.png)

O algoritmo de retropropagação é uma técnica fundamental para o treinamento de redes neurais artificiais, permitindo a otimização eficiente dos pesos da rede através do cálculo do gradiente da função de erro em relação aos parâmetros do modelo.

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Função de Erro**        | Medida quantitativa da diferença entre as saídas previstas pela rede e os valores reais desejados. Comumente utiliza-se o erro quadrático médio. [9] |
| **Gradiente Descendente** | Método de otimização que ajusta iterativamente os parâmetros na direção oposta ao gradiente da função de erro, buscando minimizá-la. [9] |
| **Regra da Cadeia**       | Princípio matemático que permite calcular as derivadas parciais de funções compostas, essencial para a propagação do erro através das camadas da rede. [12] |

> ⚠️ **Nota Importante**: A eficácia do algoritmo de retropropagação depende crucialmente da escolha adequada da taxa de aprendizado e da inicialização dos pesos da rede.

### Funcionamento do Algoritmo de Retropropagação

O algoritmo de retropropagação opera em duas fases principais: a propagação direta (forward pass) e a retropropagação (backward pass) [9].

1. **Propagação Direta**:
   - As entradas são propagadas através da rede, camada por camada.
   - Cada neurônio calcula sua ativação baseada nas entradas ponderadas e na função de ativação.
   - A saída final da rede é comparada com o valor desejado para calcular o erro.

2. **Retropropagação**:
   - O erro é propagado de volta através da rede.
   - Os gradientes são calculados para cada parâmetro usando a regra da cadeia.
   - Os pesos são atualizados na direção oposta ao gradiente.

#### Formalização Matemática

Para uma rede neural com uma camada oculta, podemos expressar o processo matematicamente [12]:

Seja $z_m = \sigma(\alpha_{0m} + \alpha_m^T X)$ a ativação da m-ésima unidade oculta, onde $\sigma$ é a função de ativação (comumente sigmóide). A saída da rede para a k-ésima classe é dada por:

$$
f_k(X) = g_k(\beta_{0k} + \beta_k^T Z)
$$

onde $Z = (z_1, z_2, ..., z_M)$ e $g_k$ é a função de ativação da camada de saída.

O erro quadrático para uma única observação é:

$$
R_i = \sum_{k=1}^K (y_{ik} - f_k(x_i))^2
$$

As derivadas parciais em relação aos parâmetros são [12]:

$$
\frac{\partial R_i}{\partial \beta_{km}} = -2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i)z_{mi}
$$

$$
\frac{\partial R_i}{\partial \alpha_{m\ell}} = -\sum_{k=1}^K 2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i)\beta_{km}\sigma'(\alpha_m^T x_i)x_{i\ell}
$$

> ✔️ **Ponto de Destaque**: A eficiência computacional do algoritmo de retropropagação vem da sua capacidade de reutilizar cálculos intermediários, reduzindo significativamente o número de operações necessárias.

### 1. Estrutura da Rede

Vamos começar definindo a estrutura básica da rede neural:

- Camada de entrada: $X = (x_1, x_2, ..., x_p)$, onde $p$ é o número de características de entrada.
- Camada oculta: $M$ unidades ocultas.
- Camada de saída: $K$ unidades de saída (uma para cada classe em problemas de classificação).

### 2. Propagação Forward

#### 2.1 Cálculo das Ativações da Camada Oculta

Para cada unidade oculta $m = 1, 2, ..., M$, calculamos sua ativação $z_m$ da seguinte forma [12]:

$$ z_m = \sigma(\alpha_{0m} + \alpha_m^T X) $$

Onde:
- $\alpha_{0m}$ é o termo de viés (bias) para a m-ésima unidade oculta.
- $\alpha_m = (\alpha_{m1}, \alpha_{m2}, ..., \alpha_{mp})$ é o vetor de pesos conectando a entrada à m-ésima unidade oculta.
- $\sigma$ é a função de ativação, geralmente a função sigmóide: $\sigma(v) = \frac{1}{1 + e^{-v}}$

> ✔️ **Ponto de Destaque**: A função sigmóide introduz não-linearidade na rede, permitindo que ela aprenda relações complexas nos dados.

#### 2.2 Cálculo das Saídas

Para cada unidade de saída $k = 1, 2, ..., K$, calculamos sua saída $f_k(X)$ como [12]:

$$ f_k(X) = g_k(\beta_{0k} + \beta_k^T Z) $$

Onde:
- $Z = (z_1, z_2, ..., z_M)$ é o vetor de ativações da camada oculta.
- $\beta_{0k}$ é o termo de viés para a k-ésima unidade de saída.
- $\beta_k = (\beta_{k1}, \beta_{k2}, ..., \beta_{kM})$ é o vetor de pesos conectando a camada oculta à k-ésima unidade de saída.
- $g_k$ é a função de ativação da camada de saída, que pode variar dependendo do problema:
  - Para regressão: $g_k(t) = t$ (função identidade)
  - Para classificação binária: $g_k(t) = \sigma(t)$ (sigmóide)
  - Para classificação multiclasse: $g_k(t) = \frac{e^t}{\sum_{j=1}^K e^{t_j}}$ (softmax)

### 3. Cálculo do Erro

Para uma única observação $i$, o erro quadrático é definido como [12]:

$$ R_i = \sum_{k=1}^K (y_{ik} - f_k(x_i))^2 $$

Onde $y_{ik}$ é o valor real da k-ésima saída para a i-ésima observação.

### 4. Retropropagação (Backpropagation)

O objetivo do treinamento é minimizar o erro total $R = \sum_{i=1}^N R_i$ ajustando os pesos da rede. Isso é feito através do algoritmo de retropropagação, que calcula os gradientes do erro em relação aos pesos.

#### 4.1 Gradientes para a Camada de Saída

Para cada peso $\beta_{km}$ conectando a m-ésima unidade oculta à k-ésima unidade de saída [12]:

$$ \frac{\partial R_i}{\partial \beta_{km}} = -2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i)z_{mi} $$

#### 4.2 Gradientes para a Camada Oculta

Para cada peso $\alpha_{m\ell}$ conectando a $\ell$-ésima entrada à m-ésima unidade oculta [12]:

$$ \frac{\partial R_i}{\partial \alpha_{m\ell}} = -\sum_{k=1}^K 2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i)\beta_{km}\sigma'(\alpha_m^T x_i)x_{i\ell} $$

### 5. Atualização dos Pesos

Os pesos são atualizados iterativamente usando o gradiente descendente:

$$ \beta_{km}^{(new)} = \beta_{km}^{(old)} - \eta \sum_{i=1}^N \frac{\partial R_i}{\partial \beta_{km}} $$

$$ \alpha_{m\ell}^{(new)} = \alpha_{m\ell}^{(old)} - \eta \sum_{i=1}^N \frac{\partial R_i}{\partial \alpha_{m\ell}} $$

Onde $\eta$ é a taxa de aprendizagem.

> ⚠️ **Nota Importante**: A escolha da taxa de aprendizagem é crucial. Uma taxa muito alta pode levar à divergência, enquanto uma taxa muito baixa pode resultar em convergência lenta.

### 6. Eficiência Computacional

A eficiência do algoritmo de retropropagação vem da sua capacidade de reutilizar cálculos intermediários. Definimos:

$$ \delta_{ki} = -2(y_{ik} - f_k(x_i))g'_k(\beta_k^T z_i) $$
$$ s_{mi} = \sigma'(\alpha_m^T x_i)\sum_{k=1}^K \beta_{km}\delta_{ki} $$

Assim, podemos reescrever os gradientes como:

$$ \frac{\partial R_i}{\partial \beta_{km}} = \delta_{ki}z_{mi} $$
$$ \frac{\partial R_i}{\partial \alpha_{m\ell}} = s_{mi}x_{i\ell} $$

Isso permite que o algoritmo calcule os gradientes em duas passadas pela rede: uma forward para computar as ativações e outra backward para computar os deltas e atualizar os pesos.

✔️ **Ponto de Destaque**: A reutilização de cálculos intermediários reduz a complexidade computacional de $O(W^2)$ para $O(W)$, onde $W$ é o número total de pesos na rede.

### Questões Técnicas

1. Como a escolha da função de ativação na camada oculta afeta a capacidade da rede neural de aprender relações não-lineares nos dados?

2. Explique por que o algoritmo de retropropagação é computacionalmente eficiente para treinar redes neurais. Como ele difere de um cálculo direto dos gradientes?

3. Considerando uma rede neural com 100 entradas, 50 unidades ocultas e 10 saídas, quantos parâmetros (pesos e vieses) precisam ser aprendidos durante o treinamento?

### Implementação do Algoritmo

A implementação do algoritmo de retropropagação pode ser resumida nos seguintes passos [13]:

1. Inicialização dos pesos com valores aleatórios próximos a zero.
2. Para cada época de treinamento:
   a. Propagação direta dos dados de entrada.
   b. Cálculo do erro na camada de saída.
   c. Retropropagação do erro e cálculo dos gradientes.
   d. Atualização dos pesos usando o gradiente descendente.

Exemplo simplificado em Python:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4) 
        self.weights2 = np.random.rand(4, 1)                 
        self.y = y
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T,  np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2
```

Este exemplo demonstra uma implementação básica do algoritmo de retropropagação para uma rede neural com uma camada oculta.

#### Questões Técnicas/Teóricas

1. Como o algoritmo de retropropagação lida com o problema do desvanecimento do gradiente em redes neurais profundas?
2. Explique como a escolha da função de ativação pode impactar a eficácia do algoritmo de retropropagação.

### Variantes e Otimizações

O algoritmo de retropropagação básico pode ser estendido e otimizado de várias formas [13]:

1. **Momentum**: Adiciona um termo de momento para acelerar a convergência e evitar mínimos locais.
2. **Aprendizado em Lote vs. Online**: O aprendizado em lote atualiza os pesos após processar todo o conjunto de treinamento, enquanto o online atualiza após cada amostra.
3. **Taxa de Aprendizado Adaptativa**: Ajusta a taxa de aprendizado durante o treinamento para melhorar a convergência.

> ❗ **Ponto de Atenção**: A escolha entre aprendizado em lote e online pode afetar significativamente a velocidade de convergência e a qualidade da solução final.

### Aplicações e Limitações

O algoritmo de retropropagação é amplamente utilizado em diversos domínios, incluindo:

- Reconhecimento de padrões
- Processamento de linguagem natural
- Visão computacional

No entanto, também apresenta limitações:

👍 **Vantagens**:
- Eficiência computacional para redes feed-forward
- Capacidade de aprender representações complexas

👎 **Desvantagens**:
- Suscetibilidade a mínimos locais
- Dificuldade em treinar redes muito profundas (problema do desvanecimento/explosão do gradiente)

### Conclusão

O algoritmo de retropropagação revolucionou o treinamento de redes neurais artificiais, proporcionando um método eficiente para ajustar os pesos da rede e minimizar o erro de previsão [9]. Sua capacidade de lidar com funções não-lineares complexas e sua eficiência computacional o tornaram a base para muitos avanços em aprendizado profundo. No entanto, desafios como o desvanecimento do gradiente em redes profundas levaram ao desenvolvimento de técnicas mais avançadas, como redes residuais e normalização em lote, que complementam e estendem as capacidades do algoritmo de retropropagação básico [13].

### Questões Avançadas

1. Compare o desempenho e a eficácia do algoritmo de retropropagação com métodos de otimização de segunda ordem, como o algoritmo de Levenberg-Marquardt, para o treinamento de redes neurais profundas.

2. Discuta como as técnicas de regularização, como dropout e normalização em lote, interagem com o processo de retropropagação e afetam a convergência do treinamento.

3. Explique como o algoritmo de retropropagação poderia ser modificado para treinar redes neurais recorrentes, considerando a natureza temporal das dependências em sequências de dados.

### Referências

[9] "Typically we don't want the global minimizer of R(θ), as this is likely to be an overfit solution. Instead some regularization is needed: this is achieved directly through a penalty term, or indirectly by early stopping." (Trecho de ESL II)

[12] "Here is back-propagation in detail for squared error loss. Let z_mi = σ(α_0m + α_m^T x_i), from (11.5) and let z_i = (z_1i, z_2i, ..., z_Mi). Then we have R(θ) ≡ ΣRi = Σ(y_ik - f_k(x_i))^2, with derivatives ∂R_i/∂β_km = -2(y_ik - f_k(x_i))g'_k(β_k^T z_i)z_mi, ∂R_i/∂α_mℓ = -Σ2(y_ik - f_k(x_i))g'_k(β_k^T z_i)β_km σ'(α_m^T x_i)x_iℓ." (Trecho de ESL II)

[13] "Given these derivatives, a gradient descent update at the (r + 1)st iteration has the form β_km^(r+1) = β_km^(r) - γ_r Σ ∂R_i/∂β_km^(r), α_mℓ^(r+1) = α_mℓ^(r) - γ_r Σ ∂R_i/∂α_mℓ^(r), where γ_r is the learning rate, discussed below." (Trecho de ESL II)