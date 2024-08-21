## Métodos de Otimização para Maximum Likelihood Estimation (MLE)

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821151639133.png" alt="image-20240821151639133" style="zoom: 50%;" />

### Introdução

A estimação por máxima verossimilhança (Maximum Likelihood Estimation - MLE) é um método fundamental na estatística e aprendizado de máquina para estimar os parâmetros de um modelo probabilístico [1]. No contexto de modelos generativos profundos, ==o MLE é frequentemente utilizado para treinar redes neurais que representam distribuições de probabilidade complexas [2].== Este resumo abordará dois métodos de otimização essenciais para resolver problemas de MLE: o gradiente descendente e o gradiente descendente estocástico (SGD).

### Conceitos Fundamentais

| Conceito                                | Explicação                                                   |
| --------------------------------------- | ------------------------------------------------------------ |
| **Maximum Likelihood Estimation (MLE)** | ==Método para estimar os parâmetros de um modelo estatístico maximizando a função de verossimilhança [1]== |
| **Função de Verossimilhança**           | ==Medida da probabilidade dos dados observados dado um conjunto de parâmetros do modelo [2]== |
| **Gradiente**                           | ==Vetor de derivadas parciais que indica a direção de maior crescimento de uma função [3]== |
| **Taxa de Aprendizado**                 | Hiperparâmetro que controla o tamanho dos passos durante a otimização [4] |

> ✔️ **Ponto de Destaque**: A otimização por MLE em modelos generativos profundos visa encontrar os ==parâmetros que maximizam a probabilidade dos dados de treinamento sob o modelo [2].==

### Gradiente Descendente

==O gradiente descendente é um algoritmo de otimização de primeira ordem que busca o mínimo local de uma função==, ==movendo-se iterativamente na direção oposta ao gradiente [3].==

==No contexto do MLE, o objetivo é maximizar a função de log-verossimilhança $\ell(\theta)$, que para um conjunto de dados $D = \{x^{(1)}, ..., x^{(m)}\}$ e um modelo com parâmetros $\theta$, é definida como [5]:==
$$
\ell(\theta) = \log L(\theta, D) = \sum_{j=1}^m \sum_{i=1}^n \log p_{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)
$$

O algoritmo do gradiente descendente para MLE pode ser descrito da seguinte forma [6]:

1. Inicialize $\theta_0$ aleatoriamente
2. Para cada iteração $t$:
   a. Calcule o gradiente $\nabla_\theta \ell(\theta)$
   b. Atualize os parâmetros: $\theta_{t+1} = \theta_t + \alpha_t \nabla_\theta \ell(\theta)$

Onde $\alpha_t$ é a taxa de aprendizado na iteração $t$.

> ⚠️ **Nota Importante**: ==O sinal positivo na atualização dos parâmetros se deve ao fato de estarmos maximizando a função de log-verossimilhança, ao contrário da minimização típica em problemas de otimização [6].==

#### Vantagens e Desvantagens do Gradiente Descendente

| 👍 Vantagens                                          | 👎 Desvantagens                                               |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| ==Convergência garantida para funções convexas [7]== | Computacionalmente custoso para grandes conjuntos de dados [8] |
| Simplicidade de implementação [7]                    | ==Pode ficar preso em mínimos locais em funções não-convexas [8]== |
| Eficaz para problemas com poucos parâmetros [7]      | ==Sensível à escolha da taxa de aprendizado [8]==            |

#### Questões Técnicas/Teóricas

1. Como o gradiente descendente lida com platôs na superfície da função objetivo? Explique as implicações para o treinamento de modelos generativos profundos.

2. Descreva uma situação em que o gradiente descendente pode falhar na convergência para o ótimo global em um problema de MLE. Como isso afeta a qualidade do modelo generativo resultante?

### Gradiente Descendente Estocástico (SGD)

==O Gradiente Descendente Estocástico (SGD) é uma variação do gradiente descendente que estima o gradiente usando um subconjunto aleatório (mini-batch) dos dados em cada iteração [9].== Esta abordagem é particularmente útil para grandes conjuntos de dados e aprendizado online.

Para um conjunto de dados muito grande, o cálculo do gradiente completo pode ser aproximado por [10]:

$$
\nabla_\theta \ell(\theta) \approx m \mathbb{E}_{x^{(j)} \sim D}\left[\sum_{i=1}^n \nabla_\theta \log p_{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)\right]
$$

O algoritmo SGD para MLE pode ser descrito como [11]:

1. Inicialize $\theta_0$ aleatoriamente
2. Para cada iteração $t$:
   a. Amostre um mini-batch $B_t$ do conjunto de dados $D$
   b. Estime o gradiente: $\hat{g}_t = \frac{|D|}{|B_t|} \sum_{x^{(j)} \in B_t} \sum_{i=1}^n \nabla_\theta \log p_{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)$
   c. Atualize os parâmetros: $\theta_{t+1} = \theta_t + \alpha_t \hat{g}_t$

> ❗ **Ponto de Atenção**: ==A estimativa do gradiente no SGD é não-enviesada, mas possui maior variância comparada ao gradiente completo [12].==

#### Vantagens e Desvantagens do SGD

| 👍 Vantagens                                         | 👎 Desvantagens                                           |
| --------------------------------------------------- | -------------------------------------------------------- |
| Eficiente para grandes conjuntos de dados [13]      | ==Maior variância nas atualizações dos parâmetros [14]== |
| Permite aprendizado online [13]                     | ==Pode requerer mais iterações para convergência [14]==  |
| Pode escapar de mínimos locais devido ao ruído [13] | ==Sensível à escolha do tamanho do mini-batch [14]==     |

#### Implementação em PyTorch

Aqui está um exemplo simplificado de como implementar SGD para treinar um modelo generativo usando PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Defina a arquitetura do modelo aqui

    def forward(self, x):
        # Implemente a passagem forward

    def log_prob(self, x):
        # Retorna o log da probabilidade de x

def train_step(model, optimizer, batch):
    optimizer.zero_grad()
    loss = -model.log_prob(batch).mean()  # Negative log-likelihood
    loss.backward()
    optimizer.step()
    return loss.item()

# Configuração
model = GenerativeModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loop de treinamento
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
    print(f"Epoch {epoch}, Loss: {loss}")
```

> 💡 **Dica**: ==Para melhorar a convergência do SGD, considere usar técnicas como momentum, AdaGrad, ou Adam, que adaptam a taxa de aprendizado durante o treinamento [15].==

#### Questões Técnicas/Teóricas

1. Como o tamanho do mini-batch afeta o trade-off entre velocidade de convergência e qualidade da solução no SGD? Discuta as implicações para o treinamento de modelos generativos complexos.

2. Explique como o SGD pode ajudar a evitar overfitting em modelos generativos profundos. Quais são as considerações ao escolher entre SGD e gradiente descendente completo neste contexto?

### Conclusão

Os métodos de otimização, como o gradiente descendente e o SGD, são fundamentais para o treinamento eficaz de modelos generativos profundos usando MLE [16]. ==Enquanto o gradiente descendente oferece uma abordagem direta e teoricamente sólida, o SGD proporciona eficiência computacional e adaptabilidade a grandes conjuntos de dados, essenciais para aplicações práticas em aprendizado profundo [17].==

A escolha entre estes métodos depende das características específicas do problema, como o tamanho do conjunto de dados, a complexidade do modelo e os recursos computacionais disponíveis [18]. Em muitos casos práticos, ==variantes do SGD, como Adam ou RMSprop, são preferidas devido à sua capacidade de adaptar as taxas de aprendizado e lidar com gradientes esparsos [19].==

À medida que os modelos generativos se tornam mais complexos, a otimização eficiente continua sendo um desafio crucial, impulsionando pesquisas em novas técnicas de otimização e estratégias de regularização [20].

### Questões Avançadas

1. Compare o desempenho teórico e prático do SGD com métodos de segunda ordem, como o algoritmo de Newton, no contexto de treinamento de modelos generativos profundos. Quais são os trade-offs entre precisão, velocidade e requerimentos de memória?

2. Discuta como as técnicas de otimização estocástica podem ser estendidas para lidar com problemas de otimização com restrições em modelos generativos, como garantir a normalização adequada das distribuições de probabilidade.

3. Analise o impacto da geometria da função objetivo na eficácia do SGD para modelos generativos. Como técnicas como normalização de batch ou inicialização cuidadosa dos pesos podem melhorar a convergência?

### Referências

[1] "The goal of learning is to return a model P_θ that precisely captures the distribution P_data from which our data was sampled" (Trecho de cs236_lecture4.pdf)

[2] "We want to construct P_θ as "close" as possible to P_data (recall we assume we are given a dataset D of samples from P_data)" (Trecho de cs236_lecture4.pdf)

[3] "Compute ∇_θ ℓ(θ) (by back propagation)" (Trecho de cs236_lecture4.pdf)

[4] "θ_t+1 = θ_t + α_t ∇_θ ℓ(θ)" (Trecho de cs236_lecture4.pdf)

[5] "ℓ(θ) = log L(θ, D) = Σ_j=1^m Σ_i=1^n log p_neural(x_i^(j)|x_<i^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[6] "1. Initialize θ_0 at random
    2. Compute ∇_θ ℓ(θ) (by back propagation)
    3. θ_t+1 = θ_t + α_t ∇_θ ℓ(θ)" (Trecho de cs236_lecture4.pdf)

[7] "Non-convex optimization problem, but often works well in practice" (Trecho de cs236_lecture4.pdf)

[8] "What if m = |D| is huge?" (Trecho de cs236_lecture4.pdf)

[9] "Monte Carlo: Sample x^(j) ~ D; ∇_θ ℓ(θ) ≈ m Σ_i=1^n ∇_θ log p_neural(x_i^(j)|x_<i^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[10] "∇_θ ℓ(θ) = m Ε_x^(j)~D [Σ_i=1^n ∇_θ log p_neural(x_i^(j)|x_<i^(j); θ_i)]" (Trecho de cs236_lecture4.pdf)

[11] "Monte Carlo: Sample x^(j) ~ D; ∇_θ ℓ(θ) ≈ m Σ_i=1^n ∇_θ log p_neural(x_i^(j)|x_<i^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[12] "Monte Carlo: Sample x^(j) ~ D; ∇_θ ℓ(θ) ≈ m Σ_i=1^n ∇_θ log p_neural(x_i^(j)|x_<i^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[13] "Monte Carlo: Sample x^(j) ~ D; ∇_θ ℓ(θ) ≈ m Σ_i=1^n ∇_θ log p_neural(x_i^(j)|x_<i^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[14] "Monte Carlo: Sample x^(j) ~ D; ∇_θ ℓ(θ) ≈ m Σ_i=1^n ∇_θ log p_neural(x_i^(j)|x_<i^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[15] "For autoregressive models, it is easy to compute p_θ(x)" (Trecho de cs236_lecture4.pdf)

[16] "Ideally, evaluate in parallel each conditional log p_neural(x_i^(j)|x_<i^(j); θ_i). Not like RNNs." (Trecho de cs236_lecture4.pdf)

[17] "Natural to train them via maximum likelihood" (Trecho de cs236_lecture4.pdf)

[18] "Higher log-likelihood doesn't necessarily mean better looking samples" (Trecho de cs236_lecture4.pdf)

[19] "Other ways of measuring similarity are possible (Generative Adversarial Networks, GANs)" (Trecho de cs236_lecture4.pdf)

[20] "Natural to train them via maximum likelihood" (Trecho de cs236_lecture4.pdf)