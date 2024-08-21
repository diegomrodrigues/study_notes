## Estimação de Gradientes em Maximum Likelihood Estimation (MLE) com Mini-Batches

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240821152106481.png" alt="image-20240821152106481" style="zoom: 50%;" />

### Introdução

A estimação de máxima verossimilhança (Maximum Likelihood Estimation - MLE) é uma técnica fundamental em aprendizado de máquina e estatística para estimar os parâmetros de um modelo probabilístico. No contexto de modelos generativos profundos, o MLE é frequentemente utilizado em conjunto com técnicas de otimização baseadas em gradiente, como o gradiente descendente estocástico (SGD). Este resumo focará na aplicação do cálculo de gradientes em MLE, com ênfase especial na estimação de gradientes usando mini-batches, uma técnica crucial para treinar modelos em grandes conjuntos de dados [1][2].

### Conceitos Fundamentais

| Conceito                                | Explicação                                                   |
| --------------------------------------- | ------------------------------------------------------------ |
| **Maximum Likelihood Estimation (MLE)** | ==Método para estimar os parâmetros de um modelo estatístico maximizando a função de verossimilhança dos dados observados [1].== |
| **Gradiente Descendente**               | Algoritmo de otimização que iterativamente ajusta os parâmetros na direção oposta ao gradiente da função objetivo [2]. |
| **Mini-Batch**                          | Subconjunto aleatório do conjunto de dados usado para estimar o gradiente em cada iteração do treinamento [2]. |

> ✔️ **Ponto de Destaque**: A estimação de gradientes com mini-batches permite treinar modelos em grandes conjuntos de dados de forma eficiente, equilibrando velocidade computacional e precisão estatística [2].

### Formulação Matemática do MLE

O objetivo do MLE é encontrar os parâmetros θ que maximizam a probabilidade dos dados observados. Para um conjunto de dados $D = {x^{(1)}, ..., x^{(m)}}$, a função de log-verossimilhança é dada por [1]:

$$
\ell(\theta) = \log L(\theta, D) = \sum_{j=1}^m \sum_{i=1}^n \log p_\text{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)
$$

Onde:
- ==$p_\text{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)$ é a probabilidade condicional modelada por uma rede neural==
- $\theta_i$ são os parâmetros associados à i-ésima variável
- $n$ é o número de variáveis
- $m$ é o tamanho do conjunto de dados

### Cálculo do Gradiente

O gradiente da log-verossimilhança com respeito aos parâmetros θ é [2]:

$$
\nabla_\theta \ell(\theta) = \sum_{j=1}^m \sum_{i=1}^n \nabla_\theta \log p_\text{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)
$$

Este gradiente é a soma das contribuições de todas as amostras e todas as variáveis do modelo.

#### Questões Técnicas/Teóricas

1. Como a forma da função de log-verossimilhança influencia a escolha do algoritmo de otimização para MLE em modelos generativos profundos?
2. Quais são as implicações práticas de usar o gradiente da log-verossimilhança em vez do gradiente da verossimilhança direta?

### Estimação de Gradientes com Mini-Batches

A estimação de gradientes com mini-batches é uma técnica crucial para treinar modelos em grandes conjuntos de dados. ==Ela se baseia no princípio de que podemos obter uma estimativa não-enviesada do gradiente usando apenas um subconjunto dos dados [2].==

#### Formulação Matemática

Para um mini-batch B de tamanho b << m, o gradiente estimado é [2]:

$$
\nabla_\theta \ell(\theta) \approx \frac{m}{b} \sum_{j \in B} \sum_{i=1}^n \nabla_\theta \log p_\text{neural}(x_i^{(j)}|x_{<i}^{(j)}; \theta_i)
$$

Esta estimativa é não-enviesada e sua variância diminui à medida que o tamanho do mini-batch aumenta.

> ❗ **Ponto de Atenção**: ==O fator de escala m/b é crucial para manter a estimativa não-enviesada do gradiente [2].==

#### Implementação em PyTorch

Aqui está um exemplo simplificado de como implementar o cálculo de gradientes com mini-batches em PyTorch:

```python
import torch
import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Definição da arquitetura da rede

    def forward(self, x):
        # Implementação do modelo autoregressivo
        return log_probs

def train_step(model, optimizer, batch):
    optimizer.zero_grad()
    log_probs = model(batch)
    loss = -log_probs.mean()  # Negative log-likelihood
    loss.backward()
    optimizer.step()
    return loss.item()

# Treinamento
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
```

Neste exemplo, `AutoregressiveModel` representa um modelo generativo autoregressivo, e `train_step` realiza uma única atualização de gradiente usando um mini-batch.

### Vantagens e Desvantagens da Estimação com Mini-Batches

| 👍 Vantagens                                                 | 👎 Desvantagens                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| Permite treinar em grandes conjuntos de dados [2]           | ==Introduz ruído na estimativa do gradiente [2]==           |
| ==Acelera a convergência em termos de iterações [2]==       | Pode requerer ajustes cuidadosos na taxa de aprendizado [2] |
| ==Melhora a generalização devido ao ruído estocástico [2]== | ==Pode levar a oscilações no processo de otimização [2]==   |

### Técnicas Avançadas

1. **Momentum**: Incorpora informações de gradientes passados para suavizar a trajetória de otimização [3].

2. **Adaptive Learning Rates**: Algoritmos como Adam ajustam automaticamente as taxas de aprendizado para cada parâmetro [3].

3. **Gradient Clipping**: Limita a norma do gradiente para evitar explosões de gradiente em modelos profundos [3].

#### Questões Técnicas/Teóricas

1. Como o tamanho do mini-batch afeta o trade-off entre velocidade de treinamento e qualidade da estimativa do gradiente?
2. Quais são as considerações ao escolher entre SGD com mini-batches e métodos de otimização de segunda ordem para MLE em modelos generativos?

### Conclusão

A estimação de gradientes com mini-batches é uma técnica fundamental para treinar modelos generativos profundos usando MLE. Ela permite o treinamento eficiente em grandes conjuntos de dados, equilibrando custo computacional e precisão estatística. Embora introduza ruído nas estimativas de gradiente, este ruído pode ter efeitos benéficos na generalização do modelo. A implementação eficaz desta técnica, juntamente com outras otimizações como momentum e taxas de aprendizado adaptativas, é crucial para o sucesso do treinamento de modelos generativos modernos [1][2][3].

### Questões Avançadas

1. Como a estrutura de dependência em modelos autoregressivos afeta a eficiência da estimação de gradientes com mini-batches? Proponha uma estratégia para otimizar o cálculo de gradientes levando em conta essas dependências.

2. Discuta as implicações teóricas e práticas de usar estimativas de gradiente enviesadas (como em alguns métodos de redução de variância) versus estimativas não-enviesadas no contexto de MLE para modelos generativos profundos.

3. Desenvolva uma análise comparativa entre o uso de mini-batches em MLE e em outros paradigmas de treinamento para modelos generativos, como GAN (Generative Adversarial Networks) ou VAE (Variational Autoencoders). Como as características específicas de cada abordagem influenciam a escolha do tamanho do mini-batch e outras hiper-parâmetros de otimização?

### Referências

[1] "Maximize arg max_θ L(θ, D) = arg max_θ log L(θ, D)" (Trecho de cs236_lecture4.pdf)

[2] "ℓ(θ) = log L(θ, D) = ∑_j=1^m ∑_i=1^n log p_neural(x_i^(j)|x_{<i}^(j); θ_i)" (Trecho de cs236_lecture4.pdf)

[3] "1. Initialize θ_0 at random
2. Compute ∇_θ ℓ(θ) (by back propagation)
3. θ_t+1 = θ_t + α_t ∇_θ ℓ(θ)" (Trecho de cs236_lecture4.pdf)