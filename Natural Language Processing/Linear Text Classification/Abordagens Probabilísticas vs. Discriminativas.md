# Abordagens Probabilísticas vs. Discriminativas: Distinguindo entre Classificadores Probabilísticos e Discriminativos

![Diagrama comparativo mostrando Naïve Bayes (representado por uma rede bayesiana) de um lado e SVM/Perceptron (representado por um hiperplano separador) do outro, com setas apontando para suas características distintas.]

## Introdução

Na classificação de texto e no aprendizado de máquina em geral, duas abordagens fundamentais se destacam: os classificadores **probabilísticos** e os **discriminativos**. Compreender essa distinção é crucial para apreciar as diferentes metodologias de aprendizado e suas implicações teóricas e práticas [1]. ==Este resumo foca em distinguir essas duas abordagens, utilizando o **Naïve Bayes** como exemplo de classificador probabilístico e o **Perceptron** e a **SVM** (Support Vector Machine) como exemplos de classificadores discriminativos.==

Os classificadores probabilísticos, como o Naïve Bayes, ==baseiam-se na modelagem da distribuição de probabilidade conjunta dos dados e rótulos.== Por outro lado, os classificadores discriminativos, como o Perceptron e a SVM, ==focam diretamente na tarefa de discriminação entre classes, sem modelar explicitamente a distribuição dos dados [2]==. Essa diferença fundamental leva a distintas abordagens de aprendizado, representações de conhecimento e desempenhos variados em diversas tarefas de classificação.

## Conceitos Fundamentais

| Conceito                         | Explicação                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Classificador Probabilístico** | ==Modela a distribuição de probabilidade conjunta $p(X, Y)$ dos dados $X$ e rótulos $Y$.== No caso do Naïve Bayes, isso é feito através da decomposição $p(X, Y) = p(Y) p(X \mid Y)$. |
| **Classificador Discriminativo** | ==Foca diretamente na modelagem da probabilidade condicional $p(Y \mid X)$ ou na fronteira de decisão entre classes==, sem a necessidade de modelar a distribuição dos dados [3]. |
| **Naïve Bayes**                  | Um classificador probabilístico que faz a suposição "ingênua" de ==independência condicional== entre as características dado a classe. Isso simplifica significativamente o cálculo da verossimilhança [4]. |
| **Perceptron**                   | Um classificador linear discriminativo que aprende atualizando pesos iterativamente com base nos erros de classificação. ==É um dos algoritmos mais simples de aprendizado de máquina [5].== |
| **SVM (Support Vector Machine)** | Um classificador discriminativo que busca encontrar o hiperplano de margem máxima que separa as classes. ==É conhecido por sua capacidade de generalização e robustez [6].== |

> ⚠️ **Nota Importante**: A escolha entre abordagens probabilísticas e discriminativas pode ter um impacto significativo no desempenho do modelo, dependendo da natureza dos dados e da tarefa em questão [7].

### Modelagem Probabilística vs. Discriminativa

![Gráfico comparativo mostrando a função de decisão de um classificador Naïve Bayes (curvas de probabilidade) e um SVM (hiperplano) em um espaço bidimensional.]

A distinção fundamental entre as abordagens probabilística e discriminativa reside na forma como elas modelam o problema de classificação [8].

#### Modelagem Probabilística (Naïve Bayes)

O Naïve Bayes, como classificador probabilístico, modela a distribuição conjunta $p(X, Y)$ [9]. Para um problema de classificação binária, temos:

$$
p(Y \mid X) = \frac{p(X \mid Y) p(Y)}{p(X)}
$$

Onde:

- $p(Y \mid X)$ é a probabilidade posterior.
- $p(X \mid Y)$ é a verossimilhança.
- $p(Y)$ é a probabilidade a priori.
- $p(X)$ é a evidência (constante de normalização).

==A suposição "ingênua" de independência condicional do Naïve Bayes permite decompor a verossimilhança [10]:==
$$
p(X \mid Y) = \prod_{j=1}^V p(X_j \mid Y)
$$

Onde $V$ é o número de características.

#### Modelagem Discriminativa (Perceptron e SVM)

==Os classificadores discriminativos, como o Perceptron e a SVM, modelam diretamente a fronteira de decisão entre as classes [11]==. Para um classificador linear, a função de decisão tem a forma:
$$
f(X) = \text{sign}(\theta \cdot X + b)
$$

Onde:

- $\theta$ é o vetor de pesos.
- $b$ é o viés.
- $X$ é o vetor de características.

No caso da SVM, busca-se maximizar a margem entre as classes [12]:

$$
\max_{\theta, b} \quad \frac{2}{\lVert \theta \rVert} \quad \text{sujeito a} \quad y_i (\theta \cdot x_i + b) \geq 1, \quad \forall i
$$

#### Perguntas Teóricas

1. **Derive a expressão para a atualização de pesos do Perceptron e explique como ela difere conceitualmente da estimativa de máxima verossimilhança no Naïve Bayes.**
2. **Considerando um conjunto de dados linearmente separável, prove que a SVM de margem rígida sempre encontrará uma solução ótima, enquanto o Perceptron pode não convergir em um número finito de iterações.**
3. **Analise teoricamente como a suposição de independência condicional do Naïve Bayes afeta sua capacidade de modelar interações complexas entre características, em comparação com abordagens discriminativas.**

### Estimação de Parâmetros

A estimação de parâmetros é um aspecto crucial que diferencia as abordagens probabilísticas e discriminativas [13].

#### Naïve Bayes

No Naïve Bayes, os parâmetros são estimados usando o princípio da máxima verossimilhança [14]. Para um vocabulário de $V$ palavras e $K$ classes, temos:

$$
\phi_{y,j} = \frac{\text{count}(y, j)}{\sum_{j'=1}^V \text{count}(y, j')} = \frac{\sum_{i: y^{(i)}=y} x_j^{(i)}}{\sum_{j'=1}^V \sum_{i: y^{(i)}=y} x_{j'}^{(i)}}
$$

Onde $\text{count}(y, j)$ é a contagem da palavra $j$ em documentos com rótulo $y$.

==Para evitar problemas com palavras não vistas no treinamento, é comum usar **suavização de Laplace** [15]:==
$$
\phi_{y,j} = \frac{\alpha + \text{count}(y, j)}{V\alpha + \sum_{j'=1}^V \text{count}(y, j')}
$$

Onde $\alpha$ é o hiperparâmetro de suavização.

#### Perceptron

O Perceptron atualiza seus pesos de forma online, baseando-se nos erros de classificação [16]:

$$
\theta^{(t)} = \theta^{(t-1)} + y^{(i)} x^{(i)}
$$

Se a previsão está incorreta, ou seja, se $y^{(i)} (\theta^{(t-1)} \cdot x^{(i)}) \leq 0$.

#### SVM

A SVM resolve um problema de otimização quadrática para encontrar o hiperplano de margem máxima [17]:

$$
\min_{\theta, b} \quad \frac{1}{2} \lVert \theta \rVert^2 + C \sum_{i=1}^N \xi_i
$$

Sujeito a:

$$
y_i (\theta \cdot x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \quad \forall i
$$

Onde $\xi_i$ são variáveis de folga e $C$ é um hiperparâmetro de regularização.

> ❗ **Ponto de Atenção**: A estimação de parâmetros em modelos discriminativos geralmente envolve otimização numérica complexa, enquanto em modelos probabilísticos como o Naïve Bayes, muitas vezes é possível obter estimativas de forma fechada [18].

### Funções de Perda e Otimização

As funções de perda e os métodos de otimização utilizados diferem fundamentalmente entre as abordagens probabilísticas e discriminativas [19].

#### Naïve Bayes

==A função objetivo do Naïve Bayes é maximizar a **log-verossimilhança** [20]:==
$$
\mathcal{L}(\phi, \mu) = \sum_{i=1}^N \left[ \log p_{\text{mult}}(x^{(i)}; \phi_{y^{(i)}}) + \log p_{\text{cat}}(y^{(i)}; \mu) \right]
$$

A otimização desta função leva às estimativas de máxima verossimilhança mencionadas anteriormente.

#### Perceptron

O Perceptron minimiza uma ==função de perda baseada nos erros de classificação [21]:==

$$
\ell_{\text{Perceptron}}(\theta; x^{(i)}, y^{(i)}) = - y^{(i)} (\theta \cdot x^{(i)})
$$

Atualizando os pesos apenas quando a classificação está incorreta.

#### SVM

A SVM minimiza uma ==combinação de perda de **hinge** (dobradiça) e regularização $L2$ [22]:==

$$
L_{\text{SVM}} = \frac{1}{2} \lVert \theta \rVert^2 + C \sum_{i=1}^N \max \left( 0, 1 - y^{(i)} (\theta \cdot x^{(i)} + b) \right)
$$

> ✔️ **Destaque**: A escolha da função de perda e do método de otimização tem implicações significativas na interpretabilidade do modelo, na velocidade de convergência e na capacidade de generalização [23].

### Vantagens e Desvantagens

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Naïve Bayes**: Simples, rápido de treinar, eficaz com dados esparsos [24]. | **Naïve Bayes**: A suposição de independência pode ser irrealista, pode sofrer com o problema de probabilidade zero [25]. |
| **Perceptron**: Simples de implementar, online e eficiente [26]. | **Perceptron**: Pode não convergir para dados não linearmente separáveis, sensível à ordem dos dados [27]. |
| **SVM**: Boa generalização, eficaz em espaços de alta dimensão [28]. | **SVM**: Treinamento pode ser computacionalmente intensivo, a escolha do kernel pode ser desafiadora [29]. |

#### Perguntas Teóricas

1. **Derive a expressão para o gradiente da função de perda da SVM e compare-a com o gradiente da log-verossimilhança do Naïve Bayes. Discuta as implicações dessas diferenças na otimização.**
2. **Analise teoricamente como a suposição de independência condicional do Naïve Bayes afeta sua capacidade de modelar interações complexas entre características, em comparação com a SVM e o Perceptron.**
3. **Considerando um conjunto de dados não linearmente separável, prove que a SVM com kernel pode encontrar uma solução, enquanto o Perceptron linear falhará. Discuta as implicações teóricas desta diferença.**

### Implementação Avançada

Aqui está um exemplo avançado de implementação de um classificador SVM usando **PyTorch**, demonstrando como a abordagem discriminativa pode ser implementada de forma eficiente [30]:

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
    return torch.mean(torch.clamp(1 - outputs.view(-1) * labels, min=0))

def train_svm(model, X, y, learning_rate=0.01, num_epochs=1000):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = hinge_loss(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Época [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Exemplo de uso
X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0]], dtype=torch.float32)
y = torch.tensor([1, 1, -1, -1], dtype=torch.float32)

model = SVM(input_dim=2)
train_svm(model, X, y)
```

Este código implementa uma SVM linear usando PyTorch, demonstrando como a otimização baseada em gradiente pode ser aplicada a um classificador discriminativo [31].

## Conclusão

A distinção entre classificadores probabilísticos e discriminativos é fundamental na teoria e na prática do aprendizado de máquina. O Naïve Bayes, como representante da abordagem probabilística, oferece uma modelagem explícita da distribuição dos dados, permitindo inferências probabilísticas diretas. Por outro lado, classificadores discriminativos como o Perceptron e a SVM focam na fronteira de decisão, frequentemente alcançando melhor desempenho em tarefas de classificação pura [32].

A escolha entre essas abordagens depende de vários fatores, incluindo a natureza dos dados, o tamanho do conjunto de treinamento, a necessidade de interpretabilidade e os requisitos computacionais. Enquanto o Naïve Bayes pode ser mais adequado para conjuntos de dados pequenos ou quando estimativas de probabilidade são necessárias, a SVM e o Perceptron podem oferecer melhor desempenho em tarefas de alta dimensionalidade ou quando a suposição de independência do Naïve Bayes é violada [33].

Compreender as diferenças teóricas e práticas entre essas abordagens é crucial para os cientistas de dados, permitindo a escolha informada de modelos e a interpretação adequada dos resultados em diversos cenários de aprendizado de máquina [34].

## Perguntas Teóricas Avançadas

1. **Considere um problema de classificação binária com características $X$ e rótulos $Y$. Derive a expressão para o erro de Bayes e compare-a com o limite inferior do erro de generalização da SVM de margem rígida. Discuta as implicações teóricas dessa comparação.**

2. **Prove que, para um conjunto de dados linearmente separável, o algoritmo do Perceptron converge em um número finito de iterações. Compare essa garantia de convergência com a da SVM e discuta as implicações práticas dessas diferenças teóricas.**

3. **Analise teoricamente como a suposição de independência condicional do Naïve Bayes afeta sua capacidade de modelar interações complexas entre características. Proponha e analise alternativas que permitam ao Naïve Bayes capturar tais interações sem sacrificar demasiadamente a simplicidade computacional.**

---

**Referências**

[1] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[2] Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. *Advances in Neural Information Processing Systems*, 14.

[3] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

[4] Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

[5] Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386.

[6] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

[7] Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.

[8] Jordan, M. I. (2002). An introduction to probabilistic graphical models.

[9] Russell, S. J., & Norvig, P. (2016). *Artificial Intelligence: A Modern Approach*. Pearson.

[10] Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.

[11] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

[12] Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.

[13] Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*. Cambridge University Press.

[14] Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification*. Wiley.

[15] Lidstone, G. J. (1920). Note on the general case of the Bayes–Laplace formula for inductive or a posteriori probabilities. *Transactions of the Faculty of Actuaries*, 8, 182-192.

[16] Freund, Y., & Schapire, R. E. (1999). Large margin classification using the perceptron algorithm. *Machine Learning*, 37(3), 277-296.

[17] Platt, J. (1999). *Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods*. MIT Press.

[18] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[19] Rifkin, R., & Klautau, A. (2004). In defense of one-vs-all classification. *Journal of Machine Learning Research*, 5, 101-141.

[20] Bottou, L., & Bousquet, O. (2008). The tradeoffs of large scale learning. *Advances in Neural Information Processing Systems*, 20.

[21] Rosenblatt, F. (1957). The perceptron—a perceiving and recognizing automaton. *Report 85-460-1*, Cornell Aeronautical Laboratory.

[22] Collobert, R., & Bengio, S. (2004). Links between perceptrons, MLPs and SVMs. *Proceedings of the 21st International Conference on Machine Learning*.

[23] Joachims, T. (1998). Making large-scale SVM learning practical. *Advances in Kernel Methods*, 169-184.

[24] Zhang, H. (2004). The optimality of Naive Bayes. *AAAI*, 3(1), 562-567.

[25] Hand, D. J., & Yu, K. (2001). Idiot's Bayes—not so stupid after all? *International Statistical Review*, 69(3), 385-398.

[26] Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119-139.

[27] Novikoff, A. B. J. (1962). On convergence proofs on perceptrons. *Proceedings of the Symposium on the Mathematical Theory of Automata*, 615-622.

[28] Burges, C. J. (1998). A tutorial on support vector machines for pattern recognition. *Data Mining and Knowledge Discovery*, 2(2), 121-167.

[29] Bennett, K. P., & Campbell, C. (2000). Support vector machines: Hype or hallelujah? *ACM SIGKDD Explorations Newsletter*, 2(2), 1-13.

[30] Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.

[31] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[32] Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. *Proceedings of the Twenty-First International Conference on Machine Learning*.

[33] Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29(2-3), 103-130.

[34] Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.

[35] Vapnik, V. (2013). *The Nature of Statistical Learning Theory*. Springer Science & Business Media.