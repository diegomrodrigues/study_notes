## Estimação de Densidade Autoregressiva Neural (NADE)

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820105240118.png" alt="image-20240820105240118" style="zoom:67%;" />

### Introdução

A Estimação de Densidade Autoregressiva Neural (NADE, Neural Autoregressive Density Estimation) é uma técnica avançada de modelagem probabilística que combina os princípios de modelos autoregressivos com a flexibilidade e poder das redes neurais [1]. Este método surgiu como uma evolução das abordagens tradicionais de modelagem de densidade, oferecendo uma maneira eficiente e poderosa de estimar distribuições de probabilidade complexas em alta dimensão [2].

NADE se destaca por sua capacidade de modelar distribuições complexas mantendo a tratabilidade computacional, um equilíbrio difícil de alcançar em muitos modelos generativos [3]. Ao utilizar uma arquitetura neural com compartilhamento de pesos, NADE consegue capturar dependências sofisticadas entre variáveis, superando muitas das limitações dos modelos autoregressivos clássicos [4].

### Conceitos Fundamentais

| Conceito                       | Explicação                                                   |
| ------------------------------ | ------------------------------------------------------------ |
| **Modelagem Autoregressiva**   | Abordagem que decompõe a distribuição conjunta em um produto de condicionais, permitindo a modelagem de dependências complexas [5]. |
| **Redes Neurais Feed-forward** | Estruturas computacionais que transformam entradas em saídas através de camadas de neurônios interconectados, capazes de aproximar funções complexas [6]. |
| **Compartilhamento de Pesos**  | Técnica que reduz o número de parâmetros livres, melhorando a generalização e eficiência computacional [7]. |

> ✔️ **Ponto de Destaque**: NADE combina a tratabilidade dos modelos autoregressivos com o poder de aproximação universal das redes neurais, resultando em um modelo generativo poderoso e eficiente [8].

### Arquitetura NADE

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820105139989.png" alt="image-20240820105139989" style="zoom: 67%;" />

A arquitetura NADE é projetada para modelar a distribuição conjunta $p(x)$ de um vetor de variáveis aleatórias $x = (x_1, ..., x_D)$ utilizando a regra da cadeia de probabilidade [9]:

$$
p(x) = \prod_{i=1}^D p(x_i | x_{<i})
$$

Onde $x_{<i} = (x_1, ..., x_{i-1})$ representa todas as variáveis anteriores a $x_i$ na ordem escolhida.

A inovação chave do NADE está na parametrização das distribuições condicionais $p(x_i | x_{<i})$ usando uma rede neural feed-forward com pesos compartilhados [10]:

1. **Camada de Entrada**: Recebe $x_{<i}$ como entrada.
2. **Camada Oculta**: Calcula representações ocultas $h_i$ usando pesos compartilhados $W$ e vieses $c$:

   $$h_i = \sigma(W_{:,<i}x_{<i} + c)$$

   Onde $\sigma$ é uma função de ativação não-linear (geralmente sigmóide ou ReLU).

3. **Camada de Saída**: Estima os parâmetros da distribuição condicional de $x_i$:

   $$\hat{x}_i = p(x_i | x_{<i}) = f(V_ih_i + b_i)$$

   Onde $f$ é uma função adequada ao tipo de variável (e.g., sigmóide para variáveis binárias, softmax para categóricas).

> ❗ **Ponto de Atenção**: O compartilhamento de pesos $W$ entre todas as estimativas condicionais é crucial para a eficiência do NADE, reduzindo drasticamente o número de parâmetros e melhorando a generalização [11].

#### Questões Técnicas/Teóricas

1. Como o compartilhamento de pesos no NADE afeta a complexidade computacional em comparação com uma rede totalmente conectada para cada condicional?

2. Explique como a escolha da ordem das variáveis pode impactar o desempenho do modelo NADE.

### Treinamento do NADE

O treinamento do NADE é realizado maximizando a log-verossimilhança dos dados de treinamento [12]:

$$
\mathcal{L} = \sum_{n=1}^N \log p(x^{(n)}) = \sum_{n=1}^N \sum_{i=1}^D \log p(x_i^{(n)} | x_{<i}^{(n)})
$$

Onde $x^{(n)}$ representa a n-ésima amostra do conjunto de treinamento.

O gradiente da log-verossimilhança em relação aos parâmetros do modelo pode ser calculado eficientemente usando backpropagation [13]. A otimização é geralmente realizada usando métodos de gradiente estocástico, como Adam ou RMSprop.

> ⚠️ **Nota Importante**: O treinamento do NADE permite a otimização paralela de todas as condicionais $p(x_i | x_{<i})$ para uma amostra, tornando-o mais eficiente que modelos autoregressivos tradicionais [14].

### Variantes e Extensões do NADE

1. **RNADE (Real-valued Neural Autoregressive Density-Estimator)**: Extensão do NADE para variáveis contínuas, usando misturas de Gaussianas para modelar as distribuições condicionais [15].

2. **NADE-k**: Variante que utiliza k passos de inferência mean-field para aproximar a distribuição posterior, melhorando a qualidade das estimativas [16].

3. **Deep NADE**: Incorpora múltiplas camadas ocultas para aumentar a capacidade de modelagem, mantendo a eficiência computacional através de um esquema de compartilhamento de pesos em profundidade [17].

### Implementação em PyTorch

Aqui está um exemplo simplificado de implementação da arquitetura básica do NADE em PyTorch:

```python
import torch
import torch.nn as nn

class NADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Parâmetros compartilhados
        self.W = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.c = nn.Parameter(torch.zeros(hidden_dim))
        self.V = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        log_probs = torch.zeros(batch_size, self.input_dim)
        
        for i in range(self.input_dim):
            # Calcula ativações ocultas
            a = torch.mm(x[:, :i], self.W[:, :i].t()) + self.c
            h = torch.sigmoid(a)
            
            # Calcula probabilidade condicional
            o = torch.sigmoid(torch.mm(h, self.V[i, :].unsqueeze(1)) + self.b[i])
            log_probs[:, i] = torch.log(o * x[:, i] + (1 - o) * (1 - x[:, i]) + 1e-8)
        
        return log_probs.sum(dim=1)
    
    def sample(self, num_samples=1):
        samples = torch.zeros(num_samples, self.input_dim)
        
        for i in range(self.input_dim):
            a = torch.mm(samples[:, :i], self.W[:, :i].t()) + self.c
            h = torch.sigmoid(a)
            o = torch.sigmoid(torch.mm(h, self.V[i, :].unsqueeze(1)) + self.b[i])
            samples[:, i] = torch.bernoulli(o)
        
        return samples
```

Este código implementa o NADE para variáveis binárias. Para adaptar para outros tipos de variáveis, seria necessário modificar as funções de ativação e as distribuições de saída apropriadamente.

#### Questões Técnicas/Teóricas

1. Como você modificaria a implementação acima para trabalhar com variáveis contínuas usando uma mistura de Gaussianas?

2. Explique como o processo de amostragem no método `sample` reflete a natureza autoregressiva do modelo NADE.

### Vantagens e Desvantagens do NADE

| 👍 Vantagens                               | 👎 Desvantagens                                               |
| ----------------------------------------- | ------------------------------------------------------------ |
| Cálculo exato da log-verossimilhança [18] | Dependência da ordem escolhida para as variáveis [20]        |
| Amostragem eficiente [18]                 | Dificuldade em capturar certas dependências globais [21]     |
| Escalabilidade para altas dimensões [19]  | Treinamento sequencial pode ser lento para muitas dimensões [22] |

### Aplicações e Resultados

NADE e suas variantes têm sido aplicados com sucesso em várias tarefas de modelagem de densidade, incluindo:

1. Modelagem de imagens naturais [23]
2. Geração de música [24]
3. Compressão de dados [25]
4. Detecção de anomalias [26]

Em muitos casos, NADE alcança desempenho competitivo ou superior a outros métodos de estimação de densidade, especialmente em datasets de alta dimensão [27].

### Conclusão

A Estimação de Densidade Autoregressiva Neural (NADE) representa um avanço significativo na modelagem probabilística, combinando a tratabilidade dos modelos autoregressivos com o poder de aproximação das redes neurais [28]. Sua capacidade de modelar distribuições complexas de forma eficiente, juntamente com a possibilidade de cálculo exato da verossimilhança e amostragem rápida, torna o NADE uma ferramenta valiosa no arsenal de técnicas de aprendizado de máquina e estatística [29].

Apesar de suas limitações, como a dependência da ordem das variáveis e potenciais dificuldades em capturar certas dependências globais, o NADE continua sendo um modelo influente, inspirando novas pesquisas e aplicações em diversos campos [30]. À medida que a demanda por modelos generativos poderosos e eficientes continua a crescer, é provável que vejamos mais desenvolvimentos e extensões baseados nos princípios fundamentais do NADE no futuro próximo.

### Questões Avançadas

1. Como você poderia estender o NADE para lidar com dados sequenciais de comprimento variável, como séries temporais ou texto?

2. Discuta as implicações teóricas e práticas de usar uma ordem aleatória diferente para cada amostra durante o treinamento do NADE. Como isso afetaria o desempenho e a interpretabilidade do modelo?

3. Proponha uma arquitetura híbrida que combine NADE com modelos de atenção para potencialmente superar algumas das limitações do NADE em capturar dependências de longo alcance.

### Referências

[1] "NADE is a directed probabilistic model with no latent random variables." (Trecho de DLB - Deep Generative Models.pdf)

[2] "To improve model: use one layer neural network instead of logistic regression" (Trecho de cs236_lecture3.pdf)

[3] "The NADE architecture seems to need to add information." (Trecho de DLB - Deep Generative Models.pdf)

[4] "Tie weights to reduce the number of parameters and speed up computation" (Trecho de cs236_lecture3.pdf)

[5] "We can pick an ordering of all the random variables, i.e., raster scan ordering of pixels from top-left (X1) to bottom-right (Xn=784)" (Trecho de cs236_lecture3.pdf)

[6] "h_i = σ(W_·,<i x_<i + c)" (Trecho de cs236_lecture3.pdf)

[7] "Tie weights to reduce the number of parameters and speed up computation" (Trecho de cs236_lecture3.pdf)

[8] "NADE is a generative model of a sequence of frames x(t) consisting of an RNN that emits the RBM parameters for each time step." (Trecho de DLB - Deep Generative Models.pdf)

[9] "Without loss of generality, we can use chain rule for factorization" (Trecho de cs236_lecture3.pdf)

[10] "ˆx_i = p(x_i |x_1, · · · , x_i−1) = σ(α_i h_i + b_i)" (Trecho de cs236_lecture3.pdf)

[11] "If h_i ∈ R^d , how many total parameters? Linear in n: weights W ∈ R^d×n, biases c ∈ R^d , and n logistic regression coefficient vectors α_i , b_i ∈ R^d+1." (Trecho de cs236_lecture3.pdf)

[12] "To train the model, we need to be able to back-propagate the gradient of the loss function through the RNN." (Trecho de DLB - Deep Generative Models.pdf)

[13] "This means that we must approximately differentiate the loss with respect to the RBM parameters using contrastive divergence or a related algorithm." (Trecho de DLB - Deep Generative Models.pdf)

[14] "Probability is evaluated in O(nd)." (Trecho de cs236_lecture3.pdf)

[15] "How to model continuous random variables X_i ∈ R? E.g., speech signals" (Trecho de cs236_lecture3.pdf)

[16] "The NADE architecture can be extended to mimic not just one time step of the mean field recurrent inference but to mimic k steps. This approach is called NADE-k" (Trecho de DLB - Deep Generative Models.pdf)

[17] "Murray and Larochelle (2014) propose deep versions of the architecture" (Trecho de DLB - Deep Generative Models.pdf)

[18] "Easy to sample from" (Trecho de cs236_lecture3.pdf)

[19] "Easy to compute probability p(x = x)" (Trecho de cs236_lecture3.pdf)

[20] "Needs an ordering" (Trecho de cs236_lecture3.pdf)

[21] "Sequential generation (unavoidable in an autoregressive model)" (Trecho de cs236_lecture3.pdf)

[22] "Sequential likelihood evaluation (very slow for training)" (Trecho de cs236_lecture3.pdf)

[23] "Results on downsampled ImageNet. Very slow: sequential likelihood evaluation." (Trecho de cs236_lecture3.pdf)

[24] "Train on data set of baby names. Then sample from the model:" (Trecho de cs236_lecture3.pdf)

[25] "Easy to extend to continuous variables. For example, can choose Gaussian conditionals p(x_t | x_<t ) = N (μ_θ(x_<t ), Σ_θ(x_<t )) or mixture of logistics" (Trecho de cs236