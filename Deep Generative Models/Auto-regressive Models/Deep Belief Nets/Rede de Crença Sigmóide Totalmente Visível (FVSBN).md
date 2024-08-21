## Rede de Crença Sigmóide Totalmente Visível (FVSBN)

<img src="C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240819090552882.png" alt="image-20240819090552882" style="zoom: 80%;" />

### Introdução

A Rede de Crença Sigmóide Totalmente Visível (FVSBN - Fully Visible Sigmoid Belief Network) é um modelo probabilístico poderoso para representar distribuições sobre variáveis binárias. Este modelo pertence à classe de modelos autoregressivos, que decompõem a distribuição conjunta em um produto de distribuições condicionais [1]. A FVSBN é particularmente interessante devido à sua capacidade de modelar dependências complexas entre variáveis, mantendo uma estrutura computacionalmente tratável.

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Modelo Autoregressivo** | Um modelo que expressa a probabilidade conjunta como um produto de probabilidades condicionais, seguindo uma ordem específica das variáveis. [1] |
| **Regressão Logística**   | Método usado para modelar cada probabilidade condicional na FVSBN, permitindo capturar relações não-lineares entre as variáveis. [6] |
| **Amostragem Ancestral**  | Técnica para gerar amostras da distribuição modelada, seguindo a ordem definida pelo modelo autoregressivo. [7] |

> ⚠️ **Nota Importante**: A FVSBN é um modelo totalmente observável, diferindo de muitos outros modelos probabilísticos que incluem variáveis latentes.

### Estrutura do Modelo FVSBN

A FVSBN modela uma distribuição conjunta sobre variáveis binárias $X = (X_1, ..., X_n)$ usando a regra da cadeia de probabilidade:

$$
p(x_1, ..., x_n) = p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)...p(x_n|x_1,...,x_{n-1})
$$

Cada probabilidade condicional é modelada usando regressão logística [6]:

$$
p(X_i = 1|x_1, ..., x_{i-1}) = \sigma(\alpha_i^0 + \sum_{j=1}^{i-1} \alpha_i^j x_j)
$$

Onde $\sigma(z) = \frac{1}{1 + e^{-z}}$ é a função sigmóide e $\alpha_i^j$ são os parâmetros do modelo.

### Avaliação da Probabilidade Conjunta

Para avaliar $p(x_1, ..., x_n)$ para um dado vetor $x$, seguimos estes passos [7]:

1. Calcular $p(x_1)$ usando os parâmetros apropriados.
2. Para $i = 2$ até $n$:
   - Calcular $p(x_i|x_1, ..., x_{i-1})$ usando a equação da regressão logística.
3. Multiplicar todas as probabilidades calculadas.

> ✔️ **Ponto de Destaque**: A avaliação da probabilidade conjunta pode ser realizada em tempo $O(n^2)$, onde $n$ é o número de variáveis.

### Amostragem da Distribuição

![image-20240819233713726](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240819233713726.png)

Para gerar uma amostra da distribuição modelada pela FVSBN [7]:

1. Amostrar $x_1 \sim p(x_1)$.
2. Para $i = 2$ até $n$:
   - Calcular $p(x_i = 1|x_1, ..., x_{i-1})$.
   - Amostrar $x_i \sim \text{Bernoulli}(p(x_i = 1|x_1, ..., x_{i-1}))$.

Este processo é conhecido como amostragem ancestral e garante que a amostra gerada segue a distribuição modelada pela FVSBN.

#### Implementação em PyTorch

Aqui está uma implementação simplificada da FVSBN em PyTorch:

```python
import torch
import torch.nn as nn

class FVSBN(nn.Module):
    def __init__(self, n_variables):
        super(FVSBN, self).__init__()
        self.n_variables = n_variables
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.randn(i)) for i in range(1, n_variables + 1)
        ])
    
    def forward(self, x):
        log_probs = []
        for i in range(self.n_variables):
            if i == 0:
                prob = torch.sigmoid(self.alphas[i])
            else:
                prob = torch.sigmoid(torch.sum(self.alphas[i][:i] * x[:i]))
            log_prob = torch.where(x[i] == 1, torch.log(prob), torch.log(1 - prob))
            log_probs.append(log_prob)
        return torch.sum(torch.stack(log_probs))

    def sample(self):
        x = torch.zeros(self.n_variables)
        for i in range(self.n_variables):
            if i == 0:
                prob = torch.sigmoid(self.alphas[i])
            else:
                prob = torch.sigmoid(torch.sum(self.alphas[i][:i] * x[:i]))
            x[i] = torch.bernoulli(prob)
        return x
```

Esta implementação permite tanto a avaliação da log-probabilidade de uma amostra quanto a geração de novas amostras da distribuição modelada.

#### Questões Técnicas/Teóricas

1. Como a complexidade computacional da avaliação da probabilidade conjunta na FVSBN se compara com a de um modelo de Boltzmann restrito (RBM)?

2. Quais são as implicações da ordem escolhida para as variáveis na FVSBN em termos de modelagem e desempenho?

### Vantagens e Desvantagens da FVSBN

| 👍 Vantagens                                       | 👎 Desvantagens                                               |
| ------------------------------------------------- | ------------------------------------------------------------ |
| Modelagem direta da distribuição conjunta [1]     | Necessidade de especificar uma ordem para as variáveis [6]   |
| Amostragem eficiente via amostragem ancestral [7] | Potencial perda de informação devido à ordem fixa das variáveis |
| Avaliação exata da probabilidade conjunta [7]     | Escalabilidade limitada para um grande número de variáveis   |

### Extensões e Variantes

1. **NADE (Neural Autoregressive Distribution Estimation)**: Uma extensão da FVSBN que utiliza redes neurais mais complexas para modelar as distribuições condicionais, permitindo capturar dependências mais sofisticadas entre as variáveis [9].

2. **MADE (Masked Autoencoder for Distribution Estimation)**: Uma variante que utiliza mascaramento para criar uma rede neural que respeita a estrutura autoregressiva, permitindo computação paralela eficiente [10].

A formulação matemática do NADE é similar à FVSBN, mas com uma camada oculta adicional:

$$
p(x_i = 1|x_{<i}) = \sigma(V_i h_i + b_i)
$$

$$
h_i = \sigma(W_{<i} x_{<i} + c)
$$

Onde $W_{<i}$ representa as primeiras $i-1$ colunas de $W$, e $V_i$ é a $i$-ésima linha de $V$.

### Aplicações Práticas

A FVSBN e suas variantes têm sido aplicadas com sucesso em diversas áreas:

1. **Modelagem de Imagens**: Utilizada para modelar distribuições de pixels em imagens, permitindo a geração de novas imagens realistas [11].

2. **Processamento de Linguagem Natural**: Aplicada na modelagem de sequências de palavras ou caracteres, útil para tarefas como completamento de texto [12].

3. **Detecção de Anomalias**: A capacidade de avaliar probabilidades exatas torna a FVSBN útil para identificar amostras atípicas em conjuntos de dados [13].

> 💡 **Dica**: A FVSBN pode ser usada como um bloco de construção em modelos mais complexos, como redes profundas de crença (DBNs).

### Conclusão

A Rede de Crença Sigmóide Totalmente Visível (FVSBN) representa um marco importante no desenvolvimento de modelos probabilísticos para dados binários. Sua estrutura autoregressiva permite uma modelagem eficiente e interpretável de distribuições complexas, tornando-a uma ferramenta valiosa em diversos domínios de aprendizado de máquina e inteligência artificial.

A FVSBN serve como base para modelos mais avançados, como NADE e MADE, que estendem seus princípios para capturar dependências ainda mais complexas. Embora tenha limitações, como a necessidade de especificar uma ordem fixa para as variáveis, a FVSBN continua sendo um modelo relevante, especialmente em cenários onde a interpretabilidade e a avaliação exata de probabilidades são cruciais.

### Questões Avançadas

1. Como você poderia modificar a estrutura da FVSBN para permitir a modelagem de variáveis contínuas? Quais seriam os desafios e as possíveis abordagens para superar essas limitações?

2. Considere um cenário onde você precisa modelar uma distribuição sobre 1000 variáveis binárias. Compare e contraste o uso de uma FVSBN com outras abordagens, como Redes Neurais Variacionais ou Modelos de Fluxo Normalizado, em termos de expressividade, eficiência computacional e facilidade de treinamento.

3. Proponha uma arquitetura híbrida que combine os princípios da FVSBN com técnicas de atenção usadas em modelos de linguagem modernos. Como essa arquitetura poderia superar algumas das limitações da FVSBN original, especialmente em relação à ordem fixa das variáveis?

### Referências

[1] "We can pick an ordering of all the random variables, i.e., raster scan ordering of pixels from top-left (X1) to bottom-right (Xn=784)" (Trecho de cs236_lecture3.pdf)

[6] "p(xi|x1, · · · , xi−1) = σ(αi0 + Pi−1 j=1 αi j xj)" (Trecho de cs236_lecture3.pdf)

[7] "How to sample from p(x1, · · · , x784)? 1 Sample x1 ∼ p(x1) (np.random.choice([1,0],p=[ˆx1, 1 − ˆx1])) 2 Sample x2 ∼ p(x2 | x1 = x1) 3 Sample x3 ∼ p(x3 | x1 = x1, x2 = x2) · · ·" (Trecho de cs236_lecture3.pdf)

[9] "NADE: Neural Autoregressive Density Estimation To improve model: use one layer neural network instead of logistic regression" (Trecho de cs236_lecture3.pdf)

[10] "MADE: Masked Autoencoder for Distribution Estimation 1 Challenge: An autoencoder that is autoregressive (DAG structure) 2 Solution: use masks to disallow certain paths (Germain et al., 2015)." (Trecho de cs236_lecture3.pdf)

[11] "Results on downsampled ImageNet. Very slow: sequential likelihood evaluation." (Trecho de cs236_lecture3.pdf)

[12] "Train on Wikipedia. Then sample from the model:" (Trecho de cs236_lecture3.pdf)

[13] "Application in Adversarial Attacks and Anomaly detection Machine learning methods are vulnerable to adversarial examples Can we detect them?" (Trecho de cs236_lecture3.pdf)