## Maximização da Log-Verossimilhança Esperada em Modelos Generativos Profundos

![image-20240820085209837](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820085209837.png)

### Introdução

A maximização da log-verossimilhança esperada é um princípio fundamental no treinamento de modelos generativos profundos. Este método busca encontrar os parâmetros do modelo que melhor explicam os dados observados, maximizando a probabilidade de gerar esses dados sob o modelo aprendido. Neste resumo, exploraremos a relação íntima entre a log-verossimilhança esperada e a divergência de Kullback-Leibler (KL), bem como a aproximação empírica da log-verossimilhança utilizada na prática [1].

### Conceitos Fundamentais

| Conceito                         | Explicação                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Log-verossimilhança esperada** | A média do logaritmo da probabilidade que o modelo atribui aos dados, calculada sobre a distribuição verdadeira dos dados. [1] |
| **Divergência KL**               | Uma medida de dissimilaridade entre duas distribuições de probabilidade, frequentemente usada para comparar a distribuição do modelo com a distribuição verdadeira dos dados. [2] |
| **Aproximação empírica**         | Técnica que utiliza amostras finitas para estimar quantidades que envolvem expectativas sobre toda a distribuição de dados. [3] |

> ⚠️ **Nota Importante**: A maximização da log-verossimilhança esperada é equivalente à minimização da divergência KL entre a distribuição verdadeira dos dados e a distribuição do modelo.

### Relação com a Divergência KL

![image-20240820155008911](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820155008911.png)

A relação entre a log-verossimilhança esperada e a divergência KL é fundamental para entender o processo de aprendizagem em modelos generativos profundos. Vamos explorar esta relação matematicamente [2]:

Seja $P_{data}$ a distribuição verdadeira dos dados e $P_θ$ a distribuição do modelo parametrizado por $θ$. A divergência KL entre essas distribuições é dada por:

$$
D_{KL}(P_{data} || P_θ) = \mathbb{E}_{x \sim P_{data}}\left[\log \frac{P_{data}(x)}{P_θ(x)}\right]
$$

Expandindo esta expressão, obtemos:

$$
D_{KL}(P_{data} || P_θ) = \mathbb{E}_{x \sim P_{data}}[\log P_{data}(x)] - \mathbb{E}_{x \sim P_{data}}[\log P_θ(x)]
$$

O primeiro termo, $\mathbb{E}_{x \sim P_{data}}[\log P_{data}(x)]$, é a entropia da distribuição verdadeira dos dados, que é constante em relação aos parâmetros do modelo $θ$. O segundo termo, $-\mathbb{E}_{x \sim P_{data}}[\log P_θ(x)]$, é o negativo da log-verossimilhança esperada.

Portanto, minimizar a divergência KL é equivalente a maximizar a log-verossimilhança esperada:

$$
\arg\min_θ D_{KL}(P_{data} || P_θ) = \arg\max_θ \mathbb{E}_{x \sim P_{data}}[\log P_θ(x)]
$$

Esta equivalência fundamenta o uso da maximização da log-verossimilhança como critério de treinamento para modelos generativos [4].

> ✔️ **Ponto de Destaque**: A minimização da divergência KL leva o modelo a atribuir alta probabilidade às regiões onde a densidade de dados é alta, e baixa probabilidade onde a densidade é baixa.

#### Questões Técnicas/Teóricas

1. Como a assimetria da divergência KL afeta a escolha entre minimizar $D_{KL}(P_{data} || P_θ)$ versus $D_{KL}(P_θ || P_{data})$ no contexto de modelos generativos?
2. Dado um conjunto de dados binários, como você interpretaria uma mudança na log-verossimilhança esperada de -0.7 para -0.5 após o treinamento do modelo?

### Aproximação Empírica da Log-Verossimilhança

Na prática, não temos acesso à distribuição verdadeira $P_{data}$, mas apenas a um conjunto finito de amostras $\{x^{(1)}, ..., x^{(m)}\}$. Portanto, aproximamos a log-verossimilhança esperada usando a média empírica [3]:

$$
\mathbb{E}_{x \sim P_{data}}[\log P_θ(x)] \approx \frac{1}{m} \sum_{i=1}^m \log P_θ(x^{(i)})
$$

Esta aproximação é conhecida como log-verossimilhança empírica e é a função objetivo que efetivamente maximizamos durante o treinamento [5].

A qualidade desta aproximação depende do tamanho e da representatividade do conjunto de dados. Pelo teorema do limite central, a média amostral converge para a expectativa verdadeira à medida que o tamanho da amostra aumenta [6].

> ❗ **Ponto de Atenção**: A aproximação empírica pode levar ao overfitting se o conjunto de dados for pequeno ou não representativo da distribuição verdadeira.

Para mitigar o overfitting, técnicas de regularização são frequentemente empregadas, como:

1. Regularização L1/L2 nos parâmetros do modelo
2. Dropout
3. Data augmentation
4. Early stopping

A escolha da técnica de regularização depende da arquitetura específica do modelo e das características do problema [7].

#### Implementação em PyTorch

Aqui está um exemplo simplificado de como implementar a maximização da log-verossimilhança empírica para um modelo generativo em PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.network(z)
    
    def log_prob(self, x):
        # Assuming binary data
        return torch.sum(x * torch.log(self(z)) + (1 - x) * torch.log(1 - self(z)), dim=1)

model = GenerativeModel()
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in dataloader:
        z = torch.randn(batch.size(0), 100)
        log_probs = model.log_prob(batch)
        loss = -torch.mean(log_probs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Este código demonstra como calcular e otimizar a log-verossimilhança empírica para um modelo gerador simples de dados binários [8].

#### Questões Técnicas/Teóricas

1. Como você modificaria o código acima para lidar com dados contínuos em vez de binários?
2. Que precauções devem ser tomadas ao calcular $\log P_θ(x)$ numericamente para evitar problemas de estabilidade?

### Limitações e Considerações Práticas

Embora a maximização da log-verossimilhança seja teoricamente sólida, existem desafios práticos na sua aplicação a modelos generativos profundos:

1. **Intratabilidade**: Para muitos modelos complexos, calcular $P_θ(x)$ exatamente é intratável, exigindo aproximações ou técnicas de amostragem [9].

2. **Dimensionalidade Alta**: Em espaços de alta dimensão, a log-verossimilhança pode não ser uma medida confiável da qualidade do modelo, um fenômeno conhecido como "curse of dimensionality" [10].

3. **Modos Colapsados**: A maximização da log-verossimilhança pode levar a modelos que capturam apenas um subconjunto dos modos da distribuição verdadeira, um problema conhecido como "mode collapse" [11].

4. **Sensibilidade a Outliers**: A log-verossimilhança pode ser muito sensível a amostras atípicas, levando a modelos que atribuem probabilidade excessiva a regiões improváveis do espaço de dados [12].

Para abordar essas limitações, várias técnicas alternativas têm sido propostas:

- **Variational Autoencoders (VAEs)**: Maximizam um lower bound da log-verossimilhança, tornando o problema tratável para modelos complexos [13].

- **Generative Adversarial Networks (GANs)**: Evitam o cálculo direto da log-verossimilhança, usando um critério de treinamento baseado em discriminação [14].

- **Normalizing Flows**: Permitem o cálculo exato da log-verossimilhança para certos tipos de modelos, usando transformações invertíveis [15].

> 💡 **Insight**: A escolha entre maximizar a log-verossimilhança diretamente ou usar métodos alternativos depende do equilíbrio entre tratabilidade computacional, qualidade das amostras geradas e estabilidade do treinamento.

### Conclusão

A maximização da log-verossimilhança esperada é um princípio fundamental no treinamento de modelos generativos profundos, intimamente relacionado à minimização da divergência KL. Sua aproximação empírica fornece uma base prática para o treinamento, embora venha com desafios e limitações.

Compreender profundamente esses conceitos é crucial para desenvolver e aplicar efetivamente modelos generativos em diversos domínios, desde processamento de imagens até modelagem de linguagem natural. À medida que o campo evolui, novas técnicas continuam a ser desenvolvidas para superar as limitações das abordagens baseadas puramente em log-verossimilhança, expandindo as fronteiras do que é possível em aprendizado generativo profundo [16].

### Questões Avançadas

1. Como você abordaria o problema de avaliar a qualidade de um modelo generativo quando a log-verossimilhança não pode ser calculada diretamente? Discuta as vantagens e desvantagens de métricas alternativas.

2. Considere um cenário onde você está treinando um modelo generativo para imagens médicas raras. Como você lidaria com o trade-off entre maximizar a log-verossimilhança e evitar a geração de falsos positivos potencialmente perigosos?

3. Explique como o princípio da Informação Mútua Máxima (MaxMI) se relaciona com a maximização da log-verossimilhança em modelos generativos. Em que situações o MaxMI poderia ser preferível?

### Referências

[1] "We want to construct P_θ as "close" as possible to P_data (recall we assume we are given a dataset D of samples from P_data)" (Trecho de cs236_lecture4.pdf)

[2] "D(P_data||P_θ) = E_x∼P_data[log(P_data(x)/P_θ(x))] = E_x∼P_data[log P_data(x)] - E_x∼P_data[log P_θ(x)]" (Trecho de cs236_lecture4.pdf)

[3] "Approximate the expected log-likelihood E_x∼P_data[log P_θ(x)] with the empirical log-likelihood: E_D[log P_θ(x)] = (1/|D|) Σ_x∈D log P_θ(x)" (Trecho de cs236_lecture4.pdf)

[4] "Then, minimizing KL divergence is equivalent to maximizing the expected log-likelihood arg min_P_θ D(P_data||P_θ) = arg min_P_θ -E_x∼P_data[log P_θ(x)] = arg max_P_θ E_x∼P_data[log P_θ(x)]" (Trecho de cs236_lecture4.pdf)

[5] "Maximum likelihood learning is then: max_P_θ (1/|D|) Σ_x∈D log P_θ(x)" (Trecho de cs236_lecture4.pdf)

[6] "Convergence: By law of large numbers ĝ = (1/T) Σ_t=1^T g(x_t) → E_P[g(x)] for T → ∞" (Trecho de cs236_lecture4.pdf)

[7] "Generalization: the data is a sample, usually there is vast amount of samples that you have never seen. Your model should generalize well to these "never-seen" samples." (Trecho de cs236_lecture4.pdf)

[8] "Goal : maximize arg max_θ L(θ,D) = arg max_θ log L(θ,D)" (Trecho de cs236_lecture4.pdf)

[9] "Problem: In general we do not know P_data." (Trecho de cs236_lecture4.pdf)

[10] "Example. Suppose we represent each image with a vector X of 784 binary variables (black vs. white pixel). How many possible states (= possible images) in the model? 2^784 ≈ 10^236. Even 10^7 training examples provide extremely sparse coverage!" (Trecho de cs236_lecture4.pdf)

[11] "When we have small amount of data, multiple models can fit well, or even better than the true model. Moreover, small perturbations on D will result in very different estimates" (Trecho de cs236_lecture4.pdf)

[12] "If the hypothesis space is very limited, it might not be able to represent P_data, even with unlimited data" (Trecho de cs236_lecture4.pdf)

[13] "Soft preference for "simpler" models: Occam Razor." (Trecho de cs236_lecture4.pdf)

[14] "Evaluate generalization performance on a held-out validation set" (Trecho de cs236_lecture4.pdf)

[15] "Higher log-likelihood doesn't necessarily mean better looking samples" (Trecho de cs236_lecture4.pdf)

[16] "Other ways of measuring similarity are possible (Generative Adversarial Networks, GANs)" (Trecho de cs236_lecture4.pdf)