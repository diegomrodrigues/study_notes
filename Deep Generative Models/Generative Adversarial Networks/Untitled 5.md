## Two-Sample Tests: Comparing Distributions Through Sampling

<image: A statistical diagram showing two distinct sample distributions with overlapping tails, accompanied by test statistics and p-value representation>

### Introdução

**Two-sample tests** são ferramentas estatísticas fundamentais utilizadas para comparar duas populações ou distribuições com base em amostras independentes. Estes testes desempenham um papel crucial em várias áreas da ciência de dados e aprendizado de máquina, especialmente quando se trata de avaliar a eficácia de modelos generativos ou comparar distribuições de dados reais e sintéticos [1].

> 💡 **Destaque**: Two-sample tests são essenciais para avaliar se duas amostras provêm da mesma distribuição, um conceito fundamental em aprendizado não supervisionado e validação de modelos generativos.

### Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Hipótese Nula (H₀)**        | Afirma que não há diferença significativa entre as duas populações ou distribuições amostradas. [1] |
| **Hipótese Alternativa (H₁)** | Propõe que existe uma diferença significativa entre as duas populações ou distribuições. [1] |
| **Estatística de Teste**      | Uma medida quantitativa calculada a partir dos dados das amostras, usada para avaliar a evidência contra a hipótese nula. [2] |
| **Valor-p**                   | A probabilidade de obter um resultado tão ou mais extremo que o observado, assumindo que a hipótese nula é verdadeira. [2] |

### Formulação Matemática do Two-Sample Test

Em um contexto de aprendizado de máquina generativo, podemos formalizar o two-sample test da seguinte maneira [3]:

Dado $S_1 = \{x \sim P\}$ e $S_2 = \{x \sim Q\}$, onde $P$ e $Q$ são distribuições desconhecidas, queremos testar:

$$
H_0: P = Q \quad \text{vs} \quad H_1: P \neq Q
$$

A estatística de teste $T$ é calculada com base na diferença entre $S_1$ e $S_2$. Se $T < \alpha$, onde $\alpha$ é um limiar predefinido, aceitamos a hipótese nula de que $P = Q$.

> ⚠️ **Nota Importante**: A escolha do limiar $\alpha$ é crucial e afeta diretamente a taxa de erro Tipo I (falsos positivos) do teste.

### Aplicação em Modelos Generativos

No contexto de Generative Adversarial Networks (GANs) e outros modelos generativos, o two-sample test pode ser utilizado como um objetivo de treinamento [3]:

1. $S_1 = D = \{x \sim p_{data}\}$ (conjunto de treinamento)
2. $S_2 = \{x \sim p_\theta\}$ (amostras geradas pelo modelo)

O objetivo é treinar o modelo para minimizar uma métrica de two-sample test entre $S_1$ e $S_2$, efetivamente aproximando a distribuição gerada $p_\theta$ da distribuição real de dados $p_{data}$.

#### Questões Técnicas/Teóricas

1. Como a escolha do tamanho da amostra afeta o poder estatístico de um two-sample test em um contexto de validação de modelos generativos?
2. Descreva como você implementaria um two-sample test baseado em MMD (Maximum Mean Discrepancy) para avaliar a qualidade de amostras geradas por uma GAN.

### Estatísticas de Teste Comuns

Várias estatísticas de teste podem ser empregadas em two-sample tests, dependendo das características dos dados e das suposições sobre as distribuições subjacentes [4]:

1. **Diferença de Médias**:
   Para dados contínuos, assumindo normalidade:
   
   $$T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
   
   onde $\bar{X}_i$, $s_i^2$, e $n_i$ são a média amostral, variância amostral e tamanho da amostra para o grupo $i$, respectivamente.

2. **Teste de Kolmogorov-Smirnov**:
   Para distribuições contínuas sem suposições de normalidade:
   
   $$D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$$
   
   onde $F_{i,n}(x)$ é a função de distribuição cumulativa empírica para a amostra $i$.

3. **Maximum Mean Discrepancy (MMD)**:
   Particularmente útil em aprendizado de máquina:
   
   $$\text{MMD}^2(F, X, Y) = \frac{1}{n(n-1)} \sum_{i\neq j} k(x_i, x_j) + \frac{1}{m(m-1)} \sum_{i\neq j} k(y_i, y_j) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j)$$
   
   onde $k(\cdot,\cdot)$ é uma função kernel.

> ✔️ **Destaque**: A escolha da estatística de teste deve ser baseada nas características dos dados e nas suposições sobre as distribuições subjacentes. MMD é particularmente útil em contextos de aprendizado de máquina devido à sua capacidade de capturar diferenças em momentos de ordem superior.

### Interpretação dos Resultados

A interpretação dos resultados de um two-sample test envolve a análise do valor-p em relação ao nível de significância escolhido (geralmente 0,05) [5]:

- Se $p < 0,05$, rejeitamos $H_0$, indicando evidência estatística de uma diferença significativa entre as distribuições.
- Se $p \geq 0,05$, não rejeitamos $H_0$, sugerindo que não há evidência suficiente para concluir que as distribuições são diferentes.

> ❗ **Ponto de Atenção**: Um valor-p alto não prova que as distribuições são idênticas, apenas que não há evidência suficiente para concluir que são diferentes.

#### Questões Técnicas/Teóricas

1. Em um cenário de avaliação de um modelo GAN, como você interpretaria um valor-p de 0,07 em um two-sample test comparando amostras reais e geradas?
2. Discuta as vantagens e desvantagens de usar o MMD como estatística de teste em comparação com testes paramétricos tradicionais no contexto de avaliação de modelos generativos.

### Aplicações em Deep Learning e Modelos Generativos

Two-sample tests têm aplicações cruciais em deep learning e modelos generativos [6]:

1. **Avaliação de GANs**: Comparar a distribuição de amostras geradas com a distribuição de dados reais.
2. **Detecção de Drift**: Identificar mudanças na distribuição de dados ao longo do tempo em sistemas de ML em produção.
3. **Validação de Augmentação de Dados**: Verificar se dados aumentados mantêm as propriedades estatísticas dos dados originais.
4. **Teste A/B em ML**: Comparar o desempenho de diferentes versões de modelos em ambientes de produção.

```python
import torch
from torch import nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
```

Este código implementa uma classe `MMDLoss` em PyTorch que calcula a Maximum Mean Discrepancy entre duas distribuições, frequentemente usada como uma métrica de two-sample test em deep learning.

### Conclusão

Two-sample tests são ferramentas estatísticas poderosas com aplicações significativas em aprendizado de máquina e ciência de dados, especialmente no contexto de modelos generativos. Eles fornecem um framework rigoroso para comparar distribuições, essencial para validar a qualidade de amostras geradas e detectar mudanças em distribuições de dados. A escolha apropriada da estatística de teste e a interpretação cuidadosa dos resultados são cruciais para aplicações bem-sucedidas em cenários do mundo real.

### Questões Avançadas

1. Como você projetaria um experimento para comparar a eficácia de diferentes estatísticas de two-sample test (e.g., t-test, KS-test, MMD) na avaliação de modelos GAN para diferentes tipos de dados (e.g., imagens, texto, séries temporais)?

2. Discuta as implicações teóricas e práticas de usar two-sample tests como objetivos de treinamento em modelos generativos. Como isso se compara com abordagens tradicionais baseadas em verossimilhança?

3. Em um cenário de aprendizado federado, onde dados estão distribuídos em múltiplos dispositivos, como você adaptaria two-sample tests para avaliar a qualidade global do modelo sem comprometer a privacidade dos dados?

### Referências

[1] "The generator and discriminator both play a two-player minimax game, where the generator minimizes a two-sample test objective (pdata = pθ) and the discriminator maximizes the objective (pdata ≠ pθ)." (Excerpt from Stanford Notes)

[2] "Concretely, given S1 = {x ∼ P} and S2 = {x ∼ Q}, we compute a test statistic T according to the difference in S1 and S2 that, when less than a threshold α, accepts the null hypothesis that P = Q." (Excerpt from Stanford Notes)

[3] "Analogously, we have in our generative modeling setup access to our training set S1 = D = {x ∼ pdata} and S2 = {x ∼ pθ}. The key idea is to train the model to minimize a two-sample test objective between S1 and S2." (Excerpt from Stanford Notes)

[4] "Consider pathological cases in which our model is comprised almost entirely of noise, or our model simply memorizes the training set." (Excerpt from Stanford Notes)

[5] "Therefore we can choose any f-divergence that we desire, let p = pdata and q = pG, parameterize T by ϕ and G by θ, and obtain the following fGAN objective:" (Excerpt from Stanford Notes)

[6] "GANs are unique from all the other model families that we have seen so far, such as autoregressive models, VAEs, and normalizing flow models, because we do not train them using maximum likelihood." (Excerpt from Stanford Notes)