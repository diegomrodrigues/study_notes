## Limitações da Verossimilhança como Indicador de Qualidade

<image: Um gráfico de dispersão mostrando a relação não linear entre log-likelihood e qualidade de amostras, com pontos representando diferentes modelos generativos>

### Introdução

A verossimilhança tem sido tradicionalmente utilizada como uma métrica fundamental na avaliação de modelos generativos. No entanto, pesquisas recentes têm revelado limitações significativas nessa abordagem, especialmente quando se trata de avaliar a qualidade das amostras geradas. Este estudo aprofundado explora os casos em que altas verossimilhanças não correspondem necessariamente a amostras de alta qualidade e vice-versa, motivando assim a exploração de alternativas livre de verossimilhança (likelihood-free) [1].

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Verossimilhança**          | Medida estatística que quantifica quão bem um modelo explica os dados observados. Em modelos generativos, é frequentemente usada para avaliar o desempenho do modelo [1]. |
| **Qualidade de Amostra**     | Refere-se à fidelidade e realismo das amostras geradas por um modelo generativo em relação aos dados reais [1]. |
| **Likelihood-free Learning** | Abordagem de treinamento que não depende diretamente da avaliação da verossimilhança, mas utiliza outras métricas ou objetivos para otimizar o modelo [1]. |

> ⚠️ **Nota Importante**: A relação entre verossimilhança e qualidade de amostra não é sempre direta ou linear, o que pode levar a avaliações enganosas de modelos generativos [1].

### Casos de Divergência entre Verossimilhança e Qualidade de Amostra

<image: Dois gráficos lado a lado: um mostrando um modelo de mistura de ruído com alta verossimilhança e outro mostrando um modelo que memorizou o conjunto de treinamento>

#### Modelo de Mistura de Ruído

Um caso clássico onde a alta verossimilhança não corresponde à alta qualidade de amostra é o modelo de mistura de ruído [1]. 

Considere um modelo generativo $p_\theta(x)$ definido como:

$$
p_\theta(x) = (1-\epsilon)p_{data}(x) + \epsilon \mathcal{N}(x|0,\sigma^2I)
$$

Onde:
- $p_{data}(x)$ é a distribuição real dos dados
- $\mathcal{N}(x|0,\sigma^2I)$ é uma distribuição Gaussiana com média zero e variância $\sigma^2$
- $\epsilon$ é um pequeno valor positivo

Este modelo tem uma alta verossimilhança para os dados de treinamento, pois atribui uma probabilidade não nula a todos os pontos de dados. No entanto, as amostras geradas por este modelo serão em grande parte ruído, não refletindo a qualidade desejada [1].

> ❗ **Ponto de Atenção**: Modelos que atribuem uma pequena probabilidade de ruído podem artificialmente aumentar sua verossimilhança sem melhorar a qualidade das amostras geradas [1].

#### Memorização do Conjunto de Treinamento

Outro caso problemático ocorre quando um modelo simplesmente memoriza o conjunto de treinamento [1]. Considere um modelo definido como:

$$
p_\theta(x) = \frac{1}{N}\sum_{i=1}^N \delta(x - x_i)
$$

Onde:
- $N$ é o número de amostras no conjunto de treinamento
- $x_i$ são as amostras do conjunto de treinamento
- $\delta(x)$ é a função delta de Dirac

Este modelo terá uma verossimilhança perfeita no conjunto de treinamento, mas não generalizará para novos dados e não será capaz de gerar amostras diversas e realistas [1].

> ✔️ **Destaque**: A capacidade de generalização e a diversidade das amostras geradas são aspectos cruciais que não são capturados diretamente pela verossimilhança [1].

### Implicações Teóricas

A divergência entre verossimilhança e qualidade de amostra tem implicações profundas para a teoria de aprendizado de máquina e estatística [1]. 

Considere a decomposição da log-verossimilhança negativa esperada:

$$
\mathbb{E}_{x\sim p_{data}}[-\log p_\theta(x)] = KL(p_{data}||p_\theta) + H(p_{data})
$$

Onde:
- $KL(p_{data}||p_\theta)$ é a divergência de Kullback-Leibler entre a distribuição real e o modelo
- $H(p_{data})$ é a entropia da distribuição real

Esta decomposição mostra que minimizar a log-verossimilhança negativa é equivalente a minimizar a divergência KL. No entanto, a divergência KL tem propriedades que podem não ser ideais para capturar a qualidade visual ou semântica das amostras geradas [1].

> 💡 **Insight**: A divergência KL é assimétrica e pode levar a comportamentos indesejados, como a priorização de cobertura sobre qualidade em certas regiões do espaço de dados [1].

#### Perguntas Técnicas/Teóricas

1. Como a assimetria da divergência KL pode afetar o treinamento de modelos generativos baseados em verossimilhança?
2. Proponha uma métrica alternativa que poderia capturar melhor a qualidade visual das amostras geradas por um modelo generativo.

### Abordagens Alternativas

Dado as limitações da verossimilhança como indicador de qualidade, pesquisadores têm explorado abordagens alternativas para avaliar e treinar modelos generativos [1].

#### Two-Sample Test

Uma abordagem promissora é o uso do two-sample test, um teste estatístico que determina se dois conjuntos finitos de amostras são provenientes da mesma distribuição [1]. 

Formalmente, dado $S_1 = \{x \sim P\}$ e $S_2 = \{x \sim Q\}$, calculamos uma estatística de teste $T$ baseada na diferença entre $S_1$ e $S_2$. Se $T < \alpha$ para um limiar $\alpha$ predefinido, aceitamos a hipótese nula de que $P = Q$ [1].

Esta abordagem pode ser adaptada para o contexto de modelos generativos, onde:
- $S_1 = D = \{x \sim p_{data}\}$ (conjunto de treinamento)
- $S_2 = \{x \sim p_\theta\}$ (amostras geradas pelo modelo)

> ✔️ **Destaque**: O two-sample test oferece uma forma de comparar diretamente as distribuições empíricas dos dados reais e gerados, potencialmente capturando aspectos de qualidade que a verossimilhança ignora [1].

#### Redes Adversariais Generativas (GANs)

As GANs representam uma abordagem revolucionária que abandona completamente o uso direto da verossimilhança [1]. Em vez disso, elas formulam o problema de aprendizado como um jogo adversarial entre duas redes neurais:

1. **Gerador** $G_\theta$: Mapeia ruído $z$ para amostras $x$
2. **Discriminador** $D_\phi$: Tenta distinguir entre amostras reais e geradas

O objetivo é minimizar uma função de perda adversarial:

$$
\min_\theta \max_\phi V(G_\theta, D_\phi) = \mathbb{E}_{x\sim p_{data}}[\log D_\phi(x)] + \mathbb{E}_{z\sim p(z)}[\log(1 - D_\phi(G_\theta(z)))]
$$

> ❗ **Ponto de Atenção**: Embora as GANs tenham demonstrado resultados impressionantes em termos de qualidade de amostra, elas apresentam desafios próprios, como instabilidade de treinamento e modo de colapso [1].

#### Perguntas Técnicas/Teóricas

1. Compare as vantagens e desvantagens do uso de GANs versus modelos baseados em verossimilhança para geração de imagens de alta resolução.
2. Como você poderia combinar ideias do two-sample test com o framework GAN para melhorar a avaliação de modelos generativos?

### Conclusão

A exploração das limitações da verossimilhança como indicador de qualidade em modelos generativos revela a complexidade inerente à avaliação desses modelos [1]. Casos como modelos de mistura de ruído e memorização do conjunto de treinamento demonstram claramente que altas verossimilhanças nem sempre correspondem a amostras de alta qualidade [1].

Essa constatação motiva a busca por abordagens alternativas, como métodos baseados em two-sample tests e redes adversariais generativas, que buscam capturar aspectos mais relevantes da qualidade das amostras geradas [1]. No entanto, cada abordagem traz seus próprios desafios e limitações, indicando que a avaliação e treinamento de modelos generativos continua sendo um campo ativo e crucial de pesquisa [1].

À medida que avançamos, é provável que vejamos o desenvolvimento de métodos híbridos que combinam insights de diferentes abordagens, buscando um equilíbrio entre a solidez teórica da verossimilhança e a capacidade de capturar aspectos qualitativos importantes das amostras geradas [1].

### Perguntas Avançadas

1. Desenhe um experimento que possa quantificar empiricamente a relação entre verossimilhança e qualidade percebida de amostras em diferentes domínios (por exemplo, imagens, texto e áudio).

2. Considere um cenário onde você tem acesso a um oráculo perfeito que pode julgar a qualidade de amostras geradas. Como você integraria esse oráculo no processo de treinamento de um modelo generativo sem recorrer à verossimilhança?

3. Discuta as implicações éticas e práticas de usar modelos generativos que priorizam a qualidade percebida das amostras sobre a fidelidade estatística à distribuição de dados original.

4. Proponha uma nova métrica que combine aspectos da verossimilhança com medidas de qualidade baseadas em percepção humana. Como você validaria empiricamente a eficácia dessa métrica?

5. Analise criticamente o papel da verossimilhança na teoria da informação e discuta se os recentes avanços em modelos generativos sugerem a necessidade de uma revisão fundamental de alguns princípios da teoria da informação.

### Referências

[1] "We now move onto another family of generative models called generative adversarial networks (GANs). GANs are unique from all the other model families that we have seen so far, such as autoregressive models, VAEs, and normalizing flow models, because we do not train them using maximum likelihood. Why not? In fact, it is not so clear that better likelihood numbers necessarily correspond to higher sample quality. We know that the optimal generative model will give us the best sample quality and highest test log-likelihood. However, models with high test log-likelihoods can still yield poor samples, and vice versa. To see why, consider pathological cases in which our model is comprised almost entirely of noise, or our model simply memorizes the training set. Therefore, we turn to likelihood-free training with the hope that optimizing a different objective will allow us to disentangle our desiderata of obtaining high likelihoods as well as high-quality samples." (Excerpt from Stanford Notes)