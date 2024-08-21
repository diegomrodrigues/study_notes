## Aprendizagem de Modelos Generativos: Aproximando Distribuições de Dados

![image-20240820084315415](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240820084315415.png)

### Introdução

O objetivo fundamental da aprendizagem de modelos generativos é construir uma representação matemática $P_\theta$ que capture com precisão a distribuição subjacente $P_{data}$ da qual nossos dados foram amostrados [1]. Este problema é central em aprendizado de máquina, estatística e ciência de dados, pois um modelo que captura adequadamente a distribuição de dados permite não apenas gerar novas amostras realistas, mas também realizar inferências, detectar anomalias e compreender a estrutura intrínseca dos dados.

No entanto, este objetivo ambicioso enfrenta desafios significativos que tornam sua realização perfeita praticamente impossível na maioria dos cenários do mundo real [2]. Estes desafios surgem principalmente de duas fontes:

1. **Limitações de dados**: Conjuntos de dados finitos fornecem apenas uma aproximação da verdadeira distribuição subjacente.
2. **Restrições computacionais**: A complexidade dos modelos e algoritmos é limitada por recursos computacionais finitos.

Para ilustrar a magnitude deste desafio, consideremos um exemplo concreto do domínio do processamento de imagens [3]:

> ⚠️ **Exemplo Ilustrativo**: Suponha que representemos cada imagem como um vetor $X$ de 784 variáveis binárias (pixels pretos ou brancos). O número de estados possíveis (ou seja, imagens possíveis) neste modelo é $2^{784} \approx 10^{236}$. Mesmo com $10^7$ exemplos de treinamento, temos uma cobertura extremamente esparsa do espaço de possibilidades.

Este exemplo destaca a "maldição da dimensionalidade" que permeia muitos problemas de aprendizado de máquina, especialmente em domínios de alta dimensão como processamento de imagens, áudio e linguagem natural.

Dado que uma representação perfeita é geralmente inatingível, nosso objetivo se torna selecionar $P_\theta$ para construir a "melhor" aproximação da distribuição subjacente $P_{data}$ [4]. Isto imediatamente levanta a questão crucial: o que define "melhor" neste contexto?

### Conceitos Fundamentais

| Conceito                               | Explicação                                                   |
| -------------------------------------- | ------------------------------------------------------------ |
| **Distribuição de Dados ($P_{data}$)** | A verdadeira distribuição de probabilidade subjacente da qual os dados são amostrados. Geralmente desconhecida e apenas aproximada pelos dados observados. [1] |
| **Modelo Generativo ($P_\theta$)**     | Uma representação parametrizada da distribuição que tentamos aprender. $\theta$ representa os parâmetros do modelo. [1] |
| **Maldição da Dimensionalidade**       | O fenômeno pelo qual o número de configurações possíveis cresce exponencialmente com a dimensionalidade do espaço, tornando a amostragem esparsa em altas dimensões. [3] |

> ✔️ **Ponto de Destaque**: A busca pela "melhor" aproximação é fundamentalmente um problema de otimização, onde a definição de "melhor" determina a função objetivo e, consequentemente, as propriedades do modelo aprendido.

### Critérios de Otimalidade para Modelos Generativos

A escolha do critério de "melhor" é crucial e depende do objetivo final do modelo. Vamos explorar algumas abordagens comuns:

#### 1. Divergência KL (Kullback-Leibler)

A divergência KL é uma medida assimétrica da diferença entre duas distribuições de probabilidade:

$$
D_{KL}(P_{data} || P_\theta) = \mathbb{E}_{x \sim P_{data}} \left[ \log \frac{P_{data}(x)}{P_\theta(x)} \right]
$$

Minimizar a divergência KL é equivalente a maximizar a log-verossimilhança esperada:

$$
\arg\min_\theta D_{KL}(P_{data} || P_\theta) = \arg\max_\theta \mathbb{E}_{x \sim P_{data}}[\log P_\theta(x)]
$$

> ❗ **Ponto de Atenção**: A divergência KL não é simétrica, ou seja, $D_{KL}(P_{data} || P_\theta) \neq D_{KL}(P_\theta || P_{data})$. A escolha da ordem afeta significativamente o comportamento do modelo aprendido.

#### 2. Máxima Verossimilhança

Na prática, não temos acesso à verdadeira $P_{data}$, mas apenas a um conjunto finito de amostras $\mathcal{D} = \{x^{(1)}, ..., x^{(m)}\}$. Isto leva ao princípio da máxima verossimilhança:

$$
\theta^* = \arg\max_\theta \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \log P_\theta(x)
$$

Esta abordagem é equivalente a minimizar a divergência KL empírica entre a distribuição empírica dos dados e o modelo.

#### 3. Divergência de Jensen-Shannon

A divergência de Jensen-Shannon (JS) é uma versão simétrica da divergência KL:

$$
D_{JS}(P_{data} || P_\theta) = \frac{1}{2}D_{KL}(P_{data} || M) + \frac{1}{2}D_{KL}(P_\theta || M)
$$

onde $M = \frac{1}{2}(P_{data} + P_\theta)$

Esta métrica é particularmente relevante no contexto de Redes Adversariais Generativas (GANs).

#### 4. Distância de Wasserstein

A distância de Wasserstein, também conhecida como "Earth Mover's Distance", oferece uma alternativa robusta, especialmente útil quando as distribuições têm suporte disjunto:

$$
W(P_{data}, P_\theta) = \inf_{\gamma \in \Pi(P_{data}, P_\theta)} \mathbb{E}_{(x,y)\sim \gamma}[||x-y||]
$$

onde $\Pi(P_{data}, P_\theta)$ é o conjunto de todas as distribuições conjuntas com marginais $P_{data}$ e $P_\theta$.

> 💡 **Insight**: A distância de Wasserstein pode fornecer gradientes úteis mesmo quando as distribuições não se sobrepõem, o que é particularmente valioso nas fases iniciais do treinamento de modelos generativos.

#### Questões Técnicas/Teóricas

1. Como a escolha entre minimizar $D_{KL}(P_{data} || P_\theta)$ versus $D_{KL}(P_\theta || P_{data})$ afeta o comportamento do modelo aprendido, especialmente em regiões de baixa densidade de dados?

2. Em um cenário de aprendizagem de modelo generativo para imagens, como você justificaria a escolha entre usar a divergência KL e a distância de Wasserstein como função objetivo?

### Desafios na Aprendizagem de Modelos Generativos

A aprendizagem de modelos generativos enfrenta vários desafios fundamentais:

#### 1. Esparsidade de Dados em Altas Dimensões

Como ilustrado no exemplo inicial, em espaços de alta dimensão, mesmo grandes conjuntos de dados cobrem apenas uma fração minúscula do espaço de possibilidades. Isso leva a desafios na generalização e na captura de estruturas de baixa dimensionalidade em dados de alta dimensão.

**Formalização Matemática**: 
Seja $V_d(r)$ o volume de uma esfera d-dimensional de raio r. A fração do volume da esfera unitária contida em uma casca $\epsilon$ perto da superfície é dada por:

$$
\frac{V_d(1) - V_d(1-\epsilon)}{V_d(1)} = 1 - (1-\epsilon)^d \approx 1 - e^{-d\epsilon}
$$

Para grandes $d$, esta fração se aproxima de 1 mesmo para $\epsilon$ pequeno, ilustrando como a maioria do volume em altas dimensões está concentrada próximo à superfície.

#### 2. Modos Colapsados e Diversidade

Modelos generativos frequentemente sofrem do problema de "mode collapse", onde falham em capturar a diversidade completa da distribuição de dados.

**Exemplo Formal**: Considere um modelo $P_\theta$ treinado para minimizar $D_{KL}(P_{data} || P_\theta)$. Se $P_{data}$ tem múltiplos modos, $P_\theta$ pode concentrar toda sua massa em um único modo para minimizar a penalidade de atribuir baixa probabilidade a qualquer região de suporte de $P_{data}$.

#### 3. Avaliação de Modelos

A avaliação de modelos generativos é notoriamente difícil, pois envolve comparar distribuições de alta dimensão.

**Métrica de Avaliação**: O Inception Score (IS) para avaliação de modelos generativos de imagens é definido como:

$$
IS = \exp(\mathbb{E}_{x \sim P_\theta}[D_{KL}(p(y|x) || p(y))])
$$

onde $p(y|x)$ é a distribuição de classes predita por um classificador pré-treinado para a imagem gerada $x$, e $p(y)$ é a distribuição marginal sobre as classes.

> ⚠️ **Nota Importante**: Métricas como IS capturam apenas aspectos específicos da qualidade do modelo e podem ser enganosas se usadas isoladamente.

#### 4. Otimização Não-Convexa

A maioria dos modelos generativos modernos, especialmente aqueles baseados em redes neurais profundas, envolvem otimização de funções altamente não-convexas.

**Landscape de Otimização**: Para um modelo neural com parâmetros $\theta$, a função de perda $L(\theta)$ tipicamente tem múltiplos mínimos locais. A dinâmica de otimização pode ser aproximada por:

$$
\frac{d\theta}{dt} = -\nabla L(\theta) + \eta(t)
$$

onde $\eta(t)$ representa ruído estocástico no processo de otimização.

#### Questões Técnicas/Teóricas

1. Como o fenômeno de concentração de medida em altas dimensões afeta a capacidade de modelos generativos de capturar efetivamente a distribuição de dados reais?

2. Proponha uma abordagem para mitigar o problema de mode collapse em um modelo generativo baseado em GAN, considerando as propriedades da divergência JS.

### Técnicas Avançadas de Aprendizagem para Modelos Generativos

Para abordar os desafios mencionados, várias técnicas avançadas foram desenvolvidas:

#### 1. Variational Autoencoders (VAEs)

VAEs introduzem uma abordagem baseada em inferência variacional para aprendizagem de modelos generativos.

**Formulação Matemática**:
O objetivo de treinamento de um VAE é maximizar um lower bound (ELBO) na log-verossimilhança:
$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$

onde $q_\phi(z|x)$ é o encoder (aproximação variacional), $p_\theta(x|z)$ é o decoder, e $p(z)$ é a prior sobre o espaço latente.

> ✔️ **Ponto de Destaque**: VAEs permitem inferência eficiente e geração de amostras, mas podem produzir amostras borradas devido à natureza da divergência KL no espaço latente.

#### 2. Generative Adversarial Networks (GANs)

GANs formulam o problema de aprendizagem generativa como um jogo de soma zero entre um gerador e um discriminador.

**Formulação do Jogo**:
O objetivo é encontrar um equilíbrio de Nash no seguinte jogo de min-max:

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1-D(G(z)))]
$$

onde $G$ é o gerador e $D$ é o discriminador.

> ❗ **Ponto de Atenção**: O treinamento de GANs pode ser instável devido à natureza adversarial do processo de otimização.

#### 3. Fluxos Normalizadores

Fluxos normalizadores utilizam transformações invertíveis para mapear entre uma distribuição simples e a distribuição de dados complexa.

**Formalização**:
Seja $f$ uma transformação invertível. A mudança de variáveis fornece:

$$
\log p_X(x) = \log p_Z(f^{-1}(x)) + \log \left|\det \frac{\partial f^{-1}}{\partial x}\right|
$$

onde $p_Z$ é uma distribuição base simples (e.g., Gaussiana) e $p_X$ é a distribuição modelada.

> 💡 **Insight**: Fluxos normalizadores permitem cálculo exato da verossimilhança, mas requerem arquiteturas especiais para manter a invertibilidade e o cálculo eficiente do determinante Jacobiano.

#### 4. Diffusion Models

Modelos de difusão definem um processo de Markov forward que gradualmente adiciona ruído aos dados, e então aprendem o processo reverso.

**Processo de Difusão**:
O processo forward é definido como:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

O modelo aprende a reverter este processo, maximizando:

$$
\mathbb{E}_{q(x_{0:T})}[\log p(x_T) + \sum_{t>1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}]
$$

> ✔️ **Ponto de Destaque**: Modelos de difusão têm mostrado resultados impressionantes em geração de imagens de alta qualidade, combinando a tratabilidade dos VAEs com a qualidade das GANs.

#### Questões Técnicas/Teóricas

1. Compare e contraste as implicações teóricas de usar a divergência KL reversa (como em VAEs) versus 

2. a divergência KL direta (implícita em muitas GANs) para o treinamento de modelos generativos. Como essas escolhas afetam o comportamento do modelo em regiões de baixa densidade de dados?

   2. Descreva como você poderia combinar aspectos de fluxos normalizadores e modelos de difusão para criar um modelo generativo que aproveite as vantagens de ambas as abordagens.


### Estratégias de Otimização para Modelos Generativos

A otimização de modelos generativos apresenta desafios únicos devido à natureza de alta dimensionalidade e não-convexidade do problema. Vamos explorar algumas estratégias avançadas:

#### 1. Descida do Gradiente Estocástico (SGD) com Momentum

Para modelos baseados em máxima verossimilhança, como VAEs e fluxos normalizadores, SGD com momentum é frequentemente utilizado.

**Atualização de Parâmetros**:

$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta \nabla_\theta L(\theta_{t-1}) \\
\theta_t &= \theta_{t-1} - v_t
\end{aligned}
$$

onde $\gamma$ é o coeficiente de momentum e $\eta$ é a taxa de aprendizado.

> ✔️ **Ponto de Destaque**: Momentum ajuda a superar mínimos locais rasos e acelera a convergência em ravinas.

#### 2. Otimização Alternada para GANs

GANs requerem uma abordagem de otimização alternada devido à sua natureza adversarial.

**Algoritmo**:
1. Fixe G, atualize D: $\theta_D \leftarrow \theta_D + \eta_D \nabla_{\theta_D} V(D,G)$
2. Fixe D, atualize G: $\theta_G \leftarrow \theta_G - \eta_G \nabla_{\theta_G} V(D,G)$

> ❗ **Ponto de Atenção**: O equilíbrio entre as atualizações de G e D é crucial. Atualizações muito frequentes de D podem levar a overfitting local.

#### 3. Técnicas de Regularização

Para combater overfitting e melhorar a estabilidade, várias técnicas de regularização são empregadas:

a) **Spectral Normalization**: Normaliza os pesos das camadas para controlar o Lipschitz constante do discriminador:

$$
W_{SN} = W / \sigma(W)
$$

onde $\sigma(W)$ é o maior valor singular de W.

b) **Gradient Penalty**: Adiciona um termo de penalidade ao objetivo do discriminador:

$$
L_D = V(D,G) + \lambda \mathbb{E}_{\hat{x}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]
$$

onde $\hat{x}$ são amostras interpoladas entre dados reais e gerados.

#### 4. Adaptive Learning Rates

Algoritmos como Adam combinam as vantagens de RMSprop e momentum:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1-\beta_1^t) \\
\hat{v}_t &= v_t / (1-\beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{aligned}
$$

onde $g_t$ é o gradiente no tempo t.

> 💡 **Insight**: Adam adapta as taxas de aprendizado para cada parâmetro, o que é particularmente útil em landscapes de otimização complexos típicos de modelos generativos.

#### 5. Curriculum Learning

Introduz gradualmente a complexidade da tarefa durante o treinamento.

**Exemplo para GANs**: Comece gerando imagens de baixa resolução e gradualmente aumente a resolução:

$$
L_G(t) = \mathbb{E}_{z \sim p(z)}[-\log D(G(z, r(t)))]
$$

onde $r(t)$ é uma função que aumenta a resolução ao longo do tempo t.

#### Questões Técnicas/Teóricas

1. Como a escolha entre otimização simultânea versus alternada afeta a convergência em modelos adversariais? Discuta as implicações teóricas e práticas.

2. Proponha e justifique uma estratégia de curriculum learning para treinar um modelo de difusão para geração de imagens de alta resolução.

### Avaliação e Diagnóstico de Modelos Generativos

A avaliação de modelos generativos é notoriamente desafiadora devido à natureza de alta dimensionalidade e multimodalidade das distribuições envolvidas.

#### 1. Métricas Baseadas em Verossimilhança

Para modelos que permitem cálculo direto da verossimilhança (e.g., VAEs, fluxos):

a) **Negative Log-Likelihood (NLL)**:
$$
NLL = -\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)
$$

b) **Bits per Dimension (para dados de imagem)**:
$$
BPD = -\frac{1}{N \cdot D} \sum_{i=1}^N \log_2 p_\theta(x_i)
$$
onde D é o número de dimensões (e.g., pixels).

> ⚠️ **Nota Importante**: NLL pode ser enganoso para comparar modelos com diferentes arquiteturas ou domínios de dados.

#### 2. Métricas Baseadas em Amostras

Para modelos onde o cálculo direto da verossimilhança não é possível (e.g., GANs):

a) **Inception Score (IS)**:
$$
IS = \exp(\mathbb{E}_{x \sim p_g} D_{KL}(p(y|x) || p(y)))
$$

b) **Fréchet Inception Distance (FID)**:
$$
FID = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
$$
onde $(\mu_r, \Sigma_r)$ e $(\mu_g, \Sigma_g)$ são a média e covariância das features de Inception para dados reais e gerados, respectivamente.

> ✔️ **Ponto de Destaque**: FID é mais robusto que IS e correlaciona melhor com a qualidade perceptual humana.

#### 3. Métricas de Diversidade

a) **Birthday Paradox Test**: Gera N amostras e verifica duplicatas. O número de amostras necessário para encontrar uma duplicata é indicativo da diversidade.

b) **Diversidade Pluralística**: Para modelos condicionais, mede a diversidade das saídas para uma única entrada:
$$
D_{plural} = \frac{1}{M} \sum_{i=1}^M \min_{j \neq i} d(x_i, x_j)
$$
onde $d$ é uma métrica de distância e $x_i$ são M amostras para uma única condição.

#### 4. Diagnóstico de Mode Collapse

a) **Cobertura de Modos**: Em datasets sintéticos com modos conhecidos, mede a fração de modos capturados pelo modelo.

b) **Análise de Componentes Principais (PCA)**: Compara a distribuição dos componentes principais entre dados reais e gerados.

#### 5. Avaliação Humana

Especialmente importante para domínios perceptuais como imagens e áudio.

a) **Comparação Lado a Lado**: Avaliadores humanos comparam amostras reais e geradas.

b) **Turing Test Generativo**: Avaliadores tentam distinguir entre amostras reais e geradas.

> ❗ **Ponto de Atenção**: Avaliações humanas são custosas e podem ser inconsistentes, mas fornecem insights valiosos sobre qualidade perceptual.

#### Questões Técnicas/Teóricas

1. Como você abordaria a avaliação de um modelo generativo em um domínio onde não existe um conjunto de teste bem definido (por exemplo, geração de moléculas para descoberta de drogas)?

2. Proponha uma nova métrica que combine aspectos de avaliação baseada em verossimilhança e baseada em amostras. Discuta suas potenciais vantagens e limitações.

### Conclusão

A busca pela "melhor" aproximação de $P_{data}$ através de um modelo generativo $P_\theta$ é um problema fundamental e desafiador em aprendizado de máquina [4]. Este resumo explorou os conceitos centrais, desafios e técnicas avançadas neste campo:

1. A definição de "melhor" é crucial e depende do objetivo final do modelo, levando a diferentes critérios de otimização como divergência KL, máxima verossimilhança e distância de Wasserstein [1].

2. Desafios fundamentais incluem a esparsidade de dados em altas dimensões, mode collapse, dificuldades de avaliação e landscapes de otimização não-convexos [2,3].

3. Técnicas avançadas como VAEs, GANs, fluxos normalizadores e modelos de difusão oferecem abordagens poderosas, cada uma com seus próprios trade-offs [1,4].

4. Estratégias de otimização específicas, incluindo SGD com momentum, otimização alternada e técnicas de regularização, são cruciais para o treinamento eficaz de modelos generativos [2].

5. A avaliação e diagnóstico de modelos generativos requerem uma combinação de métricas quantitativas e avaliação qualitativa, refletindo a complexidade do problema [3,4].

À medida que o campo avança, a integração de insights teóricos com inovações práticas continua a impulsionar o desenvolvimento de modelos generativos mais poderosos e versáteis, aproximando-nos cada vez mais do objetivo de capturar verdadeiramente a riqueza e complexidade das distribuições de dados do mundo real.

### Questões Avançadas

1. Considere um cenário onde você está treinando um modelo generativo para um conjunto de dados de alta dimensão com estrutura hierárquica conhecida (por exemplo, imagens de faces com atributos como expressão, pose, iluminação). Como você poderia incorporar esse conhecimento prévio na arquitetura e no processo de treinamento do modelo para melhorar tanto a qualidade das amostras geradas quanto a interpretabilidade do espaço latente?

2. Discuta as implicações teóricas e práticas de usar um ensemble de diferentes tipos de modelos generativos (por exemplo, VAE, GAN e modelo de difusão) para aproximar $P_{data}$. Como você combinaria as saídas desses modelos e quais seriam os desafios na otimização conjunta de tal sistema?

3. Proponha uma abordagem para adaptar continuamente um modelo generativo à medida que novos dados chegam em um fluxo contínuo, mantendo a capacidade de gerar amostras de "versões" anteriores da distribuição. Como você lidaria com o problema de "esquecimento catastrófico" neste contexto?

### Referências

[1] "The goal of learning is to return a model Pθ that precisely captures the distribution Pdata from which our data was sampled" (Trecho de cs236_lecture4.pdf)

[2] "This is in general not achievable because of limited data only provides a rough approximation of the true underlying distribution computational reasons" (Trecho de cs236_lecture4.pdf)

[3] "Example. Suppose we represent each image with a vector X of 784 binary variables (black vs. white pixel). How many possible states (= possible images) in the model? 2784 ≈ 10236. Even 107 training examples provide extremely sparse coverage!" (Trecho de cs236_lecture4.pdf)

[4] "We want to select Pθ to construct the "best" approximation to the underlying distribution Pdata What is "best"?" (Trecho de cs236_lecture4.pdf)