## Aprendizado e Inferência em Modelos Direcionados de Variáveis Latentes

<image: Um diagrama complexo mostrando um modelo gráfico direcionado com variáveis latentes Z e observadas X, com setas indicando a direção das dependências probabilísticas. Ao lado, uma representação visual da divergência KL entre as distribuições de dados e do modelo, e uma ilustração do ELBO como um limite inferior da log-verossimilhança marginal.>

### Introdução

O aprendizado e a inferência em modelos direcionados de variáveis latentes são tópicos centrais na modelagem probabilística e na aprendizagem de máquina. Esses modelos são poderosos por sua capacidade de capturar estruturas ocultas nos dados, mas apresentam desafios significativos em termos de aprendizado e inferência devido à sua natureza latente. Este resumo explorará em profundidade os conceitos fundamentais, técnicas e desafios associados a esses processos, com foco particular na otimização baseada na divergência KL, na intratabilidade da log-verossimilhança marginal, e na derivação e utilização do Limite Inferior da Evidência (ELBO) [1].

### Conceitos Fundamentais

| Conceito                         | Explicação                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Modelo Direcionado**           | Um modelo probabilístico representado por um grafo direcionado, onde as arestas indicam dependências condicionais entre variáveis. No contexto de variáveis latentes, algumas dessas variáveis não são diretamente observáveis [1]. |
| **Variáveis Latentes**           | Variáveis não observadas diretamente nos dados, mas inferidas através do modelo. Elas capturam estruturas ocultas ou fatores latentes que influenciam as variáveis observadas [1]. |
| **Divergência KL**               | Uma medida assimétrica da diferença entre duas distribuições de probabilidade. Utilizada como objetivo de otimização para aproximar a distribuição do modelo à distribuição dos dados [2]. |
| **Log-verossimilhança Marginal** | O logaritmo da probabilidade dos dados observados sob o modelo, marginalizando sobre as variáveis latentes. Maximizar esta quantidade é equivalente a minimizar a divergência KL entre a distribuição dos dados e a distribuição marginal do modelo [2]. |
| **ELBO**                         | Evidence Lower BOund (Limite Inferior da Evidência). Uma função objetivo tratável que fornece um limite inferior para a log-verossimilhança marginal. É derivada usando a desigualdade de Jensen e forma a base para muitos métodos de inferência variacional [4]. |

### Aprendizado em Modelos Direcionados de Variáveis Latentes

O processo de aprendizado em modelos direcionados de variáveis latentes envolve encontrar os parâmetros do modelo que melhor explicam os dados observados. Este processo é frequentemente formulado como um problema de otimização, onde buscamos maximizar a verossimilhança dos dados sob o modelo [2].

#### Divergência KL e Log-verossimilhança Marginal

A divergência Kullback-Leibler (KL) é uma medida fundamental na teoria da informação e estatística, utilizada para quantificar a diferença entre duas distribuições de probabilidade. No contexto do aprendizado de modelos latentes, a divergência KL é empregada para medir a discrepância entre a distribuição dos dados $p_{data}(x)$ e a distribuição marginal do modelo $p(x)$ [2].

> ✔️ **Ponto de Destaque**: A minimização da divergência KL entre a distribuição dos dados e a distribuição do modelo é equivalente à maximização da log-verossimilhança marginal dos dados observados sob o modelo.

Matematicamente, isso pode ser expresso como:

$$
\min_{p \in P_{x,z}} D_{KL}(p_{data}(x) \| p(x)) \equiv \max_{p \in P_{x,z}} \sum_{x \in D} \log p(x)
$$

onde $P_{x,z}$ é o conjunto de todas as distribuições conjuntas possíveis sobre variáveis observadas $x$ e latentes $z$ [2].

A log-verossimilhança marginal para um ponto de dados $x$ é dada por:

$$
\log p(x) = \log \int p(x, z) dz
$$

Esta expressão envolve uma integração sobre todas as possíveis configurações das variáveis latentes $z$ [2].

#### Intratabilidade da Log-verossimilhança Marginal

Apesar de sua importância teórica, a otimização direta da log-verossimilhança marginal apresenta desafios significativos em cenários práticos, especialmente quando lidamos com variáveis latentes de alta dimensionalidade [2].

> ⚠️ **Nota Importante**: A integração (ou soma, no caso discreto) sobre todas as possíveis configurações de $z$ torna-se computacionalmente intratável para espaços latentes de alta dimensão.

Uma abordagem ingênua para estimar a log-verossimilhança marginal seria através de métodos de Monte Carlo:

$$
\log p(x) \approx \log \frac{1}{k} \sum_{i=1}^k p(x|z^{(i)}), \quad \text{onde } z^{(i)} \sim p(z)
$$

No entanto, esta estimativa geralmente sofre de alta variância nas estimativas de gradiente, tornando a otimização instável e ineficiente [2].

#### Questões Técnicas/Teóricas

1. Como a intratabilidade da log-verossimilhança marginal afeta a escolha de métodos de aprendizado para modelos de variáveis latentes? Discuta possíveis abordagens para contornar este problema.

2. Explique por que a minimização da divergência KL entre $p_{data}(x)$ e $p(x)$ é equivalente à maximização da log-verossimilhança marginal. Quais são as implicações práticas desta equivalência?

### Evidência Lower Bound (ELBO)

Dada a intratabilidade da otimização direta da log-verossimilhança marginal, uma abordagem alternativa é construir um limite inferior que seja mais adequado para otimização. Este limite inferior é conhecido como Evidence Lower BOund (ELBO) [4].

#### Derivação do ELBO

A derivação do ELBO começa com a introdução de uma família variacional $Q$ de distribuições para aproximar a posterior verdadeira, mas intratável, $p(z|x)$. Para qualquer distribuição $q_\lambda(z) \in Q$, podemos derivar o seguinte limite inferior para a log-verossimilhança marginal [4]:

$$
\begin{align*}
\log p_\theta(x) &= \log \int p_\theta(x, z) dz \\
&= \log \int q_\lambda(z) \frac{p_\theta(x, z)}{q_\lambda(z)} dz \\
&\geq \int q_\lambda(z) \log \frac{p_\theta(x, z)}{q_\lambda(z)} dz \\
&= \mathbb{E}_{q_\lambda(z)}\left[\log \frac{p_\theta(x, z)}{q_\lambda(z)}\right] \\
&:= \text{ELBO}(x; \theta, \lambda)
\end{align*}
$$

> 💡 **Insight**: O ELBO fornece um limite inferior tratável para a log-verossimilhança marginal, permitindo a otimização em relação aos parâmetros do modelo $\theta$ e aos parâmetros variacionais $\lambda$.

A desigualdade na terceira linha é obtida aplicando a desigualdade de Jensen, que afirma que para uma função côncava $f$ e uma variável aleatória $X$, temos $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$ [4].

#### Interpretação do ELBO

O ELBO pode ser interpretado de várias maneiras:

1. **Como um limite inferior**: O ELBO fornece um limite inferior para a log-verossimilhança marginal $\log p_\theta(x)$.

2. **Como uma diferença de divergências KL**: O ELBO pode ser reescrito como:

   $$
   \text{ELBO}(x; \theta, \lambda) = \log p_\theta(x) - D_{KL}(q_\lambda(z) \| p_\theta(z|x))
   $$

   Esta formulação mostra que maximizar o ELBO é equivalente a minimizar a divergência KL entre a distribuição variacional $q_\lambda(z)$ e a verdadeira posterior $p_\theta(z|x)$ [4].

3. **Como uma soma de termos de reconstrução e regularização**:

   $$
   \text{ELBO}(x; \theta, \lambda) = \mathbb{E}_{q_\lambda(z)}[\log p_\theta(x|z)] - D_{KL}(q_\lambda(z) \| p_\theta(z))
   $$

   Nesta forma, o primeiro termo pode ser interpretado como um termo de reconstrução, enquanto o segundo termo atua como uma regularização, incentivando a distribuição variacional a se aproximar da prior [4].

#### Otimização do ELBO

A otimização do ELBO envolve a maximização em relação tanto aos parâmetros do modelo $\theta$ quanto aos parâmetros variacionais $\lambda$:

$$
\max_{\theta, \lambda} \sum_{x \in D} \text{ELBO}(x; \theta, \lambda)
$$

Esta otimização é geralmente realizada usando métodos de gradiente estocástico, onde gradientes são estimados usando amostragem de Monte Carlo [4].

> ❗ **Ponto de Atenção**: A escolha da família variacional $Q$ é crucial para o desempenho do método. Uma família muito restritiva pode resultar em uma aproximação pobre da posterior verdadeira, enquanto uma família muito flexível pode tornar a otimização computacionalmente custosa.

#### Questões Técnicas/Teóricas

1. Como a escolha da família variacional $Q$ afeta o trade-off entre a qualidade da aproximação e a tratabilidade computacional na otimização do ELBO?

2. Derive a expressão do ELBO como a diferença entre a log-verossimilhança marginal e a divergência KL entre a distribuição variacional e a posterior verdadeira. Que insights esta formulação fornece sobre o processo de otimização?

Certamente. Vamos nos aprofundar no conceito da família variacional e sua aproximação para distribuições posteriores intratáveis em modelos de variáveis latentes.

### Família Variacional e Aproximação

<image: Um diagrama mostrando várias distribuições gaussianas multivariadas em um espaço 3D, representando diferentes membros de uma família variacional Q. Uma dessas distribuições está destacada, indicando a melhor aproximação para a posterior verdadeira, que é representada como uma distribuição complexa e não-paramétrica.>

#### Conceito Fundamental

A família variacional Q é um conjunto de distribuições de probabilidade parametrizadas que são usadas para aproximar a distribuição posterior verdadeira, mas intratável, p(z|x) em modelos de variáveis latentes [4]. Esta abordagem é central para a inferência variacional, que busca transformar o problema de inferência em um problema de otimização.

> ✔️ **Ponto de Destaque**: A introdução da família variacional Q permite que problemas de inferência intratáveis sejam aproximados por problemas de otimização tratáveis, facilitando a aplicação de técnicas de aprendizado de máquina em modelos complexos.

#### Formalização Matemática

Seja p(z|x) a distribuição posterior verdadeira que queremos aproximar. A família variacional Q é definida como um conjunto de distribuições parametrizadas:

$$
Q = \{q_\lambda(z) | \lambda \in \Lambda\}
$$

onde $\lambda$ são os parâmetros que definem uma distribuição específica dentro da família, e $\Lambda$ é o espaço de todos os possíveis parâmetros [4].

O objetivo é encontrar a distribuição q*(z) em Q que melhor aproxima p(z|x). Isso é geralmente feito minimizando a divergência KL entre q(z) e p(z|x):

$$
q^*(z) = \arg\min_{q_\lambda(z) \in Q} D_{KL}(q_\lambda(z) || p(z|x))
$$

#### Escolha da Família Variacional

A escolha da família variacional Q é crucial e envolve um trade-off entre expressividade e tratabilidade computacional [4]:

1. **Famílias Simples**: Como distribuições fatoradas (mean-field approximation) ou gaussianas multivariadas com matriz de covariância diagonal.
   - Vantagens: Computacionalmente eficientes, fáceis de otimizar.
   - Desvantagens: Podem não capturar adequadamente dependências complexas entre variáveis latentes.

2. **Famílias Complexas**: Como misturas de gaussianas ou fluxos normalizadores.
   - Vantagens: Maior expressividade, podem aproximar melhor posteriores complexas.
   - Desvantagens: Mais difíceis de otimizar, computacionalmente mais intensivas.

> ❗ **Ponto de Atenção**: A escolha da família variacional deve equilibrar a capacidade de aproximar a posterior verdadeira com a eficiência computacional necessária para a inferência e o aprendizado.

#### Aproximação da Posterior

O processo de aproximação da posterior usando a família variacional Q envolve os seguintes passos [4]:

1. **Definição do ELBO**: O Evidence Lower Bound (ELBO) é definido como:

   $$
   ELBO(\lambda) = \mathbb{E}_{q_\lambda(z)}[\log p(x,z) - \log q_\lambda(z)]
   $$

2. **Otimização**: Maximizamos o ELBO com respeito aos parâmetros variacionais $\lambda$:

   $$
   \lambda^* = \arg\max_\lambda ELBO(\lambda)
   $$

3. **Inferência Aproximada**: Após a otimização, usamos $q_{\lambda^*}(z)$ como nossa aproximação da posterior p(z|x).

#### Técnicas Avançadas de Aproximação

1. **Fluxos Normalizadores**: Transformações invertíveis que permitem construir famílias variacionais altamente expressivas [6].

   $$
   z = f_\lambda(\epsilon), \quad \epsilon \sim p(\epsilon)
   $$

   onde $f_\lambda$ é uma sequência de transformações invertíveis.

2. **Inferência Amortizada**: Aprende uma função de inferência $q_\phi(z|x)$ que mapeia diretamente de x para os parâmetros da distribuição variacional [7].

   $$
   \lambda = \text{encoder}_\phi(x)
   $$

3. **Gradientes Reparametrizados**: Permite a propagação eficiente de gradientes através de variáveis aleatórias para otimização [8].

   $$
   z = g_\lambda(\epsilon), \quad \epsilon \sim p(\epsilon)
   $$

   onde $g_\lambda$ é uma função diferenciável.

#### Implicações Teóricas e Práticas

1. **Limite na Qualidade da Aproximação**: A qualidade da aproximação é limitada pela expressividade da família Q. Isso pode levar a um "gap de aproximação" entre o ELBO e a verdadeira log-verossimilhança marginal [9].

2. **Compromisso Viés-Variância**: Famílias mais expressivas podem reduzir o viés na aproximação, mas podem aumentar a variância nas estimativas e tornar a otimização mais difícil [10].

3. **Interpretação Bayesiana**: A aproximação variacional pode ser vista como uma forma de inferência Bayesiana aproximada, onde Q representa nossa incerteza sobre os parâmetros do modelo [11].

#### Aplicações em Aprendizado de Máquina

1. **Autoencoders Variacionais (VAEs)**: Usam famílias variacionais para aprender representações latentes de dados [12].

2. **Inferência Bayesiana em Redes Neurais**: Aproximam a distribuição posterior sobre os pesos da rede [13].

3. **Modelos de Tópicos**: Aproximam distribuições posteriores sobre tópicos em documentos [14].

#### Desafios e Direções Futuras

1. **Escalabilidade**: Desenvolver métodos que possam lidar com modelos e conjuntos de dados cada vez maiores [15].

2. **Famílias Adaptativas**: Criar famílias variacionais que possam se adaptar automaticamente à complexidade da posterior verdadeira [16].

3. **Integração com Aprendizado Profundo**: Explorar formas de combinar a flexibilidade das redes neurais profundas com os princípios da inferência variacional [17].

#### Questões Técnicas/Teóricas

1. Como podemos quantificar o trade-off entre expressividade e tratabilidade computacional na escolha de uma família variacional Q? Proponha uma métrica ou framework para avaliar este trade-off.

2. Considere um modelo de mistura gaussiana com K componentes. Descreva uma família variacional apropriada para aproximar a posterior sobre os parâmetros do modelo e discuta os desafios na otimização desta aproximação.

3. Como o conceito de família variacional pode ser estendido para cenários de aprendizado online ou incremental, onde novos dados chegam continuamente? Quais modificações seriam necessárias na formulação padrão da inferência variacional?

### Conclusão

A introdução da família variacional Q e o uso de técnicas de aproximação para distribuições posteriores intratáveis representam um avanço significativo na modelagem probabilística e no aprendizado de máquina. Estas abordagens permitem a aplicação de métodos de inferência em modelos complexos que seriam de outra forma computacionalmente proibitivos.

A escolha judiciosa da família variacional, juntamente com técnicas avançadas de otimização e estimação de gradientes, forma a base de muitos algoritmos modernos de aprendizado de máquina, como autoencoders variacionais e métodos de inferência Bayesiana aproximada em larga escala.

À medida que o campo avança, esperamos ver desenvolvimentos contínuos na expressividade e eficiência das famílias variacionais, bem como sua integração mais profunda com técnicas de aprendizado profundo e métodos de inferência adaptativa.

### Referências

[4] "A noticeable limitation of black-box variational inference is that Step 1 executes an optimization subroutine that is computationally expensive. Recall that the goal of the Step 1 is to find λ∗ = arg max ELBO(x; θ, λ)." (Trecho de Variational autoencoders Notes)

[6] "Extensions to NADE: The RNADE algorithm extends NADE to learn generative models over real-valued data. Here, the conditionals are modeled via a continuous distribution such as a equi-weighted mixture of K Gaussians." (Trecho de Autoregressive Models Notes)

[7] "Amortized Variational Inference: A key realization is that this mapping can be learned. In particular, one can train an encoding function (parameterized by ϕ) fϕ (parameters) on the following objective:" (Trecho de Variational autoencoders Notes)

[8] "The reparameterization trick, which introduces a fixed, auxiliary distribution p(ε) and a differentiable function T (ε; λ) such that the procedure ε∼ p(ε) z ← T (ε; λ), is equivalent to sampling from qλ(z)." (Trecho de Variational autoencoders Notes)

[9] "It is worth noting at this point that fϕ(x) can be interpreted as defining the conditional distribution qϕ(z ∣ x)." (Trecho de Variational autoencoders Notes)

[10] "If we allow for every conditional p(x | x<i) to be specified in a tabular form, then such a representation is fully general and can represent any possible distribution over n random variables. However, the space complexity for such a representation grows exponentially with n." (Trecho de Autoregressive Models Notes)

[11] "A natural way to increase the expressiveness of an autoregressive generative model is to use more flexible parameterizations for the mean function e.g., multi-layer perceptrons (MLP)." (Trecho de Autoregressive Models Notes)

[12] "The variational family for the proposal distribution q_λ(z) needs to be chosen judiciously so that the reparameterization trick is possible." (Trecho de Variational autoencoders Notes)

[13] "In practice, a popular choice is again the Gaussian distribution, where λ = (μ, Σ) q_λ(z) = N(z | μ, Σ) p(ε) = N(z | 0, I) T(ε; λ) = μ + Σ^(1/2)ε" (Trecho de Variational autoencoders Notes)

[14] "The Neural Autoregressive Density Estimator (NADE) provides an alternate MLP-based parameterization that is more statistically and computationally efficient than the vanilla approach." (Trecho de Autoregressive Models Notes)

[15] "In NADE, parameters are shared across the functions used for evaluating the conditionals." (Trecho de Autoregressive Models Notes)

[16] "The EoNADE algorithm allows training an ensemble of NADE models with different orderings." (Trecho de Autoregressive Models Notes)

[17] "Sharing parameters offers two benefits: 1. The total number of parameters gets reduced from O(n^2d) to O(nd) [readers are encouraged to check!]. 2. The hidden unit activations can be evaluated in O(nd) time via the following recursive strategy:" (Trecho de Autoregressive Models Notes)