## Wasserstein Distance e sua Aplicação em GANs

<imagem: Uma representação visual comparando duas distribuições de probabilidade e uma seta indicando o "transporte" entre elas, simbolizando o conceito de Wasserstein distance>

### Introdução

==A **Wasserstein distance**, também conhecida como **Earth Mover's Distance (EMD)**, é uma métrica fundamental na teoria das probabilidades e estatística, com raízes na teoria do transporte ótimo [1].== Originalmente desenvolvida para resolver problemas de alocação de recursos, essa métrica ganhou destaque significativo no campo do aprendizado profundo, especialmente no contexto de **Redes Geradoras Adversariais (GANs)** [2]. ==Ela oferece uma abordagem robusta para medir a dissimilaridade entre distribuições de probabilidade, superando limitações de métricas tradicionais como a divergência de Kullback-Leibler e a divergência de Jensen-Shannon [3].==

No contexto das GANs, a Wasserstein distance emergiu como uma ==solução para problemas críticos enfrentados durante o treinamento, como instabilidade, dificuldade de convergência e o **colapso de modo** (mode collapse) [4]==. Sua introdução levou ao desenvolvimento de variantes importantes como a **Wasserstein GAN (WGAN)** e a **Wasserstein GAN com Penalidade de Gradiente (WGAN-GP)**, que representam avanços significativos na geração de imagens de alta qualidade e na estabilidade do treinamento [5].

### Contexto Histórico

A teoria do transporte ótimo, na qual a Wasserstein distance está fundamentada, foi inicialmente formulada por **Gaspard Monge** no século XVIII, ==visando resolver problemas de movimentação de terras para construção [6]==. Posteriormente, **Leonid Kantorovich** generalizou o problema, introduzindo uma formulação relaxada que permitia soluções mais práticas e computacionalmente viáveis [7]. Essa teoria encontrou aplicações em diversas áreas, como economia, física e, mais recentemente, aprendizado de máquina [8].

### Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Wasserstein Distance**   | Métrica que quantifica a diferença entre duas distribuições de probabilidade, interpretada como o custo mínimo de transformar uma distribuição em outra [9]. |
| **Earth Mover's Distance** | Termo alternativo para Wasserstein distance, visualizando as distribuições como pilhas de terra a serem movidas de um formato para outro, minimizando o esforço total [10]. |
| **Transporte Ótimo**       | Área da matemática que estuda a forma mais eficiente de transformar uma distribuição em outra, minimizando um custo associado ao transporte [11]. |
| **Lipschitz Continuity**   | Propriedade de funções onde existe uma constante $K$ tal que $|f(x) - f(y)| \leq K|x - y|$ para todos $x, y$, garantindo continuidade controlada [12]. |
| **WGAN**                   | Variante de GAN que utiliza a Wasserstein distance como função de perda, proporcionando treinamento mais estável e gradientes mais informativos [13]. |
| **WGAN-GP**                | Extensão da WGAN que introduz uma penalidade de gradiente para impor a condição de Lipschitz, melhorando a estabilidade e qualidade do treinamento [14]. |

> ⚠️ **Nota Importante**: A Wasserstein distance oferece uma métrica contínua e diferenciável entre distribuições, crucial para o treinamento efetivo de GANs, pois evita problemas de gradientes nulos ou explosivos [15].

### Fundamentos Matemáticos da Wasserstein Distance

==A Wasserstein distance é fundamentada na teoria do transporte ótimo, proporcionando uma medida robusta da diferença entre distribuições de probabilidade== [16]. ==Matematicamente, para duas distribuições de probabilidade $P$ e $Q$ definidas em um espaço métrico $(\mathcal{X}, d)$==, a **Wasserstein distance de ordem $p$** é definida como:
$$
W_p(P, Q) = \left( \inf_{\gamma \in \Pi(P, Q)} \int_{\mathcal{X} \times \mathcal{X}} d(x, y)^p \, d\gamma(x, y) \right)^{1/p}
$$

Onde:

- $\Pi(P, Q)$ ==é o conjunto de todas as medidas de probabilidade conjuntas em $\mathcal{X} \times \mathcal{X}$ com marginais $P$ e $Q$ [17].==
- $d(x, y)$ ==é a distância entre pontos $x$ e $y$ no espaço métrico $\mathcal{X}$.==
- $p \geq 1$ é a ordem da Wasserstein distance.

Esta formulação ==captura a ideia de "mover" massa de probabilidade de uma distribuição para outra com o menor custo possível==, onde o ==custo é determinado pela distância $d(x, y)$ e a quantidade de massa a ser transportada [18].==

#### Propriedades Teóricas da Wasserstein Distance

1. **Non-negatividade e Identidade dos Indiscerníveis**: $W_p(P, Q) \geq 0$ e $W_p(P, Q) = 0$ se, e somente se, $P = Q$ [19].

2. **Simetria**: $W_p(P, Q) = W_p(Q, P)$ [20].

3. **Desigualdade Triangular**: Para quaisquer distribuições $P$, $Q$ e $R$, $W_p(P, R) \leq W_p(P, Q) + W_p(Q, R)$ [21].

4. **Convergência em Wasserstein Implica Convergência em Distribuição**: Se $W_p(P_n, P) \to 0$, então $P_n$ converge em distribuição para $P$ [22].

#### Dualidade em Transporte Ótimo

==O Teorema de Kantorovich-Rubinstein estabelece que, para o caso $p = 1$, a Wasserstein distance pode ser expressa em termos de funções Lipschitz contínuas [23]:==
$$
W_1(P, Q) = \sup_{\|f\|_{\text{Lip}} \leq 1} \left\{ \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{y \sim Q}[f(y)] \right\}
$$

Onde $\|f\|_{\text{Lip}}$ é a menor constante Lipschitz de $f$, ou seja, o menor $K$ tal que $|f(x) - f(y)| \leq K|x - y|$ para todos $x, y$ [24].

==Esta dualidade é fundamental para a aplicação da Wasserstein distance em problemas de otimização, pois permite reformular um problema originalmente difícil (infimum sobre medidas conjuntas) em um problema de supremum sobre funções Lipschitz [25==

### Aplicação em Wasserstein GAN (WGAN)

A WGAN utiliza a Wasserstein distance como função de perda, substituindo a divergência de Jensen-Shannon tradicionalmente usada em GANs [26]. Isso resolve problemas relacionados à falta de suporte comum entre as distribuições real e gerada, o que pode levar a gradientes pouco informativos. A função objetivo da WGAN é expressa como:

$$
\min_{G} \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim P_r}[D(x)] - \mathbb{E}_{z \sim P_z}[D(G(z))]
$$

Onde:

- $G$ é o gerador que transforma uma variável aleatória $z$ (ruído) em uma amostra gerada $G(z)$.
- $D$ é o discriminador (chamado de crítico na WGAN) que estima o valor de $D(x)$ para amostras reais e geradas.
- $\mathcal{D}$ é o conjunto de funções $D$ 1-Lipschitz [27].
- $P_r$ é a distribuição real dos dados.
- $P_z$ é a distribuição do ruído de entrada.

O objetivo é que o crítico $D$ aproxime a diferença entre as distribuições real e gerada, enquanto o gerador $G$ tenta minimizar essa diferença [28].

> ✔️ **Destaque**: A WGAN fornece gradientes estáveis e significativos, mesmo quando as distribuições não se sobrepõem, resolvendo problemas de gradientes nulos em GANs tradicionais [29].

#### Análise Teórica da WGAN

==A eficácia da WGAN está ligada à capacidade de $D$ em aproximar funções 1-Lipschitz e à capacidade de $G$ em ajustar $P_g$ para minimizar $W_1(P_r, P_g)$ [30].== A restrição de Lipschitz é crucial para garantir a validade da dualidade de Kantorovich-Rubinstein e, consequentemente, a correta estimação da Wasserstein distance [31].

##### Condição de Lipschitz

==Para garantir que $D$ seja 1-Lipschitz, a implementação original da WGAN propõe o **weight clipping**, limitando os pesos de $D$ a um intervalo fixo $[-c, c]$ [32]==. No entanto, essa abordagem pode limitar a capacidade expressiva do modelo e levar a dificuldades no treinamento.

### Gradient Penalty Wasserstein GAN (WGAN-GP)

A WGAN-GP introduz uma penalidade de gradiente para impor a condição de Lipschitz no crítico, evitando os problemas associados ao weight clipping [33]. A função objetivo da WGAN-GP é dada por:

$$
L = \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}\left[ \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2 \right]
$$

Onde:

- $P_g$ é a distribuição gerada por $G$.
- $P_{\hat{x}}$ é a distribuição de amostras interpoladas entre $P_r$ e $P_g$, definidas como $\hat{x} = \epsilon x + (1 - \epsilon) \tilde{x}$, com $\epsilon \sim \text{Uniform}(0,1)$ [34].
- $\lambda$ é o coeficiente de penalidade que controla a importância da penalidade de gradiente.

A penalidade de gradiente força o gradiente de $D$ em relação às amostras $\hat{x}$ a ter norma unitária, garantindo a condição de Lipschitz de forma mais eficaz [35].

#### Justificativa Teórica da Penalidade de Gradiente

A penalidade de gradiente é baseada na observação de que uma função é 1-Lipschitz se, e somente se, seu gradiente em relação a todas as direções tiver norma menor ou igual a 1 [36]. Ao penalizar desvios dessa norma unitária, garantimos que $D$ permaneça próximo ao espaço de funções 1-Lipschitz [37].

> ❗ **Ponto de Atenção**: A penalidade de gradiente na WGAN-GP promove um treinamento mais estável e evita problemas de explosão ou desaparecimento de gradientes, melhorando a qualidade das amostras geradas [38].

### Análise Comparativa: WGAN vs GAN Tradicional

| 👍 **Vantagens da WGAN**                                      | 👎 **Desvantagens da GAN Tradicional**                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Treinamento mais estável [39]                                | Instabilidade durante o treinamento [40]                     |
| Gradientes informativos mesmo com suportes não sobrepostos [41] | Gradientes nulos quando as distribuições não se sobrepõem [42] |
| Correlação entre função de perda e qualidade da amostra [43] | Falta de correlação entre perda e qualidade da amostra [44]  |
| Redução do colapso de modo [45]                              | Suscetibilidade ao colapso de modo [46]                      |

### Prova Teórica: Convergência da Wasserstein GAN

Apresentaremos agora uma prova teórica simplificada da convergência da Wasserstein GAN, demonstrando sua superioridade em relação às GANs tradicionais.

**Teorema**: Sob condições adequadas e assumindo capacidade infinita do crítico $D$, a Wasserstein GAN converge para um equilíbrio global, minimizando a Wasserstein distance entre a distribuição real $P_r$ e a distribuição gerada $P_g$.

**Prova**:

1. **Dualidade de Kantorovich-Rubinstein**: Pelo teorema de Kantorovich-Rubinstein, para funções $f$ 1-Lipschitz, a Wasserstein distance é dada por:

   $$
   W_1(P_r, P_g) = \sup_{\|f\|_{\text{Lip}} \leq 1} \left\{ \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)] \right\}
   $$

2. **Função Objetivo da WGAN**: ==A WGAN busca aproximar essa supremum usando um crítico $D$ parametrizado e otimizado para maximizar a diferença das expectativas [47].==

3. **Otimização Alternada**: No treinamento da WGAN, realizamos uma otimização alternada onde:

   - **Passo do Crítico**: Otimizamos $D$ para aproximar o valor ótimo da supremum.
   - **Passo do Gerador**: Otimizamos $G$ para minimizar a diferença das expectativas, reduzindo assim $W_1(P_r, P_g)$.

4. **Convergência do Crítico**: Com capacidade suficiente, o crítico $D$ pode aproximar qualquer função 1-Lipschitz, atingindo o valor ótimo na supremum [48].

5. **Convergência do Gerador**: Ao minimizar $W_1(P_r, P_g)$, o gerador $G$ ajusta $P_g$ para aproximar $P_r$, levando à convergência das distribuições [49].

6. **Gradientes Significativos**: A continuidade da Wasserstein distance garante que o gradiente em relação aos parâmetros de $G$ seja significativo, mesmo quando $P_r$ e $P_g$ não se sobrepõem [50].

7. **Conclusão**: Portanto, sob as condições assumidas, a WGAN converge para um equilíbrio global, minimizando efetivamente a Wasserstein distance entre $P_r$ e $P_g$.

> ⚠️ **Ponto Crucial**: A convergência global da WGAN contrasta com as GANs tradicionais, que podem ficar presas em equilíbrios locais subótimos devido à natureza não contínua da divergência de Jensen-Shannon e à falta de gradientes significativos quando as distribuições não se sobrepõem [51].

### Análise Detalhada da Condição de Lipschitz

A restrição de Lipschitz é essencial para a validade da dualidade de Kantorovich-Rubinstein. Funções que não satisfazem essa condição podem levar a estimativas incorretas da Wasserstein distance [52]. Na prática, a imposição dessa restrição é desafiadora devido à capacidade finita dos modelos neurais.

#### Penalidade de Gradiente vs. Weight Clipping

- **Weight Clipping**: Simples de implementar, mas pode limitar a expressividade do crítico e introduzir artefatos no treinamento [53].
- **Penalidade de Gradiente**: Impõe a restrição de Lipschitz de forma mais suave e eficaz, permitindo que o crítico mantenha sua capacidade representacional [54].

Matematicamente, a penalidade de gradiente adiciona um termo regularizador à função de perda, garantindo que o gradiente em relação às entradas tenha norma próxima de 1 [55]:

$$
\text{Penalidade} = \lambda \mathbb{E}_{\hat{x}} \left( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \right)^2
$$

Onde $\hat{x}$ são amostras interpoladas entre o conjunto real e gerado [56].

### Relação entre Wasserstein Distance e Outras Métricas

Comparada a outras métricas e divergências usadas em aprendizado de máquina:

- **Divergência de Kullback-Leibler (KL)**: Mede a expectativa logarítmica da diferença entre duas distribuições, mas pode ser infinita se os suportes não coincidirem [57].
- **Divergência de Jensen-Shannon (JS)**: Uma versão simétrica e suavizada da KL, mas sua derivada pode ser nula quando as distribuições não se sobrepõem [58].
- **Wasserstein Distance**: Fornece uma medida finita e diferenciável mesmo quando os suportes das distribuições não se sobrepõem, tornando-a mais adequada para treinamento de modelos generativos [59].

### Implicações Teóricas no Treinamento de GANs

A adoção da Wasserstein distance no treinamento de GANs traz as seguintes implicações teóricas:

1. **Gradientes Não Nulos**: Garantia de gradientes úteis para atualizar o gerador, mesmo em estágios iniciais onde $P_g$ e $P_r$ são significativamente diferentes [60].

2. **Estabilidade de Treinamento**: Redução de oscilações e comportamentos caóticos durante o treinamento, facilitando a convergência [61].

3. **Interpretação da Função de Perda**: A perda na WGAN tem uma interpretação direta como uma estimativa da distância entre distribuições, ao contrário da perda adversarial tradicional [62].

### Conclusão

A introdução da Wasserstein distance no contexto das GANs representa um avanço significativo na teoria e prática de modelos generativos adversariais [63]. As WGANs e WGAN-GPs oferecem soluções robustas para problemas críticos enfrentados por GANs tradicionais, como instabilidade de treinamento e colapso de modo [64]. A fundamentação teórica sólida da Wasserstein distance, combinada com sua aplicabilidade prática, posiciona essas variantes como ferramentas poderosas no campo do aprendizado profundo generativo [65].

A capacidade de fornecer gradientes significativos mesmo quando as distribuições não se sobrepõem, juntamente com a correlação entre a função de perda e a qualidade das amostras geradas, torna as WGANs particularmente atraentes para aplicações que exigem alta fidelidade e diversidade nas amostras geradas [66]. O entendimento aprofundado dos fundamentos teóricos da Wasserstein distance e sua implementação em WGANs é essencial para pesquisadores e profissionais que buscam avançar o estado da arte em modelos generativos [67].