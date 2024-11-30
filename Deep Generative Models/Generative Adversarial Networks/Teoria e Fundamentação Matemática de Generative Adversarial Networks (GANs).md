# Teoria e Fundamentação Matemática de Generative Adversarial Networks (GANs)

### Introdução

As **Generative Adversarial Networks (GANs)** representam uma classe revolucionária de modelos generativos que utilizam algoritmos de aprendizado de máquina para aprender distribuições a partir de dados de treinamento e gerar novos exemplos [1]. Este framework é particularmente notável pela sua capacidade de gerar amostras sintéticas de alta qualidade, especialmente em domínios complexos como imagens.

> ⚠️ **Conceito Fundamental**: Uma GAN consiste em dois componentes principais — um gerador e um discriminador — que são treinados de forma adversarial, competindo um contra o outro em um jogo de soma zero [2].

### Conceitos Fundamentais

| Conceito                 | Explicação                                                   |
| ------------------------ | ------------------------------------------------------------ |
| **Modelo Generativo**    | Definido por uma distribuição $p(x\|w)$, onde $x$ é um vetor no espaço de dados e $w$ representa os parâmetros aprendíveis do modelo [1]. |
| **Distribuição Latente** | Tipicamente uma distribuição Gaussiana simples $p(z) = N(z\|0, I)$, que serve como entrada para o gerador [3]. |
| **Função de Perda GAN**  | Definida pela equação de erro de cross-entropia que orienta o treinamento adversarial, permitindo a aprendizagem das distribuições de probabilidade [4]. |

---

### Framework Matemático da GAN

A formulação matemática da GAN é baseada em um jogo de minimax entre o gerador ($G$) e o discriminador ($D$). O objetivo é resolver o seguinte problema [4]:

$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{data}(x)} [\ln D(x)] + \mathbb{E}_{z \sim p(z)} [\ln (1 - D(G(z)))]
$$

**Onde:**

- $D(x)$ representa a probabilidade estimada pelo discriminador de que $x$ seja uma amostra real.
- $G(z)$ é a amostra gerada pelo gerador a partir de uma variável latente $z$.
- $p_{data}(x)$ é a distribuição real dos dados.
- $p(z)$ é a distribuição latente (por exemplo, $N(0, I)$).

**Interpretação:**

- **Discriminador ($D$):** Treinado para maximizar a probabilidade de atribuir o rótulo correto às amostras reais e geradas.
- **Gerador ($G$):** Treinado para minimizar a capacidade do discriminador de distinguir entre amostras reais e geradas, tentando maximizar $\mathbb{E}_{z \sim p(z)} [\ln D(G(z))]$ em algumas implementações para melhorar o fluxo de gradientes.

**Equilíbrio de Nash:**

O equilíbrio é alcançado quando $p_G = p_{data}$, tornando $D(x) = 0.5$ para todas as $x$, indicando que o discriminador não pode diferenciar entre amostras reais e geradas.

---

### Treinamento Adversarial

O processo de treinamento envolve atualizações alternadas dos parâmetros do discriminador ($\phi$) e do gerador ($\mathbf{w}$) seguindo [5]:

1. **Atualização do Discriminador:**

   $$
   \Delta\phi = -\lambda \nabla_\phi E_n(\mathbf{w}, \phi)
   $$

   - **Objetivo:** Maximizar a função de perda em relação a $\phi$ para melhorar a capacidade de distinguir entre dados reais e gerados.
   - **Interpretação:** O discriminador ajusta seus parâmetros para aumentar a probabilidade de classificar corretamente as amostras.

2. **Atualização do Gerador:**

   $$
   \Delta\mathbf{w} = \lambda \nabla_\mathbf{w} E_n(\mathbf{w}, \phi)
   $$

   - **Objetivo:** Minimizar a função de perda em relação a $\mathbf{w}$ para gerar amostras mais realistas que enganem o discriminador.
   - **Interpretação:** O gerador ajusta seus parâmetros para produzir dados que o discriminador classifique como reais.

> 💡 **Insight Crucial**: O sinal oposto nos gradientes reflete a natureza adversarial do treinamento — o discriminador minimiza o erro enquanto o gerador o maximiza [5].

**Algoritmo de Treinamento:**

1. **Inicialização:** Iniciar $G$ e $D$ com pesos aleatórios.
2. **Para cada iteração:**
   - **Passo A (Treinar $D$):**
     - Amostrar minibatches de dados reais $\{x^{(i)}\}$.
     - Amostrar minibatches de vetores latentes $\{z^{(i)}\}$.
     - Atualizar $D$ usando o gradiente em relação a $\phi$.
   - **Passo B (Treinar $G$):**
     - Amostrar minibatches de $\{z^{(i)}\}$.
     - Atualizar $G$ usando o gradiente em relação a $\mathbf{w}$.

---

### Desafios e Soluções no Treinamento

#### 👎 **Desafios Principais:**

1. **Mode Collapse (Colapso de Modos):**

   - **Descrição:** O gerador converge para produzir um conjunto limitado de saídas, ignorando a diversidade dos dados reais [6].
   - **Consequência:** Redução na qualidade e variedade das amostras geradas.
   - **Exemplo:** Gerar apenas um tipo de rosto em vez de uma variedade de faces diferentes.

2. **Gradientes Instáveis:**

   - **Descrição:** O treinamento pode ser instável devido à natureza adversarial, levando a oscilações ou divergência [7].
   - **Causa:** Função de perda não estacionária e problemas de saturação de gradiente.
   - **Consequência:** Dificuldade em alcançar a convergência e o equilíbrio entre $G$ e $D$.

3. **Métricas de Progresso:**

   - **Descrição:** Ausência de uma métrica clara para avaliar o progresso durante o treinamento [7].
   - **Desafio:** As perdas do gerador e do discriminador não refletem diretamente a qualidade das amostras.
   - **Consequência:** Dificuldade em determinar quando interromper o treinamento ou comparar diferentes modelos.

#### 👍 **Soluções Propostas:**

1. **Wasserstein GAN (WGAN):**

   - **Descrição:** Utiliza a distância de Wasserstein como métrica alternativa para a função de perda [8].
   - **Benefícios:**
     - Fornece gradientes significativos mesmo quando as distribuições de $p_G$ e $p_{data}$ não se sobrepõem.
     - Melhora a estabilidade do treinamento.
   - **Função de Perda:**
     $$
     E_{WGAN} = \mathbb{E}_{x \sim p_{data}} [D(x)] - \mathbb{E}_{z \sim p(z)} [D(G(z))]
     $$
   - **Condições:**
     - O discriminador (agora chamado de crítico) deve ser 1-Lipschitz contínuo.

2. **Gradient Penalty (Penalidade de Gradiente):**

   - **Descrição:** Introduz uma penalidade no gradiente para estabilizar o treinamento [9].
   - **Implementação:**
     - Adiciona um termo à função de perda que penaliza desvios na norma do gradiente em relação a 1.
     - **Termo de Penalidade:**
       $$
       E_{GP} = \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}} \left[ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \right]
       $$
     - Onde $\hat{x}$ é uma amostra interpolada entre $x \sim p_{data}$ e $G(z)$.
   - **Benefícios:**
     - Promove um gradiente estável e limita a função discriminadora.
     - Evita a necessidade de clipping dos pesos do discriminador.

---

### [Seção Teórica Avançada] Análise do Ponto de Equilíbrio de Nash

**Pergunta:** Como podemos provar que o ponto de equilíbrio de uma GAN ocorre quando a distribuição do gerador coincide com a distribuição real dos dados?

**Resposta:** A prova envolve a análise da função de erro da GAN no limite de redes flexíveis infinitas [10]. Vamos considerar a forma contínua da função de erro e mostrar que, no ponto de equilíbrio, $p_G(x) = p_{data}(x)$.

**Passo 1: Definir a Função de Erro Contínua**

$$
E(p_G, D) = - \int p_{data}(x) \ln D(x) \, dx - \int p_G(x) \ln (1 - D(x)) \, dx
$$

**Passo 2: Encontrar o Discriminador Ótimo $D^*$ para um Gerador Fixo $p_G$**

Para maximizar $E$ em relação a $D(x)$, derivamos em relação a $D(x)$ e igualamos a zero:

$$
\frac{\delta E}{\delta D(x)} = -\frac{p_{data}(x)}{D(x)} + \frac{p_G(x)}{1 - D(x)} = 0
$$

Resolvendo para $D(x)$, obtemos:

$$
D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}
$$

**Passo 3: Substituir $D^*(x)$ na Função de Erro para Obter $E(p_G, D^*)$**

$$
E(p_G, D^*) = - \int p_{data}(x) \ln \left( \frac{p_{data}(x)}{p_{data}(x) + p_G(x)} \right) dx - \int p_G(x) \ln \left( \frac{p_G(x)}{p_{data}(x) + p_G(x)} \right) dx
$$

**Passo 4: Mostrar que $E(p_G, D^*)$ é Mínimo Quando $p_G = p_{data}$**

Observamos que $E(p_G, D^*)$ é proporcional à divergência de Jensen-Shannon (JS) entre $p_{data}$ e $p_G$:

$$
E(p_G, D^*) = -\ln(4) + 2 \cdot JS(p_{data} \| p_G)
$$

A divergência JS atinge o valor mínimo de zero quando $p_G = p_{data}$. Portanto, o erro $E(p_G, D^*)$ é minimizado quando as distribuições coincidem.

**Conclusão:**

O ponto de equilíbrio de Nash na GAN ocorre quando o gerador reproduz exatamente a distribuição dos dados reais, ou seja, $p_G(x) = p_{data}(x)$. Nesse ponto, o discriminador não consegue distinguir entre amostras reais e geradas, atribuindo uma probabilidade de 0.5 para ambas.

---

### [Seção Teórica Avançada] Divergência KL em Distribuições Gaussianas para GANs

**Pergunta:** Como podemos derivar a forma fechada da divergência KL entre duas distribuições normais no contexto de GANs?

**Resposta:** ==Vamos demonstrar a derivação da divergência KL entre duas distribuições normais unidimensionais com a mesma variância $\epsilon^2$, mas médias diferentes $\theta$ e $\theta_0$.==

#### **Passo 1: Definição das Distribuições**

- **Distribuição do Modelo (Gerador):** $p_\theta(x) = N(x \mid \theta, \epsilon^2)$
- **Distribuição dos Dados (Real):** $p_{data}(x) = N(x \mid \theta_0, \epsilon^2)$

#### **Passo 2: Fórmula da Divergência KL**

A divergência KL de $p_\theta$ para $p_{data}$ é dada por:

$$
KL(p_\theta \| p_{data}) = \int p_\theta(x) \ln \left( \frac{p_\theta(x)}{p_{data}(x)} \right) dx
$$

#### **Passo 3: Substituir as Funções de Densidade**

As funções de densidade das distribuições normais são:

$$
p(x) = \frac{1}{\sqrt{2\pi \epsilon^2}} \exp\left( -\frac{(x - \mu)^2}{2\epsilon^2} \right)
$$

Substituindo na fórmula da divergência KL:

$$
KL(p_\theta \| p_{data}) = \int p_\theta(x) \left[ -\frac{(x - \theta)^2}{2\epsilon^2} + \frac{(x - \theta_0)^2}{2\epsilon^2} \right] dx
$$

#### **Passo 4: Simplificar a Expressão**

Simplificando a diferença nos expoentes:

$$
KL(p_\theta \| p_{data}) = \frac{1}{2\epsilon^2} \int p_\theta(x) \left[ (x - \theta_0)^2 - (x - \theta)^2 \right] dx
$$

Expandindo os quadrados:

$$
(x - \theta_0)^2 - (x - \theta)^2 = [ (x^2 - 2 x \theta_0 + \theta_0^2) - (x^2 - 2 x \theta + \theta^2) ] = 2 x (\theta - \theta_0) + \theta_0^2 - \theta^2
$$

#### **Passo 5: Calcular a Expectativa**

A expectativa é tomada sobre $p_\theta(x) = N(x \mid \theta, \epsilon^2)$, então:

- $E_{x \sim p_\theta}[x] = \theta$
- $E_{x \sim p_\theta}[x^2] = \theta^2 + \epsilon^2$

Calculando a expectativa:

$$
\int p_\theta(x) \cdot 2 x (\theta - \theta_0) dx = 2 (\theta - \theta_0) E_{x \sim p_\theta}[x] = 2 (\theta - \theta_0) \theta
$$

$$
\int p_\theta(x) (\theta_0^2 - \theta^2) dx = (\theta_0^2 - \theta^2)
$$

Portanto:

$$
KL(p_\theta \| p_{data}) = \frac{1}{2\epsilon^2} \left[ 2 (\theta - \theta_0) \theta + (\theta_0^2 - \theta^2) \right]
$$

Simplificando:

$$
KL(p_\theta \| p_{data}) = \frac{1}{2\epsilon^2} \left[ 2 \theta (\theta - \theta_0) + \theta_0^2 - \theta^2 \right]
$$

$$
= \frac{1}{2\epsilon^2} \left[ 2 \theta (\theta - \theta_0) + \theta_0^2 - \theta^2 \right]
$$

$$
= \frac{1}{2\epsilon^2} \left[ 2 \theta^2 - 2 \theta \theta_0 + \theta_0^2 - \theta^2 \right]
$$

$$
= \frac{1}{2\epsilon^2} \left[ \theta^2 - 2 \theta \theta_0 + \theta_0^2 \right]
$$

$$
= \frac{1}{2\epsilon^2} (\theta - \theta_0)^2
$$

#### **Conclusão**

Demonstramos que:

$$
KL(p_\theta \| p_{data}) = \frac{(\theta - \theta_0)^2}{2\epsilon^2}
$$

> 💡 **Insight:** ==A divergência KL entre duas distribuições normais unidimensionais com a mesma variância é proporcional ao quadrado da diferença entre suas médias, dividido pela variância.==

---

### [Seção Teórica Avançada] Implicações para o Treinamento de GANs

**Pergunta:** Como este resultado da divergência KL impacta o treinamento de GANs?

**Análise:**

1. **Sensibilidade à Diferença de Médias:**

   - A divergência KL é proporcional a $(\theta - \theta_0)^2$.
   - Pequenas diferenças entre $\theta$ e $\theta_0$ resultam em pequenas divergências, mas grandes diferenças aumentam significativamente a divergência.

2. **Influência da Variância ($\epsilon^2$):**

   - A divergência KL é inversamente proporcional à variância.
   - Variâncias pequenas ($\epsilon^2$ pequeno) amplificam a divergência para uma mesma diferença de médias.
   - Isso indica que distribuições com variâncias menores são mais sensíveis a desvios na média.

3. **Gradiente em Relação a $\theta$:**

   - O gradiente da divergência KL em relação a $\theta$ é:

     $$
     \frac{\partial}{\partial \theta} KL(p_\theta \| p_{data}) = \frac{\theta - \theta_0}{\epsilon^2}
     $$

   - ==Indica que o gradiente é proporcional à diferença das médias e inversamente proporcional à variância.==

**Implicações Práticas no Treinamento de GANs:**

- **Estabilidade do Treinamento:**

  - ==Em casos onde a variância é pequena, os gradientes podem ser grandes, levando a oscilações ou divergência durante o treinamento.==
  - Ajustar a variância ou implementar técnicas de regularização pode ajudar a estabilizar o treinamento.

- **Aprendizado Eficiente:**

  - O gradiente fornece uma direção clara para ajustar $\theta$ em direção a $\theta_0$, facilitando a convergência.
  - Em espaços de alta dimensionalidade, considerar as covariâncias entre dimensões torna-se importante.

- **Modelagem de Distribuições Complexas:**

  - ==Para distribuições mais complexas do que Gaussianas univariadas, o princípio se estende, mas requer consideração das matrizes de covariância e possíveis correlações entre variáveis.==

---

### [Seção de Aplicação] Extensão para Casos Multidimensionais

**Pergunta:** Como este resultado se generaliza para distribuições gaussianas multivariadas no contexto de GANs?

**Resposta:**

Para distribuições Gaussianas multivariadas com médias $\boldsymbol{\mu}_1$, $\boldsymbol{\mu}_2$ e matriz de covariância comum $\Sigma$, a divergência KL é dada por:

$$
KL(N(\boldsymbol{\mu}_1, \Sigma) \| N(\boldsymbol{\mu}_2, \Sigma)) = \frac{1}{2} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)
$$

**Interpretação:**

- **Métrica de Mahalanobis:**
  - ==A divergência KL é proporcional ao quadrado da distância de Mahalanobis entre as médias.==
  - A matriz de covariância $\Sigma$ leva em conta a escala e a orientação dos dados.

- **Considerações para GANs:**
  - ==No treinamento de GANs em espaços de alta dimensionalidade, é crucial considerar as correlações entre diferentes dimensões.==
  - ==O gerador deve aprender não apenas as médias das distribuições, mas também as relações entre as variáveis.==

**Implicações Práticas:**

- **Regularização:**
  - A introdução de termos de regularização que penalizam diferenças nas covariâncias pode melhorar a qualidade das amostras geradas.
- **Modelagem de Dependências:**
  - Arquiteturas avançadas, como redes neurais profundas, são capazes de capturar dependências complexas entre variáveis, aproximando melhor a distribuição real dos dados.

---

### [Seção Teórica Avançada] Análise da Convexidade e Convergência

**Pergunta:** Como a geometria da função objetivo da GAN afeta sua convergência?

**Análise:**

- **Natureza Não-Convexa:**
  - A função objetivo das GANs é não-convexa em relação aos parâmetros do gerador e do discriminador.
  - Isso significa que existem múltiplos mínimos locais e pontos de sela, dificultando a otimização.

- **Oscilações no Treinamento:**
  - Devido à natureza adversarial, o treinamento pode entrar em ciclos ou oscilações, em vez de convergir para um ponto estável.
  - A interação dinâmica entre $G$ e $D$ cria um campo de vetores que pode não ser gradiente de uma função escalar.

- **Análise da Matriz Hessiana:**
  - A matriz Hessiana combinada do sistema não é definida positiva, o que implica na presença de pontos de sela.
  - Isso afeta a estabilidade dos métodos de otimização baseados em gradiente.

**Implicações:**

- **Métodos de Otimização Alternativos:**
  - Técnicas como otimização baseada em momentos, taxas de aprendizado adaptativas e regularização podem ajudar na convergência.
  - Algoritmos que consideram a dinâmica do sistema, como otimização adversarial, podem ser mais eficazes.

- **Estabilidade do Treinamento:**
  - Ajustes cuidadosos nos hiperparâmetros e na arquitetura das redes podem mitigar problemas de instabilidade.

---

### [Seção Teórica Avançada] Dinâmica do Gradiente em GANs

**Pergunta:** Como podemos analisar a dinâmica do sistema de equações diferenciais que governa o treinamento de GANs?

**Análise:**

- **Formulação do Sistema Dinâmico:**
  - As atualizações dos parâmetros podem ser vistas como um sistema de equações diferenciais ordinárias (EDOs):

    $$
    \begin{cases}
    \dot{\theta}_G = \nabla_{\theta_G} V(G, D) \\
    \dot{\theta}_D = -\nabla_{\theta_D} V(G, D)
    \end{cases}
    $$

  - Onde $\theta_G$ e $\theta_D$ são os parâmetros do gerador e do discriminador, respectivamente.

- **Comportamento do Sistema:**
  - **Pontos Fixos:** Pontos onde $\dot{\theta}_G = 0$ e $\dot{\theta}_D = 0$ correspondem a equilíbrios.
  - **Análise de Estabilidade:** Estudar a estabilidade desses pontos usando teoria de sistemas dinâmicos.

- **Exemplo Simplificado:**
  - Considere uma função de custo quadrática simplificada:

    $$
    V(\theta_G, \theta_D) = \theta_G \theta_D
    $$

  - As equações de movimento são:

    $$
    \dot{\theta}_G = \theta_D \\
    \dot{\theta}_D = -\theta_G
    $$

  - Este sistema tem soluções oscilatórias, indicando que os parâmetros seguem trajetórias circulares no espaço de fase.

**Implicações:**

- **Oscilações no Treinamento:**
  - As GANs podem apresentar comportamento cíclico, com o gerador e o discriminador melhorando alternadamente.
- **Métodos de Estabilização:**
  - Introduzir termos de regularização ou modificar a função de perda para amortecer as oscilações.
- **Análise Teórica:**
  - Compreender a dinâmica pode orientar o design de algoritmos de otimização mais eficazes.

---

### [Seção Teórica Avançada] Teoria da Informação em GANs

**Pergunta:** Como a divergência de Jensen-Shannon (JS) se relaciona com a divergência KL no contexto de GANs?

**Análise:**

- **Definição da Divergência de Jensen-Shannon:**

  $$
  JS(p \| q) = \frac{1}{2} KL\left( p \left\| \frac{p + q}{2} \right. \right) + \frac{1}{2} KL\left( q \left\| \frac{p + q}{2} \right. \right)
  $$

- **Relação com a Divergência KL:**

  - A divergência JS é uma versão suavizada e simétrica da divergência KL.
  - Ao contrário da divergência KL, a divergência JS é sempre finita e definida mesmo quando os suportes das distribuições não se sobrepõem.

- **Aplicação em GANs:**

  - A função objetivo original das GANs é proporcional à divergência JS entre $p_{data}$ e $p_G$.
  - Minimizar a divergência JS promove a aproximação entre as distribuições real e gerada.

**Propriedades Importantes:**

- **Simetria:**

  $$
  JS(p \| q) = JS(q \| p)
  $$

- **Limites:**

  - $0 \leq JS(p \| q) \leq \ln 2$

- **Suavidade:**

  - A divergência JS é mais suave que a divergência KL, evitando problemas de gradiente inexistente quando as distribuições não se sobrepõem.

**Implicações no Treinamento:**

- **Gradientes Significativos:**

  - A divergência JS fornece gradientes úteis mesmo em estágios iniciais do treinamento.
- **Estabilidade:**

  - Contribui para uma maior estabilidade no treinamento em comparação com a divergência KL direta.

---

### [Seção Teórica Avançada] Regularização de Gradiente em GANs

**Pergunta:** Como a penalidade de gradiente afeta a estabilidade do treinamento?

**Análise:**

- **Objetivo da Penalidade de Gradiente:**

  - Forçar o discriminador a ser uma função 1-Lipschitz contínua.
  - Estabilizar o treinamento garantindo que os gradientes do discriminador não explodam ou desapareçam.

- **Função de Perda com Penalidade de Gradiente (WGAN-GP):**

  $$
  E_{WGAN-GP}(w, \phi) = \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}} [D(\hat{x})] - \mathbb{E}_{z \sim p(z)} [D(G(z))] + \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}} \left[ (\| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1)^2 \right]
  $$

  - Onde $\hat{x}$ é uma amostra interpolada entre $x \sim p_{data}$ e $G(z)$.

- **Impacto da Penalidade:**

  - **Estabilização dos Gradientes:**
    - Evita que os gradientes do discriminador se tornem muito grandes ou pequenos.
  - **Treinamento Mais Suave:**
    - Promove uma superfície de perda mais suave, facilitando a otimização.
  - **Cumprimento da Condição de Lipschitz:**
    - Assegura que o discriminador satisfaça a condição necessária para que a distância de Wasserstein seja válida.

**Implementação Prática:**

- **Cálculo do Gradiente:**

  - O gradiente é computado em relação às amostras interpoladas, o que requer o uso de técnicas de diferenciação automática.
- **Escolha do $\lambda$:**

  - O hiperparâmetro $\lambda$ controla a força da penalidade e deve ser ajustado para equilibrar a regularização e a aprendizagem.

---

### [Seção Teórica Avançada] Métricas de Convergência

**Pergunta:** Como podemos quantificar matematicamente a convergência de uma GAN?

**Análise:**

1. **Distância de Wasserstein (WGAN):**

   - **Definição:**

     $$
     W_1(p_{data}, p_G) = \inf_{\gamma \in \Pi(p_{data}, p_G)} \mathbb{E}_{(x, y) \sim \gamma} [\| x - y \|]
     $$

     - Onde $\Pi(p_{data}, p_G)$ é o conjunto de todas as distribuições conjuntas com marginais $p_{data}$ e $p_G$.

   - **Interpretação:**

     - Mede o "custo mínimo" para transportar massa de probabilidade de $p_G$ para $p_{data}$.

2. **Inception Score (IS):**

   - **Definição:**

     $$
     IS = \exp \left( \mathbb{E}_{x \sim p_G} [KL(p(y | x) \| p(y))] \right)
     $$

     - Onde $p(y | x)$ é a probabilidade de classificação da imagem $x$ em uma classe $y$ por um modelo pretreinado (por exemplo, Inception v3).
     - $p(y)$ é a distribuição marginal sobre as classes.

   - **Interpretação:**

     - Avalia a qualidade e diversidade das imagens geradas.
     - Um alto IS indica que as imagens são diversas (alta entropia em $p(y)$) e facilmente reconhecíveis (baixa entropia em $p(y | x)$).

3. **Fréchet Inception Distance (FID):**

   - **Definição:**

     $$
     FID = \| \mu_1 - \mu_2 \|^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2 (\Sigma_1 \Sigma_2)^{1/2})
     $$

     - Onde $(\mu_1, \Sigma_1)$ são a média e covariância das características extraídas das imagens reais.
     - $(\mu_2, \Sigma_2)$ são a média e covariância das características extraídas das imagens geradas.

   - **Interpretação:**

     - Mede a diferença entre as distribuições das características das imagens reais e geradas.
     - Valores menores indicam maior similaridade entre as distribuições.

**Considerações:**

- **Complementaridade das Métricas:**

  - Nenhuma métrica captura todos os aspectos da qualidade das amostras.
  - Usar múltiplas métricas fornece uma avaliação mais abrangente.

- **Dependência de Modelos Pretreinados:**

  - IS e FID dependem de modelos pretreinados, o que pode introduzir viés.

- **Avaliação Humana:**

  - Em muitos casos, a avaliação visual humana é necessária para complementar as métricas quantitativas.

---

# Referências

[1] Goodfellow, I., et al. "Generative Adversarial Networks." *Advances in Neural Information Processing Systems*, 2014.

[2] Mirza, M., and Osindero, S. "Conditional Generative Adversarial Nets." *arXiv preprint arXiv:1411.1784*, 2014.

[3] Kingma, D. P., and Welling, M. "Auto-Encoding Variational Bayes." *arXiv preprint arXiv:1312.6114*, 2013.

[4] Nowozin, S., Cseke, B., and Tomioka, R. "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization." *Advances in Neural Information Processing Systems*, 2016.

[5] Salimans, T., et al. "Improved Techniques for Training GANs." *Advances in Neural Information Processing Systems*, 2016.

[6] Metz, L., et al. "Unrolled Generative Adversarial Networks." *arXiv preprint arXiv:1611.02163*, 2016.

[7] Arjovsky, M., and Bottou, L. "Towards Principled Methods for Training Generative Adversarial Networks." *arXiv preprint arXiv:1701.04862*, 2017.

[8] Arjovsky, M., Chintala, S., and Bottou, L. "Wasserstein GAN." *arXiv preprint arXiv:1701.07875*, 2017.

[9] Gulrajani, I., et al. "Improved Training of Wasserstein GANs." *Advances in Neural Information Processing Systems*, 2017.

[10] Goodfellow, I. "NIPS 2016 Tutorial: Generative Adversarial Networks." *arXiv preprint arXiv:1701.00160*, 2016.
