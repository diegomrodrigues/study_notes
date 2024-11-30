# A Função de Demanda Monetária de Goldfeld: Uma Análise Teórica de Equações de Diferenças

<imagem: Um gráfico tridimensional mostrando a relação entre demanda monetária (m), renda (y) e taxa de juros (r), com linhas de tendência temporal mostrando o efeito defasado>

### Introdução

A **Função de Demanda Monetária de Goldfeld** representa um marco significativo na modelagem econométrica de demanda por moeda utilizando equações de diferenças [1]. Este modelo, desenvolvido por Stephen M. Goldfeld em 1973, ==relaciona a demanda por moeda a variáveis econômicas fundamentais e incorpora elementos dinâmicos através de um termo autorregressivo.==

### Conceitos Fundamentais

| Conceito                         | Explicação                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Equação de Diferenças Linear** | ==Uma expressão que relaciona uma variável $y_t$ com seus valores defasados e outras variáveis explicativas $w_t$== [1]. Formalmente: $y_t = \phi y_{t-1} + w_t$ |
| **Demanda Monetária**            | ==Quantidade de moeda que os agentes econômicos desejam manter, modelada por Goldfeld como função da renda, taxa de juros e valores defasados [2]== |
| **Dinâmica Temporal**            | Processo pelo qual choques em variáveis explicativas se propagam ao longo do tempo através do termo autorregressivo [3] |

> ⚠️ **Nota Importante**: A estabilidade do modelo depende criticamente do valor do coeficiente autorregressivo $\phi$. Se $|\phi| < 1$, o sistema é estável [4].

### Formulação Matemática do Modelo de Goldfeld

O modelo original de Goldfeld, conforme apresentado no contexto [5], é expresso como:

$$
m_t = 0.27 + 0.72m_{t-1} + 0.19I_t - 0.045r_{bt} - 0.019r_{ct}
$$

Onde:
- $m_t$: logaritmo das holdings reais de moeda do público
- $I_t$: logaritmo da renda agregada real
- $r_{bt}$: logaritmo da taxa de juros em contas bancárias
- $r_{ct}$: logaritmo da taxa de juros em commercial papers

==Esta equação pode ser reescrita na forma canônica de uma equação de diferenças de primeira ordem [6]:==
$$
m_t = \phi m_{t-1} + w_t
$$

Onde:
- $\phi = 0.72$
- $w_t = 0.27 + 0.19I_t - 0.045r_{bt} - 0.019r_{ct}$

### Análise de Estabilidade Dinâmica

A estabilidade do modelo de Goldfeld pode ser analisada através do coeficiente autorregressivo $\phi = 0.72$ [7]. Como $|\phi| < 1$, o sistema é estável, implicando que:

1. **Convergência**: ==Choques temporários têm efeitos que diminuem geometricamente ao longo do tempo==
2. **Multiplicador de Longo Prazo**: O efeito total de uma mudança permanente em $w_t$ é dado por $1/(1-\phi) = 1/(1-0.72) = 3.57$ [8]

> 💡 **Insight**: A magnitude do multiplicador de longo prazo indica que alterações permanentes nas variáveis explicativas têm um efeito amplificado sobre a demanda por moeda.

### Propriedades dos Multiplicadores Dinâmicos

O modelo de Goldfeld permite calcular multiplicadores dinâmicos que descrevem como mudanças nas variáveis explicativas afetam a demanda por moeda ao longo do tempo [9]. 

#### Multiplicadores de Impacto

Para uma variação unitária em cada componente de $w_t$, temos os seguintes multiplicadores de impacto [10]:

| Variável                          | Multiplicador de Impacto |
| --------------------------------- | ------------------------ |
| Renda ($I_t$)                     | 0.19                     |
| Taxa de Juros Bancária ($r_{bt}$) | -0.045                   |
| Taxa Commercial Paper ($r_{ct}$)  | -0.019                   |

#### Multiplicadores Dinâmicos

==Os efeitos ao longo do tempo são dados por [11]:==
$$
\frac{\partial m_{t+j}}{\partial w_t} = \phi^j
$$

Para o modelo de Goldfeld, com $\phi = 0.72$, temos:

$$
\frac{\partial m_{t+j}}{\partial I_t} = 0.19 \times (0.72)^j
$$

> ⚠️ **Ponto Crucial**: A resposta dinâmica decai geometricamente a uma taxa de 0.72 por período [12].

### Análise de Valor Presente

==O valor presente dos efeitos futuros de uma mudança em $w_t$ pode ser calculado como [13]:==
$$
\sum_{j=0}^{\infty} \beta^j \frac{\partial m_{t+j}}{\partial w_t} = \sum_{j=0}^{\infty} \beta^j \phi^j = \frac{1}{1 - \beta\phi}
$$

Onde $\beta = 1/(1+r)$ é o fator de desconto.

### [Pergunta Teórica]: Como a Estrutura de Defasagens do Modelo de Goldfeld Afeta a Velocidade de Ajustamento da Demanda Monetária?

**Resposta:**

A velocidade de ajustamento é determinada pelo coeficiente autorregressivo $\phi = 0.72$ [14]. O ajuste para o equilíbrio de longo prazo segue um processo geométrico:

1. **Ajuste Imediato**: $(1-\phi) = 28\%$ do ajuste ocorre no primeiro período
2. **Ajuste Subsequente**: $\phi(1-\phi) = 20.16\%$ no segundo período
3. **Meia-Vida**: O tempo necessário para completar 50% do ajuste é dado por:

$$
t_{1/2} = \frac{\ln(0.5)}{\ln(\phi)} = \frac{\ln(0.5)}{\ln(0.72)} \approx 2.11 \text{ períodos}
$$

### [Problema Teórico Avançado]: Demonstre a Convergência do Multiplicador de Longo Prazo no Modelo de Goldfeld

**Solução:**

==Para provar a convergência do multiplicador de longo prazo, devemos mostrar que [15]:==

1) A soma dos efeitos converge:
   
$$
\sum_{j=0}^{\infty} \phi^j = \frac{1}{1-\phi}
$$

Prova:
- Seja $S_n = \sum_{j=0}^{n} \phi^j$
- Então $\phi S_n = \sum_{j=1}^{n+1} \phi^j$
- Subtraindo: $(1-\phi)S_n = 1 - \phi^{n+1}$
- Logo: $S_n = \frac{1-\phi^{n+1}}{1-\phi}$
- Como $|\phi| < 1$, $\lim_{n \to \infty} \phi^{n+1} = 0$
- Portanto: $\lim_{n \to \infty} S_n = \frac{1}{1-\phi}$

2) O multiplicador total é:

$$
\frac{0.19}{1-0.72} = 0.68
$$

Este valor representa a elasticidade-renda de longo prazo da demanda por moeda [16].

[Continua...]

[Continuação do resumo sobre a Função de Demanda Monetária de Goldfeld]

### Análise de Estabilidade Estrutural

A estabilidade estrutural do modelo de Goldfeld pode ser analisada através da decomposição dos efeitos dinâmicos [17]. 

#### Decomposição da Dinâmica

O processo pode ser decomposto em componentes permanentes e transitórios [18]:

$$
m_t = m_t^* + c_t
$$

Onde:
- $m_t^*$ é o componente de equilíbrio de longo prazo
- $c_t$ é o componente cíclico ou transitório

> ✔️ **Destaque**: A decomposição permite identificar o comportamento assintótico do modelo e sua resposta a choques [19].

### [Pergunta Teórica Avançada]: Qual é a Relação Entre os Autovalores do Sistema e a Persistência dos Choques Monetários?

**Resposta:**

A persistência dos choques é determinada pela estrutura dos autovalores [20]. Para o modelo de Goldfeld:

1. **Autovalor Único**: $\lambda = 0.72$ (por ser uma equação de primeira ordem)

2. **Função de Resposta ao Impulso**:
   
$$
\frac{\partial m_{t+j}}{\partial \epsilon_t} = \lambda^j = (0.72)^j
$$

3. **Persistência Total**:

$$
\sum_{j=0}^{\infty} \lambda^j = \frac{1}{1-0.72} = 3.57
$$

Este valor indica que um choque unitário tem um efeito acumulado de 3.57 unidades [21].

### Propriedades Assintóticas

O comportamento assintótico do modelo apresenta características importantes [22]:

1. **Convergência Geométrica**:
   
$$
\lim_{j \to \infty} \frac{\partial m_{t+j}}{\partial w_t} = \lim_{j \to \infty} \phi^j = 0
$$

2. **Estacionariedade**:
   - A variância incondicional existe e é finita
   - A função de autocorrelação decai geometricamente

> ❗ **Ponto de Atenção**: A estacionariedade é garantida pela condição $|\phi| < 1$ [23].

### [Análise Teórica]: Derivação da Função de Verossimilhança do Modelo de Goldfeld

Considerando o modelo completo [24]:

$$
m_t = 0.27 + 0.72m_{t-1} + 0.19I_t - 0.045r_{bt} - 0.019r_{ct} + \epsilon_t
$$

Assumindo $\epsilon_t \sim N(0,\sigma^2)$, a função de log-verossimilhança é:

$$
\mathcal{L}(\theta) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{t=1}^{n} (m_t - \beta'x_t)^2
$$

Onde:
- $\theta = (\beta',\sigma^2)'$
- $\beta = (0.27, 0.72, 0.19, -0.045, -0.019)'$
- $x_t = (1, m_{t-1}, I_t, r_{bt}, r_{ct})'$

### Implicações Práticas e Limitações

#### 👍 Vantagens
- Captura ajustamento parcial da demanda por moeda [25]
- Permite análise de efeitos dinâmicos [26]
- Estrutura matemática tratável [27]

#### 👎 Desvantagens
- Assume coeficientes constantes ao longo do tempo [28]
- Não captura não-linearidades [29]
- Restrições na especificação da dinâmica [30]

### Conclusão

O modelo de Goldfeld representa uma contribuição fundamental para a análise da demanda monetária, combinando rigor teórico com aplicabilidade empírica [31]. Sua estrutura de equação de diferenças de primeira ordem permite uma análise detalhada das dinâmicas de ajustamento, enquanto mantém tratabilidade matemática [32].

### Referências

[1] "This book is concerned with the dynamic consequences of events over time..." *(Differential Equations_16-40.pdf)*

[2] "Goldfeld's model related the log of the real money holdings of the public..." *(Differential Equations_16-40.pdf)*

[3] "The dynamic multiplier [1.1.10] depends only on j, the length of time..." *(Differential Equations_16-40.pdf)*

[...]

[32] "The system is thus stable whenever $(\phi_1, \phi_2)$ lies within the triangular region..." *(Differential Equations_16-40.pdf)*

[Nota: As referências continuariam com todos os trechos relevantes do contexto utilizados]