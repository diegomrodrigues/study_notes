**1. Equações de Diferenças**

As equações de diferenças desempenham um papel crucial na análise de sistemas dinâmicos discretos, sendo fundamentais na modelagem de processos que evoluem ao longo do tempo em intervalos discretos. Elas são ferramentas matemáticas essenciais em diversas áreas, como economia, engenharia, física e biologia, permitindo descrever a evolução temporal de variáveis de interesse.

**1.1 Equações de Diferenças de Primeira Ordem**

* **Equação Linear de Primeira Ordem**: Uma equação de diferenças linear de primeira ordem é expressa como:

  $$y_t = \phi y_{t-1} + w_t$$

  Nesta equação:

  - $y_t$ é o valor atual da variável dependente.
  - $y_{t-1}$ é o valor defasado em um período.
  - $\phi$ é o coeficiente autorregressivo, que indica a influência do passado sobre o presente.
  - $w_t$ é um termo de entrada ou perturbação externa no período $t$.

  Essa forma captura a ideia de que o valor atual de uma variável é influenciado por seu valor passado e por uma entrada externa. A linearidade implica que as mudanças em $y_t$ são proporcionais às mudanças em $y_{t-1}$ e $w_t$.

* **Função de Demanda Monetária de Goldfeld**: O modelo de demanda monetária proposto por Goldfeld (1973) é um exemplo prático de aplicação de uma equação de diferenças de primeira ordem. A função de demanda monetária pode ser escrita como:

  $$m_t = \alpha + \beta y_t + \gamma r_t + \delta m_{t-1} + \epsilon_t$$

  Onde:

  - $m_t$ é a quantidade demandada de moeda.
  - $y_t$ é a renda.
  - $r_t$ é a taxa de juros.
  - $m_{t-1}$ é a quantidade de moeda demandada no período anterior.
  - $\epsilon_t$ é um termo de erro aleatório.

  Simplificando, podemos resumir as variáveis explanatórias em um único termo $w_t$, obtendo uma equação de diferenças de primeira ordem.

* **Análise de Sistemas Dinâmicos**: O objetivo é entender como mudanças em $w_t$ afetam $y_t$ ao longo do tempo. Isso envolve estudar a estabilidade do sistema, sua resposta a choques e a evolução temporal das variáveis.

**Solução de uma Equação de Diferenças por Substituição Recursiva**

* **Método de Substituição Recursiva**: A solução da equação de diferenças é obtida substituindo iterativamente $y_{t-1}$:

  $$y_t = \phi y_{t-1} + w_t$$
  $$y_{t-1} = \phi y_{t-2} + w_{t-1}$$
  $$\vdots$$
  $$y_1 = \phi y_0 + w_1$$

  Substituindo sucessivamente, chegamos à solução geral:

  $$y_t = \phi^t y_0 + \sum_{j=0}^{t-1} \phi^j w_{t-1-j}$$

  Essa expressão mostra que $y_t$ depende exponencialmente do valor inicial $y_0$ e da soma ponderada das entradas passadas $w_t$.

* **Solução da Equação de Primeira Ordem**: A solução obtida permite analisar o comportamento de $y_t$ ao longo do tempo, considerando diferentes valores de $\phi$ e as características da sequência $w_t$.

**Multiplicadores Dinâmicos**

* **Derivada Parcial e Multiplicador Dinâmico**: O multiplicador dinâmico mede a sensibilidade de $y_t$ a uma variação em $w_{t-j}$:

  $$\frac{\partial y_t}{\partial w_{t-j}} = \phi^j$$

  Isso indica como um choque em $w_{t-j}$ afeta $y_t$, considerando a persistência temporal capturada por $\phi$.

* **Invariância Temporal dos Multiplicadores Dinâmicos**: Em sistemas lineares com coeficientes constantes, os multiplicadores dependem apenas do lag $j$, não do tempo absoluto $t$. Isso é consequência da propriedade de estacionariedade estrita desses sistemas.

* **Exemplo com o Modelo de Goldfeld**: Aplicando o modelo de demanda monetária, podemos calcular os multiplicadores dinâmicos específicos e interpretar o efeito de políticas econômicas ou mudanças nas taxas de juros sobre a demanda por moeda ao longo do tempo.

* **Respostas Dinâmicas Baseadas em $\phi$**:

  - **Estável ($0 < |\phi| < 1$)**: O sistema converge para um equilíbrio, e os efeitos dos choques diminuem exponencialmente.
  - **Explosivo ($|\phi| > 1$)**: O sistema diverge, com os efeitos dos choques aumentando ao longo do tempo.
  - **Caso Limite ($\phi = 1$)**: O sistema é não estacionário (caminhada aleatória), e os choques têm efeitos permanentes.

**Cálculos de Longo Prazo e Valor Presente**

* **Cálculo do Valor Presente**: O valor presente de uma sequência futura de $y_t$ é dado por:

  $$PV = \sum_{s=0}^{\infty} \beta^s y_{t+s}$$

  Onde $\beta = \frac{1}{1 + r}$ é o fator de desconto e $r$ é a taxa de juros. Esse cálculo é fundamental em finanças para avaliar o valor atual de fluxos de caixa futuros.

* **Multiplicador do Valor Presente**: O efeito de uma mudança em $w_t$ sobre o valor presente de $y_t$ é:

  $$\frac{\partial PV}{\partial w_t} = \sum_{j=0}^{\infty} \beta^j \frac{\partial y_{t+j}}{\partial w_t}$$

  Substituindo o multiplicador dinâmico, obtemos uma expressão que integra os efeitos descontados dos choques ao longo do tempo.

* **Função de Resposta ao Impulso**: A sequência $\{\phi^j\}$ representa a função de resposta ao impulso, descrevendo a evolução temporal do sistema em resposta a um choque unitário em $w_t$.

* **Efeito de Longo Prazo de uma Mudança Permanente**: Quando $w_t$ sofre uma mudança permanente de magnitude $\Delta w$, o efeito de longo prazo sobre $y_t$ é:

  $$\Delta y = \frac{\Delta w}{1 - \phi}$$

  Isso resulta da soma da progressão geométrica infinita dos efeitos dos choques persistentes.

* **Efeito Cumulativo de uma Mudança Transitória**: Para uma mudança única em $w_t$, o efeito total acumulado ao longo do tempo é o mesmo que o efeito de longo prazo de uma mudança permanente, devido à propriedade das séries geométricas.

**1.2 Equações de Diferenças de Ordem $p$**

* **Equação Linear de Ordem $p$**: Uma equação de diferenças linear de ordem $p$ é dada por:

  $$y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + w_t$$

  Essa equação captura dependências mais complexas, permitindo que $y_t$ seja influenciado por múltiplas defasagens. Isso é essencial para modelar processos com memória longa ou características cíclicas.

* **Representação Vetorial da Equação de Ordem $p$**: A equação escalar pode ser reescrita em forma vetorial, definindo um vetor de estado:

  $$\boldsymbol{\xi}_t = \begin{bmatrix} y_t \\ y_{t-1} \\ \vdots \\ y_{t-p+1} \end{bmatrix}$$

  E uma matriz de transição $\mathbf{F}$:

  $$\mathbf{F} = \begin{bmatrix} \phi_1 & \phi_2 & \dots & \phi_{p-1} & \phi_p \\ 1 & 0 & \dots & 0 & 0 \\ 0 & 1 & \dots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \dots & 1 & 0 \end{bmatrix}$$

  Então, a equação vetorial é:

  $$\boldsymbol{\xi}_t = \mathbf{F} \boldsymbol{\xi}_{t-1} + \begin{bmatrix} w_t \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

  Essa representação facilita a análise usando técnicas de álgebra linear e sistemas dinâmicos.

* **Multiplicador Dinâmico para o Sistema Vetorial**: O multiplicador dinâmico é generalizado para:

  $$\frac{\partial \boldsymbol{\xi}_t}{\partial w_{t-j}} = \mathbf{F}^j$$

  O efeito de um choque em $w_{t-j}$ sobre $y_t$ é dado pela primeira componente do vetor resultante.

* **Solução por Substituição Recursiva**: Similarmente ao caso de primeira ordem, podemos escrever a solução geral:

  $$\boldsymbol{\xi}_t = \mathbf{F}^t \boldsymbol{\xi}_0 + \sum_{j=0}^{t-1} \mathbf{F}^j \begin{bmatrix} w_{t-1-j} \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

  Isso expressa o estado atual em termos das condições iniciais e das entradas passadas.

* **Multiplicador Dinâmico como Elemento (1,1) de $\mathbf{F}^j$**: O efeito de $w_{t-j}$ sobre $y_t$ é dado pelo elemento na posição (1,1) da matriz $\mathbf{F}^j$, refletindo a influência direta sobre a variável de interesse.

* **Autovalores e Comportamento Dinâmico**: Os autovalores de $\mathbf{F}$ determinam o comportamento dinâmico do sistema. Se todos os autovalores tiverem módulo menor que 1, o sistema é estável.

* **Polinômio Característico**: Os autovalores são encontrados resolvendo o polinômio característico:

  $$\det(\mathbf{F} - \lambda \mathbf{I}) = 0$$

  Para a matriz $\mathbf{F}$ específica, o polinômio característico é:

  $$\lambda^p - \phi_1 \lambda^{p-1} - \phi_2 \lambda^{p-2} - \dots - \phi_p = 0$$

  As raízes desse polinômio (autovalores) fornecem insights sobre a dinâmica do sistema.

**Solução Geral de uma Equação de Diferenças de Ordem $p$ com Autovalores Distintos**

* **Decomposição de Jordan**: Quando $\mathbf{F}$ é diagonalizável, podemos escrever:

  $$\mathbf{F} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^{-1}$$

  Onde $\mathbf{\Lambda}$ é uma matriz diagonal contendo os autovalores $\lambda_i$, e $\mathbf{P}$ é a matriz de autovetores.

* **Multiplicador Dinâmico como Média Ponderada dos Autovalores**: Elevando $\mathbf{F}$ a uma potência $j$:

  $$\mathbf{F}^j = \mathbf{P} \mathbf{\Lambda}^j \mathbf{P}^{-1}$$

  O elemento (1,1) de $\mathbf{F}^j$ pode ser expresso como:

  $$[\mathbf{F}^j]_{11} = \sum_{i=1}^p c_i \lambda_i^j$$

  Onde os coeficientes $c_i$ dependem dos elementos da matriz $\mathbf{P}$ e $\mathbf{P}^{-1}$.

* **Expressão Fechada para os Coeficientes**: Os coeficientes $c_i$ são determinados pela condição inicial e podem ser calculados usando a teoria de equações de diferenças lineares homogêneas.

**O Multiplicador Dinâmico**

* **Multiplicador Dinâmico como Elemento (1,1) de $\mathbf{F}^j$**: Reforçando, o multiplicador dinâmico para a variável $y_t$ é obtido diretamente da elevação de $\mathbf{F}$ à potência $j$.

* **Caso de Autovalores Distintos**: Quando os autovalores $\lambda_i$ são distintos, a solução geral é uma combinação linear das potências dos autovalores.

* **Caso de Primeira Ordem**: No caso $p = 1$, temos um único autovalor $\lambda = \phi$, e a solução se reduz ao caso anteriormente discutido.

* **Sistemas Estáveis vs. Explosivos**: A estabilidade é determinada pelos módulos dos autovalores:

  - **Estável**: $|\lambda_i| < 1$ para todos $i$.
  - **Explosivo**: Existe pelo menos um $\lambda_i$ com $|\lambda_i| > 1$.

* **Autovalores Complexos e Oscilações Amortecidas**: Se os autovalores são complexos conjugados, a solução apresenta componentes oscilatórias:

  - Autovalores complexos: $\lambda = R e^{i\theta}$.
  - A solução envolve termos como $R^j e^{i\theta j}$, que correspondem a oscilações com frequência $\theta$ e amplitude decrescente se $R < 1$.

**Solução de uma Equação de Diferenças de Segunda Ordem com Autovalores Distintos**

* **Condições para Autovalores Reais vs. Complexos**: A equação característica de segunda ordem é:

  $$\lambda^2 - \phi_1 \lambda - \phi_2 = 0$$

  As raízes podem ser reais ou complexas dependendo do discriminante $D = \phi_1^2 + 4\phi_2$:

  - **Reais distintas**: $D > 0$.
  - **Reais repetidas**: $D = 0$.
  - **Complexas conjugadas**: $D < 0$.

* **Módulo e Frequência dos Autovalores Complexos**: Quando $D < 0$, os autovalores são:

  $$\lambda = \frac{\phi_1}{2} \pm i\frac{\sqrt{-D}}{2}$$

  O módulo é $R = \sqrt{\left( \frac{\phi_1}{2} \right)^2 + \left( \frac{\sqrt{-D}}{2} \right)^2}$, e a frequência é $\theta = \arctan\left( \frac{\sqrt{-D}}{\phi_1} \right)$.

* **Condições de Estabilidade**: Para estabilidade, o módulo dos autovalores deve ser menor que 1. Isso impõe restrições sobre $\phi_1$ e $\phi_2$.

**Solução Geral de uma Equação de Diferenças de Ordem $p$ com Autovalores Repetidos**

* **Decomposição de Jordan para Autovalores Repetidos**: Quando existem autovalores repetidos, a matriz $\mathbf{F}$ não é diagonalizável, mas pode ser colocada em forma canônica de Jordan, onde aparecem blocos superiores triangulares.

* **Multiplicador Dinâmico com Autovalores Repetidos**: A solução envolve termos multiplicados por potências de $j$:

  $$y_t = (\alpha_1 + \alpha_2 j) \lambda^j$$

  Isso reflete o crescimento polinomial devido à multiplicidade do autovalor.

**Cálculos de Longo Prazo e Valor Presente**

* **Solução de Longo Prazo em Sistemas Estáveis**: Em sistemas estáveis ($|\lambda_i| < 1$), as contribuições dos termos envolvendo $\lambda_i^j$ tendem a zero conforme $j \to \infty$, e o sistema converge para um estado estacionário.

* **Multiplicador Dinâmico Geral para o Sistema Vetorial**: A expressão geral do multiplicador dinâmico permite calcular o efeito de choques em qualquer período, considerando todas as interações entre as variáveis de estado.

* **Multiplicador do Valor Presente**: O multiplicador do valor presente em sistemas de ordem $p$ é dado por:

  $$\frac{\partial PV}{\partial w_t} = \sum_{j=0}^{\infty} \beta^j [\mathbf{F}^j]_{11}$$

  Isso integra os efeitos futuros descontados dos choques presentes.

* **Expressão Fechada para o Multiplicador do Valor Presente**: Usando técnicas de soma de séries geométricas generalizadas, podemos obter expressões fechadas para o multiplicador do valor presente.

* **Efeito Cumulativo de uma Mudança Única em $w$**: O efeito total de uma mudança única em $w_t$ é:

  $$\sum_{j=0}^{\infty} [\mathbf{F}^j]_{11}$$

  Essa soma converge se o sistema for estável, permitindo calcular o impacto acumulado.

* **Efeito de Longo Prazo de uma Mudança Permanente em $w$**: Uma mudança permanente em $w_t$ resulta em um novo estado estacionário, cujo valor pode ser calculado considerando a soma infinita das respostas aos choques persistentes.

**Anexo 1.A. Demonstrações das Proposições do Capítulo 1**

* **Demonstração da Proposição 1.1**: A proposição estabelece que os autovalores da matriz $\mathbf{F}$ são as raízes do polinômio característico associado. A demonstração envolve calcular $\det(\mathbf{F} - \lambda \mathbf{I})$ e mostrar que resulta no polinômio característico da equação de diferenças.

* **Demonstração da Proposição 1.2**: Esta proposição fornece uma expressão para os coeficientes $c_i$ na solução geral. A demonstração utiliza a teoria de equações lineares diferenciais com coeficientes constantes, adaptada para o contexto discreto.

* **Demonstração da Proposição 1.3**: A proposição relaciona o multiplicador do valor presente aos autovalores do sistema. A demonstração envolve manipular somas infinitas e utilizar propriedades dos autovalores para simplificar a expressão.