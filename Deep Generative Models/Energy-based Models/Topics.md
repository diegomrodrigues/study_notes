**Capítulo Título:** Como Treinar seus Modelos Baseados em Energia

---

**2. Modelos Baseados em Energia (EBMs)**

* **Parametrização da função de energia:**
  - Introdução detalhada aos EBMs, enfatizando que a função de energia \( E_\theta(x) \) pode ser parametrizada usando modelos flexíveis, como redes neurais profundas, convolucionais ou recorrentes, permitindo capturar estruturas complexas nos dados.
  - Discussão sobre como a escolha da parametrização afeta a capacidade do modelo de representar distribuições de probabilidade multimodais ou com estruturas hierárquicas.
  - Exploração de técnicas de regularização e arquitetura para evitar problemas como overfitting ou instabilidade durante o treinamento.

* **Densidade de probabilidade não normalizada:**
  - Definição formal da densidade de probabilidade não normalizada \( p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)} \), onde \( Z(\theta) = \int \exp(-E_\theta(x)) dx \) é a função de partição intratável.
  - Discussão sobre as implicações de trabalhar com distribuições não normalizadas, incluindo a dificuldade de calcular probabilidades absolutas, mas a facilidade relativa de calcular razões de probabilidades.
  - Exemplos de situações onde EBMs são vantajosos em relação a modelos probabilísticos tradicionais.

* **Intratabilidade do cálculo de verossimilhança e amostragem:**
  - Análise dos desafios computacionais associados ao cálculo da função de partição \( Z(\theta) \) e sua derivada em relação aos parâmetros do modelo.
  - Discussão sobre a impossibilidade prática de realizar a integração exata em espaços de alta dimensão, o que torna métodos exatos impraticáveis.
  - Introdução à necessidade de métodos aproximados, como amostragem Monte Carlo e aproximações variacionais, para tornar o treinamento viável.

* **Extensão para modelos condicionais e múltiplas variáveis:**
  - Explicação de como EBMs podem ser estendidos para modelar distribuições condicionais \( p(y|x) \) introduzindo dependências explícitas na função de energia.
  - Discussão sobre aplicações em aprendizado supervisionado e semi-supervisionado, onde EBMs podem ser utilizados para modelar a relação entre entradas e saídas.
  - Análise de casos onde múltiplas variáveis interagem de forma complexa, e como EBMs podem capturar essas interações através de funções de energia adequadamente parametrizadas.

---

**3. Treinamento por Máxima Verossimilhança com MCMC**

* **Estimação por Máxima Verossimilhança (MLE) como padrão:**
  - Reafirmação da MLE como método padrão para aprendizado de modelos probabilísticos, buscando encontrar parâmetros \( \theta \) que maximizem a probabilidade dos dados observados.
  - Discussão sobre a aplicação de MLE em EBMs e os desafios decorrentes da presença da função de partição intratável.
  - Análise de alternativas à MLE e quando elas podem ser apropriadas.

* **Constante de normalização intratável na MLE para EBMs:**
  - Detalhamento matemático de como a função de partição \( Z(\theta) \) aparece no cálculo da verossimilhança e suas derivadas.
  - Discussão sobre a dificuldade de diferenciar \( \log Z(\theta) \) em relação aos parâmetros do modelo.
  - Apresentação de exemplos ilustrativos que mostram o impacto da intratabilidade na otimização.

* **Decomposição do gradiente da log-verossimilhança:**
  - Apresentação da fórmula do gradiente da log-verossimilhança:
    \[
    \nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) + \mathbb{E}_{p_\theta(x')}[\nabla_\theta E_\theta(x')]
    \]
  - Explicação de que o primeiro termo é computável diretamente, enquanto o segundo termo envolve uma expectativa sobre a distribuição modelo, que é intratável.
  - Discussão sobre a necessidade de estimar essa expectativa para realizar a otimização.

* **Estimativa Monte Carlo do gradiente da log-verossimilhança:**
  - Introdução de métodos de amostragem, como Cadeias de Markov Monte Carlo (MCMC), para estimar a expectativa necessária.
  - Explicação de como gerar amostras \( x' \sim p_\theta(x') \) usando MCMC para aproximar o segundo termo do gradiente.
  - Análise dos trade-offs entre precisão da estimativa e custo computacional.

* **Langevin e Hamiltonian Monte Carlo:**
  - Descrição detalhada do algoritmo de Langevin Estocástico e como ele utiliza o gradiente da energia para realizar a amostragem.
  - Discussão sobre o Hamiltonian Monte Carlo (HMC) e suas vantagens em termos de eficiência de amostragem, especialmente em espaços de alta dimensão.
  - Comparação entre os métodos, incluindo suas vantagens e desvantagens em diferentes cenários.

* **Dinâmica de Langevin e sua discretização:**
  - Apresentação da equação diferencial estocástica que descreve a dinâmica de Langevin contínua.
  - Derivação da versão discretizada utilizada na prática:
    \[
    x_{t+1} = x_t - \frac{\epsilon}{2} \nabla_x E_\theta(x_t) + \sqrt{\epsilon} \eta_t
    \]
    onde \( \eta_t \sim \mathcal{N}(0, I) \).
  - Discussão sobre a escolha do passo \( \epsilon \) e seu impacto na qualidade da amostragem.

* **Algoritmo de Langevin Ajustado por Metropolis (MALA):**
  - Introdução do MALA como uma forma de corrigir os erros de discretização introduzidos pela dinâmica de Langevin.
  - Explicação de como o passo de aceitação/rejeição de Metropolis-Hastings é incorporado para garantir que a cadeia tenha a distribuição estacionária desejada.
  - Análise do compromisso entre eficiência computacional e precisão da amostragem.

* **Divergência Contrastiva (CD) e suas variantes:**
  - Apresentação do algoritmo de Divergência Contrastiva, introduzido por Hinton, como um método para aproximar o gradiente da log-verossimilhança utilizando um número limitado de passos de MCMC.
  - Discussão sobre as variantes do CD, como CD-k, onde k indica o número de passos de MCMC, e o CD persistente (PCD), que reutiliza amostras anteriores para melhorar a eficiência.
  - Exploração de métodos avançados, como CD de campo médio e CD multigrid, que visam acelerar a convergência e reduzir o viés.

* **Viés na MCMC truncada e métodos de correção de viés:**
  - Discussão sobre o fato de que o uso de cadeias de MCMC truncadas (com poucos passos) introduz viés na estimativa do gradiente.
  - Apresentação de técnicas para mitigar esse viés, como o uso de múltiplas cadeias, acoplamento de cadeias (coupled MCMC) e métodos para estimar e corrigir o viés introduzido.
  - Análise dos trade-offs entre viés, variância e custo computacional.

---

**4. Score Matching (SM)**

* **Equivalência de PDFs com derivadas primeiras iguais:**
  - Estabelecimento do conceito de que duas densidades de probabilidade que diferem apenas por uma constante possuem a mesma derivada logarítmica (score function).
  - Exploração de como o aprendizado da função score pode ser suficiente para caracterizar a distribuição dos dados até uma constante de normalização.
  - Discussão sobre as implicações desse fato para o treinamento de EBMs.

* **Divergência de Fisher como objetivo do SM:**
  - Introdução formal da divergência de Fisher como medida da discrepância entre as funções score do modelo e dos dados:
    \[
    D_F(p_{\text{dados}} \| p_\theta) = \frac{1}{2} \int p_{\text{dados}}(x) \| \nabla_x \log p_{\text{dados}}(x) - \nabla_x \log p_\theta(x) \|^2 dx
    \]
  - Discussão sobre como minimizar a divergência de Fisher permite ajustar o modelo para aproximar a função score dos dados.

* **Integração por partes para objetivo tratável:**
  - Utilização da integração por partes para reescrever a divergência de Fisher de forma que não dependa da derivada da densidade dos dados, que é desconhecida.
  - Derivação do objetivo de Score Matching que pode ser calculado diretamente a partir dos dados e do modelo:
    \[
    J_{\text{SM}}(\theta) = \frac{1}{2} \mathbb{E}_{p_{\text{dados}}} [ \text{tr}(\nabla_x^2 E_\theta(x)) + \frac{1}{2} \| \nabla_x E_\theta(x) \|^2 ]
    \]
  - Explicação de como este objetivo pode ser minimizado usando métodos de otimização padrão.

* **Custo computacional das segundas derivadas:**
  - Discussão sobre o fato de que o cálculo do termo envolvendo o traço do Hessiano \( \text{tr}(\nabla_x^2 E_\theta(x)) \) tem custo computacional quadrático em relação à dimensão dos dados.
  - Análise do impacto desse custo em aplicações de alta dimensão e a necessidade de métodos para reduzir o custo computacional.

* **Generalizações para dados discretos ou limitados:**
  - Exploração de extensões do Score Matching para lidar com dados que não satisfazem as condições de regularidade originais, como dados discretos ou limitados a um intervalo.
  - Discussão sobre métodos como o Score Matching generalizado, que adapta a formulação para diferentes tipos de dados.
  - Apresentação de aplicações práticas onde essas generalizações são necessárias.

---

**4.1 Denoising Score Matching (DSM)**

* **Abordando limitações do SM para dados não suaves:**
  - Identificação das limitações do Score Matching padrão ao lidar com distribuições de dados não suaves ou com suporte limitado.
  - Explicação de como essas limitações podem levar a estimativas inconsistentes ou instáveis.

* **Adição de ruído aos dados para suavização:**
  - Introdução da estratégia de adicionar ruído aos dados para criar uma distribuição suavizada onde o Score Matching pode ser aplicado efetivamente.
  - Discussão sobre a escolha do nível de ruído e sua influência na suavização da distribuição.

* **Aproximação do objetivo de Score Matching ruidoso:**
  - Derivação do objetivo do Denoising Score Matching, mostrando como ele evita o cálculo de segundas derivadas e depende apenas de gradientes de primeira ordem.
  - Apresentação da formulação:
    \[
    J_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_{\tilde{x}}} [ \| \nabla_x E_\theta(\tilde{x}) - \frac{\tilde{x} - x}{\sigma^2} \|^2 ]
    \]
    onde \( \tilde{x} = x + \epsilon \), com \( \epsilon \sim \mathcal{N}(0, \sigma^2 I) \).

* **Inconsistência do DSM e questões de variância:**
  - Discussão sobre a inconsistência do DSM, onde o modelo aprende a estimar a função score da distribuição ruidosa, não a original.
  - Análise de como a variância da estimativa aumenta à medida que o nível de ruído diminui, tornando a otimização mais difícil.

* **Redução de variância usando variáveis de controle:**
  - Introdução de técnicas de redução de variância, como variáveis de controle, para melhorar a estabilidade da otimização no DSM.
  - Explicação de como essas técnicas podem ser implementadas e seus efeitos na prática.

---

**4.2 Sliced Score Matching (SSM)**

* **Abordando a inconsistência do DSM:**
  - Apresentação do SSM como uma alternativa que busca resolver a inconsistência do DSM, fornecendo estimativas consistentes da função score dos dados originais.

* **Divergência de Fisher cortada (sliced):**
  - Definição da divergência de Fisher cortada, que considera projeções unidimensionais da função score em direções aleatórias:
    \[
    D_{\text{SSM}}(p_{\text{dados}} \| p_\theta) = \frac{1}{2} \mathbb{E}_{v \sim p(v)} \mathbb{E}_{x \sim p_{\text{dados}}} [ (v^\top (\nabla_x \log p_{\text{dados}}(x) - \nabla_x \log p_\theta(x)))^2 ]
    \]
  - Discussão sobre como essa abordagem reduz o custo computacional e evita o cálculo de segundas derivadas.

* **Eficiência computacional do SSM:**
  - Análise de como o SSM tem custo computacional linear na dimensão dos dados, tornando-o adequado para aplicações de alta dimensão.
  - Comparação com o custo quadrático do Score Matching original.

* **Avaliação em forma fechada para distribuições de projeção específicas:**
  - Discussão sobre como, para certas distribuições de projeção (como a distribuição normal), partes do objetivo do SSM podem ser calculadas analiticamente.
  - Exploração de como isso simplifica a implementação e melhora a eficiência.

* **Conexão com o estimador de traço de Skilling-Hutchinson:**
  - Estabelecimento da ligação entre o SSM e o estimador de traço de Skilling-Hutchinson, mostrando como técnicas de estimativa estocástica podem ser aplicadas para evitar cálculos custosos.

---

**4.3 Conexão com a Divergência Contrastiva**

* **Score Matching como caso limite da Divergência Contrastiva:**
  - Demonstração de que o Score Matching pode ser visto como um caso especial da Divergência Contrastiva quando o número de passos de MCMC tende a zero.
  - Discussão sobre as implicações teóricas dessa conexão e como ela pode orientar a escolha de métodos de treinamento.

* **Relação entre divergências de Fisher e KL:**
  - Exploração da relação matemática entre a divergência de Fisher (usada no Score Matching) e a derivada da divergência de Kullback-Leibler (subjacente à Divergência Contrastiva).
  - Análise de como essas divergências medem discrepâncias entre distribuições de diferentes maneiras e suas implicações práticas.

---

**4.4 Modelos Generativos Baseados em Score**

* **Treinamento de modelos de score em vez de funções de energia:**
  - Introdução ao conceito de modelar diretamente a função score dos dados, \( s_\theta(x) = \nabla_x \log p_\theta(x) \), em vez da função de energia.
  - Discussão sobre as vantagens dessa abordagem, incluindo simplificações computacionais e melhor aproveitamento de técnicas de deep learning.

* **Dinâmica de Langevin anelada e processos de difusão reversa:**
  - Explicação de como a dinâmica de Langevin anelada pode ser utilizada para gerar amostras a partir do modelo, iniciando de ruído puro e gradualmente refinando as amostras.
  - Introdução aos processos de difusão reversa, que interpretam o processo de geração como a reversão de um processo de difusão que corrompe os dados.

* **Perturbações de ruído em múltiplas escalas:**
  - Discussão sobre a utilização de diferentes níveis de ruído durante o treinamento para capturar estruturas em diferentes escalas nos dados.
  - Análise de como isso ajuda o modelo a aprender funções score mais robustas e a lidar com distribuições complexas.

* **Rede de Score Condicional ao Ruído:**
  - Apresentação da Noise-Conditional Score Network (NCSN), que estima a função score para diferentes níveis de ruído usando uma única rede neural.
  - Discussão sobre a arquitetura e como ela incorpora a informação do nível de ruído como entrada adicional.

---

**5. Noise Contrastive Estimation (NCE)**

* **Contrastando distribuições de dados e ruído:**
  - Introdução ao NCE como um método que transforma o problema de modelar a distribuição dos dados em um problema de classificação binária entre dados reais e amostras de ruído.
  - Discussão sobre como essa abordagem evita a necessidade de calcular a função de partição intratável.

* **Moldura de classificação binária para NCE:**
  - Explicação detalhada de como o NCE formula o treinamento como um problema de classificação, onde o modelo aprende a distinguir entre dados reais e amostras de uma distribuição de ruído conhecida.
  - Apresentação da função de perda logística utilizada e como ela está relacionada à maximização da verossimilhança.

* **Probabilidade a posteriori do ruído dado uma amostra:**
  - Derivação da probabilidade posterior de uma amostra ter vindo da distribuição de ruído, dado o modelo e as amostras observadas.
  - Discussão sobre como essa probabilidade é utilizada para atualizar os parâmetros do modelo.

* **Constante de normalização aprendível no NCE:**
  - Destacar que no NCE a constante de normalização é tratada como um parâmetro adicional a ser aprendido, o que simplifica a estimativa.
  - Análise das implicações dessa abordagem, incluindo possíveis problemas de escala e estratégias para lidar com eles.

* **Importância da seleção da distribuição de ruído:**
  - Enfatizar o papel crítico da escolha da distribuição de ruído no sucesso do NCE.
  - Discussão sobre critérios para selecionar uma distribuição de ruído apropriada, como similaridade com os dados reais e facilidade de amostragem.
  - Exploração de estratégias adaptativas onde a distribuição de ruído é ajustada durante o treinamento.

---

**5.1 Conexão com Score Matching**

* **Recuperando o Score Matching a partir do NCE:**
  - Demonstrar matematicamente como, sob certas condições e escolhas específicas da distribuição de ruído, o objetivo do NCE aproxima o do Score Matching.
  - Discussão sobre como isso fornece uma perspectiva unificadora entre diferentes métodos de treinamento de EBMs.

* **Expansão de Taylor do objetivo do NCE para pequenas perturbações:**
  - Apresentação de como uma expansão de Taylor do objetivo do NCE, quando o ruído é pequeno, leva a uma formulação semelhante ao Sliced Score Matching.
  - Análise das condições sob as quais essa aproximação é válida e suas implicações práticas.

---

**6. Outros Métodos**

* **Minimização de diferenças/derivadas de divergências KL:**
  - Exploração de métodos que minimizam a diferença ou a derivada temporal da divergência de Kullback-Leibler para evitar a função de partição intratável.
  - Discussão sobre como esses métodos se relacionam com princípios termodinâmicos e processos dinâmicos.

* **Aprendizado por Velocidade Mínima, Fluxo de Probabilidade Mínima e Contração KL Mínima:**
  - Apresentação desses métodos como estratégias para treinar EBMs através da minimização de mudanças na distribuição ao longo do tempo.
  - Discussão sobre suas conexões teóricas com o Score Matching e o NCE, e situações onde podem ser vantajosos.

* **Minimização da Discrepância de Stein:**
  - Introdução à Discrepância de Stein como uma medida de diferença entre distribuições que não depende da função de partição.
  - Discussão sobre como a minimização da Discrepância de Stein pode ser utilizada para treinar EBMs.

* **Discrepância de Stein Kernelizada e Estimador de Traço de Skilling-Hutchinson:**
  - Apresentação de técnicas para computar a Discrepância de Stein de forma eficiente em altas dimensões, usando kernelização e estimativas estocásticas.
  - Discussão sobre a escolha de kernels e suas implicações na qualidade do modelo.

* **Treinamento Adversarial para EBMs:**
  - Introdução ao uso de treinamento adversarial, inspirado em GANs, para evitar a necessidade de amostragem MCMC durante o treinamento.
  - Discussão sobre como um "gerador" adversário pode ser usado para aproximar a distribuição modelo.

* **Objetivo maximin para treinamento adversarial:**
  - Explicação da formulação do objetivo como um problema de otimização min-max, onde o modelo busca maximizar a divergência entre os dados reais e as amostras geradas pelo adversário.
  - Comparação com a formulação original dos GANs e adaptações necessárias para EBMs.

* **Escolhas de distribuição variacional para treinamento adversarial:**
  - Discussão sobre as limitações na escolha das distribuições variacionais utilizadas pelo adversário, incluindo questões de expressividade e eficiência computacional.
  - Exploração de diferentes arquiteturas e técnicas para melhorar a capacidade do adversário.

* **Amostradores neurais ruidosos:**
  - Apresentação de amostradores neurais que incorporam ruído para aproximar a distribuição alvo, facilitando a geração de amostras.
  - Discussão sobre os desafios na estimação da entropia desses amostradores e implicações para o treinamento.

* **Uso de diferentes divergências f:**
  - Exploração de como diferentes divergências f podem ser utilizadas no contexto do treinamento adversarial, além da divergência de KL.
  - Discussão sobre como a escolha da divergência afeta a dinâmica do treinamento e a qualidade das amostras geradas.

---

Esta expansão detalhada dos tópicos fornece uma base abrangente para um estudo aprofundado e avançado de Modelos Baseados em Energia. Cada seção explora os conceitos fundamentais, desafios, técnicas avançadas e conexões entre diferentes abordagens, oferecendo um guia completo para pesquisadores e estudantes interessados em dominar este campo.