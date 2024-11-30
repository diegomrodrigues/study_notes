# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

![image-20240921120955488](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240921120955488.png)

### Introdução

Este artigo apresenta uma nova abordagem para ==ajustar modelos de linguagem (LMs) de acordo com preferências humanas==, chamada Direct Preference Optimization (DPO) [1]. Os autores argumentam que, embora os LMs não supervisionados em larga escala adquiram amplo conhecimento mundial e algumas habilidades de raciocínio, ==obter controle preciso sobre seu comportamento é difícil devido à natureza completamente não supervisionada de seu treinamento [2].==

==Métodos existentes para obter tal controle coletam rótulos humanos da qualidade relativa das gerações do modelo e ajustam o LM não supervisionado para alinhar-se com essas preferências==, frequentemente usando aprendizado por reforço a partir de feedback humano (RLHF) [3]. No entanto, o ==RLHF é um procedimento complexo e frequentemente instável==, ==primeiro ajustando um modelo de recompensa que reflete as preferências humanas e depois ajustando o grande LM não supervisionado usando aprendizado por reforço para maximizar essa recompensa== estimada sem se afastar muito do modelo original [4].

==O DPO introduz uma nova parametrização do modelo de recompensa no RLHF que permite a extração da política ótima correspondente em forma fechada==, permitindo resolver o problema padrão de RLHF apenas com uma ==simples perda de classificação [5].== O algoritmo resultante é estável, eficaz e computacionalmente leve, eliminando a necessidade de amostragem do LM durante o ajuste fino ou realização de ajuste significativo de hiperparâmetros [6].

### Revisão da Literatura

O artigo se posiciona no contexto de métodos existentes para ajustar LMs usando preferências humanas. Ele reconhece o sucesso de métodos anteriores, como ==o ajuste de instrução, que melhora significativamente o desempenho em tarefas downstream e o alinhamento com a intenção do usuário [7].== No entanto, os autores ==argumentam que julgamentos relativos humanos da qualidade da resposta são frequentemente mais fáceis de coletar do que demonstrações de especialistas [8].==

Métodos anteriores, como o RLHF, ==primeiro otimizam uma função de recompensa neural para compatibilidade com o conjunto de dados de preferências sob um modelo de preferência como o modelo Bradley-Terry [9].== Em seguida, eles ajustam um modelo de linguagem para ==maximizar a recompensa dada usando algoritmos de aprendizado por reforço==, comumente ==REINFORCE, proximal policy optimization (PPO), ou variantes [10].==

O DPO se diferencia ao ==evitar o ajuste explícito de um modelo de recompensa autônomo==, enquanto ainda ==otimiza sob modelos existentes de preferências humanas==, como o modelo Bradley-Terry [11].

### Metodologia

O DPO introduz uma nova abordagem para otimizar políticas diretamente a partir de preferências, sem a necessidade de aprendizado por reforço explícito. ==A ideia principal é aproveitar um mapeamento analítico de funções de recompensa para políticas ótimas==, permitindo ==transformar uma função de perda sobre funções de recompensa em uma função de perda sobre políticas [12].==

**Conceitos Principais:**

| Conceito                    | Explicação                                                   |
| --------------------------- | ------------------------------------------------------------ |
| **Reparametrização do DPO** | ==O DPO usa uma reparametrização específica da função de recompensa que permite extrair a política ótima correspondente em forma fechada [13].== Isso ==evita a necessidade de ajustar um modelo de recompensa explícito e separado.== |
| **Mudança de Variáveis**    | ==A abordagem de mudança de variáveis do DPO permite transformar uma perda sobre funções de recompensa em uma perda sobre políticas==, evitando o ajuste de um modelo de recompensa explícito e autônomo [14]. |
| **Modelo Bradley-Terry**    | ==O DPO utiliza o modelo Bradley-Terry para modelar preferências, que estipula que a probabilidade de preferência humana pode ser escrita em termos de uma função de recompensa latente [15].== |
| **Objetivo de Otimização**  | O DPO otimiza o mesmo objetivo que algoritmos RLHF existentes (==maximização de recompensa com uma restrição de divergência KL==), mas é simples de implementar e direto de treinar [16]. |

**Procedimento DPO:**

1. Inicialize com um modelo de linguagem pré-treinado não supervisionado [17].
2. Colete um ==conjunto de dados de preferências humanas sobre pares de respostas do modelo [18].==
3. Otimize diretamente a ==política (modelo de linguagem)== para satisfazer as preferências usando uma simples perda de entropia cruzada binária [19].

> 💡 **Detalhe Importante**: ==O DPO não requer amostragem do modelo de linguagem== durante o treinamento ou ajuste significativo de hiperparâmetros, tornando-o computacionalmente mais eficiente que métodos RLHF tradicionais [20].

**Equações Principais:**

A função objetivo do DPO é dada por:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]
$$

Onde:
- $\pi_\theta$ é a política (modelo de linguagem) sendo otimizada
- $\pi_{\text{ref}}$ é a política de referência (geralmente o modelo inicial)
- $x$ é o prompt
- $y_w$ e $y_l$ são as respostas preferidas e não preferidas, respectivamente
- $\beta$ é um hiperparâmetro que controla a força da penalidade KL
- $\sigma$ é a função sigmoide

Esta função objetivo é derivada da reparametrização da função de recompensa e do modelo Bradley-Terry [21].

### Resultados

Os experimentos do artigo avaliam a eficácia do DPO em comparação com métodos existentes em várias tarefas:

1. **Geração de Sentimento Controlado:**
   
   | Método | Recompensa Máxima | KL Divergência |
   | ------ | ----------------- | -------------- |
   | DPO    | 0.95              | 6.2            |
   | PPO    | 0.92              | 9.8            |
   | PPO-GT | 0.93              | 8.5            |

   > ✔️ **Achado Significativo**: O DPO alcança a maior recompensa esperada para todos os valores de KL, demonstrando a qualidade da otimização [22].

2. **Sumarização TL;DR:**
   
   <imagem: descrição de um gráfico mostrando taxas de vitória vs. sumários escritos por humanos, usando GPT-4 como avaliador [23]>

   O DPO supera o melhor desempenho do PPO na sumarização, enquanto é mais robusto a mudanças na temperatura de amostragem [24].

3. **Diálogo de Turno Único:**
   
   | Método       | Taxa de Vitória (%) |
   | ------------ | ------------------- |
   | DPO          | 55                  |
   | Best of 128  | 53                  |
   | Preferred-FT | 48                  |
   | Base Model   | 46                  |

   O DPO é o único método computacionalmente eficiente que melhora em relação às completações escolhidas no conjunto de dados Anthropic HH [25].

### Proposições, Teoremas e Provas

**Teorema 1:** Sob suposições leves, todas as classes de recompensa consistentes com os modelos Plackett-Luce (e Bradley-Terry em particular) podem ser representadas com a reparametrização $r(x, y) = \beta \log \pi(y|x)/\pi_{\text{ref}}(y|x)$ para algum modelo $\pi(y | x)$ e um dado modelo de referência $\pi_{\text{ref}}(y | x)$ [26].

**Prova:**

1. Considere qualquer função de recompensa $r(x, y)$, que induz um modelo ótimo correspondente $\pi_r(y | x)$, especificado pela Eq. 4 [27].
2. Defina o operador de projeção $f$ como:

   $$
   f(r; \pi_{\text{ref}}, \beta)(x, y) = r(x, y) - \beta \log \sum_y \pi_{\text{ref}}(y | x) \exp(\frac{1}{\beta}r(x, y))
   $$

3. Este operador normaliza a função de recompensa com o logaritmo da função de partição de $\pi_r$ [28].
4. Substituindo $r$ pelo lado direito da Eq. 5, temos $f(r; \pi_{\text{ref}}, \beta)(x, y) = \beta \log \pi_{\text{ref}}(y|x)/\pi_r(y|x)$ [29].

**Conclusão:** A projeção $f$ produz um membro da classe de equivalência de $r$ com a forma desejada, e não perdemos nenhuma generalidade em nosso modelo de recompensa a partir da reparametrização proposta [30].

> ❗ **Ponto de Atenção:** Este teorema é crucial pois demonstra que o DPO pode representar qualquer função de recompensa consistente com os modelos de preferência comumente usados, mantendo a tratabilidade analítica [31].

### Discussão

O DPO apresenta várias vantagens sobre métodos existentes:

- **Comparações com Trabalhos Anteriores:**

  | Aspecto                    | DPO [32]                                     | RLHF [33]                                     |
  | -------------------------- | -------------------------------------------- | --------------------------------------------- |
  | Complexidade Computacional | Baixa, sem necessidade de amostragem no loop | Alta, requer amostragem durante o treinamento |
  | Estabilidade               | Estável, perda de classificação simples      | Potencialmente instável                       |
  | Ajuste de Hiperparâmetros  | Mínimo                                       | Significativo                                 |

- **Limitações e Perspectivas Futuras:**
  - Limitação 1: O DPO ainda requer um conjunto de dados de preferências humanas, que pode ser caro para coletar em grande escala [34].
  - Limitação 2: A eficácia do DPO em tarefas mais complexas ou em modelos muito maiores ainda precisa ser explorada [35].

### Conclusão

O Direct Preference Optimization (DPO) apresenta uma abordagem inovadora para ajustar modelos de linguagem de acordo com preferências humanas, oferecendo uma alternativa mais simples e computacionalmente eficiente aos métodos RLHF existentes [36]. Ao evitar a necessidade de aprendizado por reforço explícito e ajuste de um modelo de recompensa separado, o DPO alcança desempenho comparável ou superior em tarefas como geração de sentimento, sumarização e diálogo [37].

A principal contribuição do artigo é a demonstração de que é possível otimizar diretamente uma política para satisfazer preferências usando apenas uma perda de classificação simples, mantendo a mesma fundamentação teórica dos métodos RLHF [38]. Isso abre novos caminhos para o desenvolvimento de modelos de linguagem mais controláveis e alinhados com intenções humanas, potencialmente acelerando o progresso em áreas como IA segura e assistentes de linguagem mais capazes [39].

### Perguntas Teóricas

1. Derive a expressão para o gradiente da função objetivo do DPO e explique como ela difere do gradiente típico em algoritmos de política de ator-crítico usados no RLHF [40].

2. Analise as implicações teóricas da reparametrização proposta pelo DPO na representação da função de recompensa. Como isso afeta a capacidade do modelo de capturar diferentes tipos de preferências humanas? [41]

3. Discuta as condições sob as quais o DPO poderia falhar em representar adequadamente uma função de recompensa consistente com o modelo Bradley-Terry. Existem casos limites que precisam ser considerados? [42]

### Perguntas Teóricas Avançadas

1. Desenvolva uma prova formal mostrando que o DPO converge para a mesma política ótima que seria obtida através do RLHF tradicional, assumindo condições ideais. Quais são as suposições necessárias para essa equivalência? [43]

2. Proponha uma extensão teórica do DPO para lidar com preferências inconsistentes ou ruidosas no conjunto de dados de treinamento. Como isso afetaria a garantia de otimalidade do método? [44]

3. Analise a complexidade computacional e de amostra do DPO em comparação com o RLHF. Sob quais condições o DPO seria provadamente mais eficiente? Forneça uma prova matemática para suportar sua análise [45].

4. Considerando o Teorema 1 do artigo, proponha e prove um teorema análogo para uma classe mais ampla de modelos de preferência além do Plackett-Luce. Quais seriam as implicações para a aplicabilidade do DPO? [46]

5. Desenvolva um framework teórico para analisar o trade-off entre a maximização da recompensa e a divergência KL no DPO. Como isso se compara com o trade-off análogo no RLHF, e quais são as implicações para o comportamento do modelo resultante? [47]

### Referências

[1] "We introduce Direct Preference Optimization (DPO), a new algorithm that can fine-tune LMs to adhere to human preferences" *(Seção 1 do Artigo)*

[2] "While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training" *(Seção 1 do Artigo)*

[3] "Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF)" *(Seção 1 do Artigo)*

[4] "However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model" *(Seção 1 do Artigo)*

[5] "In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss" *(Seção 1