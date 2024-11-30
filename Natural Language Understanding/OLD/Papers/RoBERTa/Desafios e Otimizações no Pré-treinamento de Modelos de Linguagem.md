## Desafios e Otimizações no Pré-treinamento de Modelos de Linguagem: Uma Análise Aprofundada do BERT e RoBERTa

<image: Um diagrama mostrando a arquitetura do BERT com camadas de atenção e feedforward, conectado a vários módulos representando diferentes tarefas de downstream, com setas indicando o fluxo de otimização e transferência de conhecimento>

### Introdução

O campo do processamento de linguagem natural (NLP) tem testemunhado avanços significativos com o advento de modelos de linguagem pré-treinados em larga escala. Entre esses modelos, o BERT (Bidirectional Encoder Representations from Transformers) emergiu como um marco, estabelecendo novos padrões de desempenho em uma variedade de tarefas de NLP. No entanto, a comparação e replicação desses modelos apresentam desafios substanciais devido à complexidade computacional, variabilidade nos dados de treinamento e sensibilidade a hiperparâmetros [1].

Este estudo aprofundado examina os desafios na comparação de modelos de linguagem pré-treinados, focando especificamente em uma replicação e otimização robusta do BERT. Através de uma análise sistemática e experimentos extensivos, investigamos como diferentes escolhas de design e estratégias de treinamento impactam o desempenho do modelo, levando ao desenvolvimento do RoBERTa (Robustly Optimized BERT Approach) [2].

> ⚠️ **Nota Importante**: A compreensão das nuances no pré-treinamento de modelos de linguagem é crucial para o avanço do campo de NLP e para a interpretação adequada das alegações de melhoria de desempenho em modelos recentes.

### Fundamentos Conceituais

| Conceito                                                     | Explicação                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **BERT (Bidirectional Encoder Representations from Transformers)** | Modelo de linguagem baseado em transformers que utiliza aprendizado não supervisionado em texto bidirecional para gerar representações contextuais profundas. É pré-treinado em duas tarefas principais: Masked Language Modeling (MLM) e Next Sentence Prediction (NSP) [3]. |
| **Masked Language Modeling (MLM)**                           | Técnica de treinamento onde tokens aleatórios no input são mascarados, e o modelo é treinado para prever esses tokens mascarados, permitindo aprendizado bidirecional [4]. |
| **Next Sentence Prediction (NSP)**                           | Tarefa de pré-treinamento onde o modelo aprende a prever se duas sentenças são consecutivas no texto original, visando capturar relações entre sentenças [5]. |
| **Fine-tuning**                                              | Processo de adaptar um modelo pré-treinado para uma tarefa específica, ajustando seus parâmetros com dados rotulados da tarefa alvo [6]. |

### 1. Desafios na Comparação de Modelos de Linguagem Pré-treinados

#### 1.1 Custo Computacional do Treinamento

O treinamento de modelos de linguagem em larga escala, como o BERT, requer recursos computacionais substanciais, o que apresenta um desafio significativo para a comunidade de pesquisa [7]. Esta limitação impacta diretamente a capacidade de replicar estudos existentes e explorar novas abordagens de forma abrangente.

> 💡 **Insight**: O alto custo computacional não apenas limita a acessibilidade da pesquisa, mas também pode levar a uma subutilização do potencial completo dos modelos devido a restrições práticas de tempo e recursos.

Para contextualizar, o treinamento do BERT original utilizou 16 TPU chips por 4 dias para o modelo BERT_BASE, e 64 TPU chips por 4 dias para o BERT_LARGE [8]. Esta escala de recursos está fora do alcance de muitos pesquisadores e instituições, criando uma barreira à entrada e à inovação no campo.

#### 1.2 Variabilidade nos Conjuntos de Dados

Um dos desafios mais críticos na comparação de modelos de linguagem pré-treinados é a variabilidade nos conjuntos de dados utilizados para o treinamento [9]. Muitos estudos empregam datasets privados ou proprietários, o que dificulta a reprodução direta dos resultados e a comparação justa entre diferentes abordagens.

> ❗ **Ponto de Atenção**: A falta de padronização nos conjuntos de dados de pré-treinamento pode levar a conclusões errôneas sobre a superioridade de certos modelos ou arquiteturas.

Para abordar esta questão, os autores do estudo sobre RoBERTa compilaram um novo dataset, CC-NEWS, comparável em tamanho a outros conjuntos de dados privados utilizados em estudos recentes [10]. Este esforço visa proporcionar um terreno mais nivelado para comparações, permitindo um controle mais rigoroso sobre os efeitos do tamanho do conjunto de dados no desempenho do modelo.

| Dataset                | Tamanho | Descrição                                            |
| ---------------------- | ------- | ---------------------------------------------------- |
| BOOKCORPUS + Wikipedia | 16GB    | Dataset original usado no BERT                       |
| CC-NEWS                | 76GB    | Novo dataset compilado para RoBERTa                  |
| OPENWEBTEXT            | 38GB    | Recriação open-source do corpus WebText              |
| STORIES                | 31GB    | Subset do CommonCrawl filtrado para estilo narrativo |

#### 1.3 Impacto dos Hiperparâmetros

As escolhas de hiperparâmetros têm um impacto substancial no desempenho final dos modelos de linguagem pré-treinados. No entanto, muitas publicações não fornecem detalhes suficientes sobre a otimização desses hiperparâmetros, dificultando a reprodução e comparação justa dos resultados [11].

> ✔️ **Destaque**: A otimização cuidadosa dos hiperparâmetros pode levar a melhorias significativas no desempenho, muitas vezes superando os ganhos atribuídos a inovações arquiteturais.

Os autores do estudo RoBERTa identificaram vários hiperparâmetros críticos que foram subexplorados no treinamento original do BERT:

1. **Tamanho do batch**: Aumentar o tamanho do batch de 256 para 8K sequências.
2. **Sequências mais longas**: Treinar com sequências de 512 tokens desde o início, sem o período inicial de sequências mais curtas.
3. **Mascaramento dinâmico**: Implementar um padrão de mascaramento que muda a cada época, em vez de um mascaramento estático.
4. **Remoção do NSP**: Eliminar a tarefa de Next Sentence Prediction, focando apenas no Masked Language Modeling.

A tabela a seguir ilustra o impacto dessas otimizações:

| Configuração        | SQuAD 2.0 (F1) | MNLI-m (Acc) | SST-2 (Acc) |
| ------------------- | -------------- | ------------ | ----------- |
| BERT original       | 76.3           | 84.3         | 92.8        |
| RoBERTa (otimizado) | 89.4           | 90.2         | 96.4        |

#### Perguntas Técnicas

1. Como a variabilidade nos conjuntos de dados de pré-treinamento pode afetar a generalização de um modelo de linguagem para tarefas downstream?

2. Quais são os trade-offs entre aumentar o tamanho do batch e o número de passos de treinamento em termos de eficiência computacional e desempenho do modelo?

3. Por que o mascaramento dinâmico pode ser mais eficaz que o mascaramento estático no pré-treinamento de modelos como o BERT?

### 2. Estudo de Replicação do BERT

<image: Um gráfico de linhas mostrando a evolução do desempenho do BERT original vs. RoBERTa em diferentes tarefas (eixo y) conforme aumenta o número de passos de treinamento (eixo x), destacando o ponto onde RoBERTa supera o BERT original>

O estudo de replicação do BERT foi conduzido com o objetivo de entender profundamente o impacto de várias decisões de design e hiperparâmetros no desempenho do modelo. Esta análise sistemática não apenas validou os resultados originais, mas também revelou oportunidades significativas para otimização [12].

#### 2.1 Análise Sistemática de Hiperparâmetros

A investigação detalhada do impacto dos hiperparâmetros no desempenho do BERT foi um dos aspectos centrais deste estudo. Os pesquisadores focaram em vários parâmetros cruciais:

1. **Tamanho do Batch**: 
   O estudo explorou o impacto de aumentar significativamente o tamanho do batch, passando de 256 sequências (usado no BERT original) para até 8K sequências [13].

2. **Taxa de Aprendizado**: 
   Foi realizada uma análise cuidadosa para determinar a taxa de aprendizado ótima para diferentes configurações de batch e número de passos de treinamento [14].

3. **Número de Épocas**: 
   O estudo investigou o efeito de treinar o modelo por períodos mais longos, aumentando substancialmente o número de passos de treinamento [15].

> 💡 **Insight**: O aumento do tamanho do batch, combinado com um ajuste adequado da taxa de aprendizado, permitiu uma otimização mais eficiente e um melhor desempenho global.

A tabela a seguir ilustra o impacto dessas mudanças nos hiperparâmetros:

| Tamanho do Batch | Passos | Taxa de Aprendizado | Perplexidade | MNLI-m (Acc) | SST-2 (Acc) |
| ---------------- | ------ | ------------------- | ------------ | ------------ | ----------- |
| 256              | 1M     | 1e-4                | 3.99         | 84.7         | 92.7        |
| 2K               | 125K   | 7e-4                | 3.68         | 85.2         | 92.9        |
| 8K               | 31K    | 1e-3                | 3.77         | 84.6         | 92.8        |

#### 2.2 Avaliação do Impacto do Tamanho dos Dados

Um aspecto crucial do estudo foi a exploração da relação entre o volume de dados de treinamento e o desempenho do modelo. Os pesquisadores não apenas replicaram o treinamento com o conjunto de dados original (BOOKCORPUS + Wikipedia, 16GB), mas também expandiram significativamente o corpus de treinamento [16].

> ⚠️ **Nota Importante**: O aumento do volume de dados de treinamento mostrou-se fundamental para melhorar o desempenho do modelo em tarefas downstream, desafiando a noção de que melhorias recentes eram principalmente devido a inovações arquiteturais.

Os pesquisadores compilaram um novo dataset, CC-NEWS, e combinaram com outros conjuntos de dados públicos, totalizando mais de 160GB de texto não comprimido [17]. Esta expansão permitiu uma análise mais robusta do impacto do tamanho dos dados no desempenho do modelo.

| Dataset Combinado        | Tamanho | SQuAD (v1.1/2.0) F1 | MNLI-m Acc | SST-2 Acc |
| ------------------------ | ------- | ------------------- | ---------- | --------- |
| BOOKS + WIKI (original)  | 16GB    | 93.6/87.3           | 89.0       | 95.3      |
| + dados adicionais       | 160GB   | 94.0/87.7           | 89.3       | 95.6      |
| + treinamento mais longo | 160GB   | 94.6/89.4           | 90.2       | 96.4      |

#### 2.3 Metodologia de Replicação

A metodologia de replicação foi meticulosamente projetada para garantir uma comparação justa e uma compreensão profunda dos fatores que influenciam o desempenho do BERT. Principais aspectos da metodologia incluem:

1. **Reimplementação do BERT**: 
   Os pesquisadores reimplementaram o BERT usando a biblioteca FAIRSEQ, garantindo flexibilidade para experimentação [18].

2. **Controle de Variáveis**: 
   Mantiveram a arquitetura do modelo constante (BERT_BASE: L=12, H=768, A=12, 110M parâmetros) enquanto variavam outros aspectos do treinamento [19].

3. **Avaliação Consistente**: 
   Utilizaram benchmarks padrão como GLUE, SQuAD e RACE para avaliar o desempenho em tarefas downstream, garantindo comparabilidade com resultados anteriores [20].

> ✔️ **Destaque**: A metodologia rigorosa permitiu isolar o impacto de diferentes escolhas de design e hiperparâmetros, fornecendo insights valiosos sobre o processo de pré-treinamento.

#### Perguntas Técnicas

1. Como o aumento do tamanho do batch afeta a convergência do modelo durante o pré-treinamento, e quais estratégias podem ser empregadas para mitigar possíveis problemas de estabilidade?

2. Qual é a relação entre o tamanho do conjunto de dados de pré-treinamento e a capacidade do modelo de generalizar para diferentes tarefas downstream? Existe um ponto de diminuição de retornos?

3. Como as escolhas de design no pré-treinamento (por exemplo, mascaramento dinâmico vs. estático) impactam a robustez do modelo em cenários de domínio cruzado?

### 3. Descobertas sobre o Subtreinamento do BERT

<image: Um diagrama de Venn comparando as capacidades do BERT original, RoBERTa, e outros modelos pós-BERT, destacando as áreas onde RoBERTa supera ou iguala os outros modelos>

As análises conduzidas no estudo de replicação do BERT revelaram uma descoberta surpreendente: o modelo BERT original estava significativamente subtreinado. Esta seção explora as evidências que suportam essa conclusão e suas implicações para o campo de NLP.

#### 3.1 Evidências de Subotimização

Os experimentos realizados forneceram evidências convincentes de que o BERT original não atingiu seu potencial máximo devido a limitações no processo de treinamento [21]. Algumas das principais observações incluem:

1. **Melhoria com Treinamento Prolongado**: 
   O desempenho do modelo continuou a melhorar significativamente além do número de passos de treinamento originalmente utilizado no BERT [22].

2. **Impacto do Tamanho do Batch**: 
   O uso de batches maiores, combinado com ajustes apropriados na taxa de aprendizado, levou a melhorias substanciais no desempenho [23].

3. **Efeito do Mascaramento Dinâmico**: 
   A implementação de um esquema de mascaramento dinâmico, em oposição ao mascaramento estático usado no BERT original, resultou em representações mais robustas [24].

> 💡 **Insight**: A descoberta do subtreinamento do BERT sugere que muitas das melhorias atribuídas a arquiteturas mais recentes podem, na verdade, ser alcançadas através de um processo de treinamento mais otimizado.

A tabela a seguir ilustra as melhorias obtidas com o RoBERTa em comparação com o BERT original:

| Modelo     | SQuAD v1.1 (F1) | SQuAD v2.0 (F1) | MNLI-m (Acc) | SST-2 (Acc) |
| ---------- | --------------- | --------------- | ------------ | ----------- |
| BERT_LARGE | 90.9            | 81.8            | 86.6         | 93.2        |
| RoBERTa    | 94.6            | 89.4            | 90.2         | 96.4        |

#### 3.2 Comparação com Modelos Posteriores

Uma das descobertas mais intrigantes do estudo foi que um BERT otimizado (RoBERTa) podia igualar ou superar o desempenho de modelos mais recentes que alegavam superioridade arquitetural [25].

> ⚠️ **Nota Importante**: Estas descobertas questionam a necessidade de arquiteturas cada vez mais complexas e sugerem que há ainda muito a ser explorado em termos de otimização de treinamento de modelos existentes.

Comparação de RoBERTa com outros modelos estado-da-arte:

| Modelo      | GLUE Score | SQuAD v2.0 (F1) | RACE Accuracy |
| ----------- | ---------- | --------------- | ------------- |
| BERT_LARGE  | 80.5       | 81.8            | 72.0          |
| XLNet_LARGE | 88.4       | 89.1            | 81.7          |
| RoBERTa     | 88.5       | 89.8            | 83.2          |

Estas comparações demonstram que o RoBERTa, essencialmente um BERT otimizado, consegue competir e até superar modelos mais recentes e complexos em várias tarefas benchmark [26].

#### Perguntas Técnicas

1. Considerando as descobertas sobre o subtreinamento do BERT, como podemos desenvolver melhores práticas para determinar quando um modelo de linguagem pré-treinado atingiu seu potencial máximo?

2. Que implicações o subtreinamento do BERT tem para a interpretação de estudos comparativos entre diferentes arquiteturas de modelos de linguagem?

3. Como as técnicas de otimização descobertas no estudo do RoBERTa podem ser aplicadas ou adaptadas para outros tipos de modelos de deep learning fora do domínio de NLP?

### 4. Otimização do Processo de Pré-treinamento

<image: Um fluxograma detalhando as etapas do processo de pré-treinamento do RoBERTa, destacando as modificações em relação ao BERT original, incluindo mascaramento dinâmico, remoção do NSP, e ajustes de hiperparâmetros>

O desenvolvimento do RoBERTa (Robustly Optimized BERT Approach) foi baseado em uma série de otimizações cuidadosamente selecionadas e testadas. Estas melhorias não apenas aumentaram o desempenho do modelo, mas também proporcionaram insights valiosos sobre o processo de pré-treinamento de modelos de linguagem em larga escala [27].

#### 4.1 Ajuste Fino de Hiperparâmetros

O ajuste fino dos hiperparâmetros foi um componente crucial na otimização do RoBERTa. Os pesquisadores exploraram extensivamente o espaço de hiperparâmetros, focando em aspectos que tinham sido subexplorados no treinamento original do BERT [28].

> ✔️ **Destaque**: A otimização cuidadosa dos hiperparâmetros provou ser tão impactante quanto inovações arquiteturais, ressaltando a importância de um processo de treinamento bem ajustado.

Principais modificações nos hiperparâmetros:

1. **Tamanho do Batch**: 
   Aumentado de 256 sequências no BERT original para 8K sequências no RoBERTa [29].

2. **Taxa de Aprendizado**: 
   Ajustada para acomodar o aumento no tamanho do batch, com um peak learning rate de 4e-4 para RoBERTaLARGE e 6e-4 para RoBERTaBASE [30].

3. **Warm-up Steps**: 
   Otimizados para 30k passos no RoBERTaLARGE e 24k no RoBERTaBASE [31].

4. **Adam Optimizer**: 
   Ajustes nos parâmetros β e ε para melhorar a estabilidade do treinamento com batches grandes [32].

A tabela a seguir resume os principais hiperparâmetros para o RoBERTaLARGE:

| Hiperparâmetro        | Valor     |
| --------------------- | --------- |
| Número de Camadas     | 24        |
| Hidden Size           | 1024      |
| FFN Inner Hidden Size | 4096      |
| Attention Heads       | 16        |
| Batch Size            | 8k        |
| Learning Rate         | 4e-4      |
| Adam ε                | 1e-6      |
| Adam β1, β2           | 0.9, 0.98 |

#### 4.2 Extensão do Tempo de Treinamento

Uma das descobertas mais significativas foi o impacto positivo de estender substancialmente o tempo de treinamento. O RoBERTa foi treinado por muito mais passos do que o BERT original, permitindo que o modelo extraísse mais conhecimento dos dados de treinamento [33].

> 💡 **Insight**: Treinar por mais tempo não apenas melhorou o desempenho, mas também revelou que muitos modelos anteriores estavam subtreinados, não atingindo seu potencial máximo.

Comparação do número de passos de treinamento:

| Modelo              | Passos de Treinamento |
| ------------------- | --------------------- |
| BERT original       | 1M                    |
| RoBERTa (inicial)   | 100K                  |
| RoBERTa (estendido) | 300K                  |
| RoBERTa (final)     | 500K                  |

Os resultados mostraram melhorias consistentes com o aumento do tempo de treinamento:

| Passos | SQuAD (v1.1/2.0) F1 | MNLI-m Acc | SST-2 Acc |
| ------ | ------------------- | ---------- | --------- |
| 100K   | 94.0/87.7           | 89.3       | 95.6      |
| 300K   | 94.4/88.7           | 90.0       | 96.1      |
| 500K   | 94.6/89.4           | 90.2       | 96.4      |

#### 4.3 Ampliação do Conjunto de Dados

A expansão significativa do conjunto de dados de treinamento foi outro fator chave na otimização do RoBERTa. Os pesquisadores não apenas utilizaram o dataset original do BERT (BOOKCORPUS + Wikipedia), mas também incorporaram conjuntos de dados adicionais [34].

> ❗ **Ponto de Atenção**: A qualidade e diversidade dos dados de treinamento são tão importantes quanto a quantidade, influenciando diretamente a capacidade de generalização do modelo.

Conjuntos de dados utilizados no treinamento do RoBERTa:

1. BOOKCORPUS + English Wikipedia (16GB)
2. CC-NEWS (76GB)
3. OPENWEBTEXT (38GB)
4. STORIES (31GB)

Total combinado: mais de 160GB de texto não comprimido [35].

O impacto da ampliação do conjunto de dados foi significativo:

| Dataset             | SQuAD (v1.1/2.0) F1 | MNLI-m Acc | SST-2 Acc |
| ------------------- | ------------------- | ---------- | --------- |
| BOOKS + WIKI (16GB) | 93.6/87.3           | 89.0       | 95.3      |
| Todos (160GB)       | 94.6/89.4           | 90.2       | 96.4      |

#### 4.4 Inovações no Processo de Treinamento

Além dos ajustes nos hiperparâmetros e na quantidade de dados, o RoBERTa introduziu algumas inovações cruciais no processo de treinamento [36]:

1. **Mascaramento Dinâmico**: 
   Em vez de usar um padrão de mascaramento estático gerado uma vez durante o pré-processamento, o RoBERTa implementou um mascaramento dinâmico que muda o padrão a cada época de treinamento [37].

2. **Remoção do Next Sentence Prediction (NSP)**: 
   Os experimentos mostraram que a tarefa de NSP não contribuía significativamente para o desempenho do modelo, levando à sua remoção no RoBERTa [38].

3. **Treinamento com Sequências Mais Longas**: 
   O RoBERTa foi treinado exclusivamente com sequências de comprimento total (512 tokens) desde o início, em contraste com o BERT que usava sequências mais curtas no início do treinamento [39].

4. **Full-Sentences Without NSP**: 
   O input foi modificado para incluir segmentos de texto contíguos de um ou mais documentos, sem a limitação de pares de sentenças imposta pelo NSP [40].

> ✔️ **Destaque**: Estas modificações no processo de treinamento demonstraram que mesmo pequenas mudanças na formulação das tarefas de pré-treinamento podem ter impactos significativos no desempenho final do modelo.

#### Perguntas Técnicas

1. Como o mascaramento dinâmico afeta a convergência do modelo durante o pré-treinamento em comparação com o mascaramento estático? Existem trade-offs a serem considerados?

2. Considerando a remoção bem-sucedida do NSP no RoBERTa, quais são as implicações para o design de tarefas de pré-treinamento em futuros modelos de linguagem?

3. Como o aumento do tamanho do batch e do número de passos de treinamento interage com a escolha da taxa de aprendizado e outros hiperparâmetros do otimizador? Quais são as considerações práticas ao escalar esses parâmetros?

### Conclusão

O estudo detalhado que levou ao desenvolvimento do RoBERTa fornece insights valiosos sobre o processo de pré-treinamento de modelos de linguagem em larga escala. As principais conclusões incluem:

1. A importância crítica da otimização de hiperparâmetros e do processo de treinamento, que pode levar a melhorias substanciais sem alterações arquiteturais [41].

2. O impacto significativo do volume e da diversidade dos dados de treinamento no desempenho final do modelo [42].

3. A eficácia de técnicas como mascaramento dinâmico e treinamento com sequências mais longas na melhoria da capacidade de generalização do modelo [43].

4. A descoberta de que modelos anteriores, como o BERT original, estavam subtreinados, sugerindo que há ainda muito potencial a ser explorado em arquiteturas existentes [44].

Essas descobertas não apenas resultaram em um modelo (RoBERTa) que estabeleceu novos estados da arte em várias tarefas de NLP, mas também forneceram diretrizes valiosas para futuros desenvolvimentos na área de modelos de linguagem pré-treinados [45].

### Perguntas Avançadas

1. Considerando as descobertas do RoBERTa sobre a importância do processo de treinamento, como podemos desenvolver métodos mais sistemáticos para explorar o espaço de hiperparâmetros em modelos de linguagem de larga escala, considerando as limitações computacionais?

2. Como as lições aprendidas com o RoBERTa podem ser aplicadas ao desenvolvimento de modelos multimodais que integram texto com outras modalidades como imagem ou áudio?

3. Dado o sucesso do RoBERTa em melhorar o BERT sem alterações arquiteturais significativas, quais são as implicações para o debate entre escala do modelo vs. inovação arquitetural no avanço do estado da arte em NLP?

4. Como podemos equilibrar o trade-off entre o aumento do tamanho dos dados de treinamento e a manutenção da qualidade e relevância desses dados para tarefas específicas de downstream?

5. Considerando o alto custo computacional do pré-treinamento de modelos como o RoBERTa, quais estratégias podem ser desenvolvidas para democratizar o acesso a este tipo de pesquisa e desenvolvimento na comunidade de NLP?

### Referências

[1] "Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[2] "We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[3] "BERT takes as input a concatenation of two segments (sequences of tokens), x₁,...,xₙ and y₁,...,yₘ. Segments usually consist of more than one natural sentence." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[4] "A random sample of the tokens in the input sequence is selected and replaced with the special token [MASK]. The MLM objective is a cross-entropy loss on predicting the masked tokens." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[5] "NSP is a binary classification loss for predicting whether two segments follow each other in the original text." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[6] "Following previous work, we evaluate our pretrained models on downstream tasks using the following three benchmarks." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[7] "Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[8] "Devlin et al. (2019) originally trained BERT BASE for 1M steps with a batch size of 256 sequences." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[9] "BERT-style pretraining crucially relies on large quantities of text. Baevski et al. (2019) demonstrate that increasing data size can result in improved end-task performance." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[10] "We also collect a large new dataset (CC-NEWS) of comparable size to other privately used datasets, to better control for training set size effects." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[11] "We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[12] "We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[13] "In Table 3 we compare perplexity and end-task performance of BERTBASE as we increase the batch size, controlling for the number of passes through the training data." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[14] "We tune the learning rate (lr) for each setting." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[15] "Finally, we pretrain RoBERTa for significantly longer, increasing the number of pretraining steps from 100K to 300K, and then further to 500K." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[16] "We consider five English-language corpora of varying sizes and domains, totaling over 160GB of uncompressed text." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[17] "CC-NEWS, which we collected from the English portion of the CommonCrawl News dataset (Nagel, 2016). The data contains 63 million English news articles crawled between September 2016 and February 2019." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[18] "We reimplement BERT in FAIRSEQ (Ott et al., 2019)." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[19] "Specifically, we begin by training BERT models with the same configuration as BERT_BASE (L = 12, H = 768, A = 12, 110M params)." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[20] "Following previous work, we evaluate our pretrained models on downstream tasks using the following three benchmarks." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[21] "We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it." (Excerpt from RoBERTa: A Robustly Optimized BERT Pretraining Approach)

[22] "Finally, we pretrain RoBERTa for significantly longer, increasing the number of pre