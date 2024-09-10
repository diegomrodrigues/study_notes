## Prompts Preenchidos: Prompts Instanciados Criados a partir de Templates

<image: Um diagrama mostrando um template de prompt com slots vazios à esquerda, setas apontando para um prompt preenchido à direita com os slots preenchidos com texto específico>

### Introdução

Prompts preenchidos, também conhecidos como prompts instanciados, são uma técnica fundamental no campo de instrução e ajuste fino de modelos de linguagem grandes (LLMs). Eles são criados a partir de templates de prompts, que são estruturas predefinidas contendo slots vazios, que são então preenchidos com informações específicas para criar prompts completos e contextualizados [1]. Esta abordagem permite a geração eficiente de grandes volumes de dados de treinamento para tarefas de instrução, aproveitando datasets existentes e diretrizes de anotação [2].

### Conceitos Fundamentais

| Conceito               | Explicação                                                   |
| ---------------------- | ------------------------------------------------------------ |
| **Template de Prompt** | Uma estrutura predefinida contendo slots vazios para serem preenchidos com informações específicas. Serve como base para criar prompts instanciados [1]. |
| **Prompt Preenchido**  | Um prompt completo criado ao preencher os slots de um template com informações específicas, como texto de entrada, perguntas ou instruções [1]. |
| **Instrução**          | Uma descrição em linguagem natural de uma tarefa a ser realizada, combinada com demonstrações rotuladas da tarefa [3]. |

> ⚠️ **Nota Importante**: Os prompts preenchidos são cruciais para o ajuste fino de instruções (instruction tuning) em LLMs, permitindo que eles aprendam a seguir instruções e realizar tarefas diversas [3].

### Processo de Criação de Prompts Preenchidos

<image: Um fluxograma mostrando as etapas de criação de prompts preenchidos: 1) Seleção do template, 2) Extração de dados do dataset, 3) Preenchimento dos slots, 4) Geração do prompt final>

O processo de criação de prompts preenchidos envolve várias etapas:

1. **Seleção do Template**: Escolha um template apropriado para a tarefa em questão. Por exemplo, para análise de sentimento:
   ```
   {{text}} Qual é o sentimento expresso nesta revisão?
   ```
   [4]

2. **Extração de Dados**: Extraia os componentes relevantes (como texto, contexto, hipótese) dos datasets de treinamento existentes [5].

3. **Preenchimento dos Slots**: Insira os dados extraídos nos slots correspondentes do template [5].

4. **Geração do Prompt Final**: Combine o template preenchido com quaisquer instruções adicionais ou demonstrações para criar o prompt final [5].

> ✔️ **Destaque**: A diversidade nos prompts é crucial. Modelos de linguagem podem ser usados para gerar paráfrases dos prompts, aumentando a variabilidade [5].

### Aplicações e Exemplos

Os prompts preenchidos são amplamente utilizados em várias tarefas de processamento de linguagem natural:

#### Análise de Sentimento

```
{{texto}} Como o revisor se sente sobre o filme?
```

Exemplo preenchido:
```
Não gostei do serviço que me foi prestado quando entrei no hotel. Também não gostei da área em que o hotel estava localizado. Muito barulho e eventos acontecendo para eu me sentir relaxado. Em resumo, nossa estadia foi
```
[6]

#### Resposta a Perguntas Extrativas

```
{{contexto}} A partir da passagem, {{pergunta}}
```

Exemplo preenchido:
```
Beyoncé Giselle Knowles-Carter (nascida em 4 de setembro de 1981) é uma cantora, compositora, produtora musical e atriz americana. Nascida e criada em Houston, Texas, ela se apresentou em várias competições de canto e dança quando criança, e alcançou fama no final dos anos 1990 como vocalista principal do grupo feminino de R&B Destiny's Child. A partir da passagem, quando Beyoncé começou a se tornar popular?
```
[7]

#### Perguntas Técnicas/Teóricas

1. Como a estrutura de um prompt preenchido pode influenciar o desempenho de um LLM em uma tarefa específica?
2. Descreva um cenário em que o uso de prompts preenchidos seria particularmente benéfico para o ajuste fino de um LLM.

### Vantagens e Desvantagens dos Prompts Preenchidos

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Permitem a criação eficiente de grandes volumes de dados de treinamento [8] | Podem introduzir viés se os templates não forem cuidadosamente projetados [9] |
| Facilitam o ajuste fino de LLMs para tarefas específicas [3] | A qualidade dos prompts depende fortemente da qualidade dos templates e dos dados de entrada [9] |
| Aumentam a consistência nos formatos de entrada para LLMs [8] | Podem limitar a criatividade do modelo se os prompts forem muito restritivos [9] |

### Técnicas Avançadas de Preenchimento de Prompts

<image: Um diagrama mostrando diferentes técnicas de preenchimento de prompts, incluindo geração automática, paráfrase e seleção baseada em similaridade>

1. **Geração Automática**: Utilização de LLMs para gerar automaticamente variações de prompts a partir de templates básicos [10].

2. **Paráfrase**: Criação de múltiplas versões de um prompt através de técnicas de paráfrase, aumentando a diversidade do conjunto de treinamento [5].

3. **Seleção Baseada em Similaridade**: Escolha dinâmica de demonstrações para incluir no prompt com base na similaridade com o exemplo atual [11].

A formulação matemática para a seleção baseada em similaridade pode ser expressa como:

$$
sim(x, y) = \frac{x \cdot y}{\|x\| \|y\|}
$$

Onde $x$ e $y$ são vetores de embeddings dos exemplos, e $sim(x, y)$ é a similaridade do cosseno entre eles [11].

#### Perguntas Técnicas/Teóricas

1. Como você implementaria um sistema de seleção dinâmica de demonstrações para prompts preenchidos usando embeddings de sentenças?
2. Quais são as considerações éticas ao usar LLMs para gerar automaticamente variações de prompts para tarefas sensíveis?

### Conclusão

Os prompts preenchidos são uma ferramenta poderosa no arsenal do processamento de linguagem natural moderno. Eles permitem a criação eficiente de grandes volumes de dados de treinamento para ajuste fino de LLMs, facilitando a adaptação desses modelos para uma ampla gama de tarefas específicas [1][3]. Ao combinar templates cuidadosamente projetados com dados extraídos de datasets existentes, os pesquisadores podem gerar prompts diversificados e contextualizados que melhoram significativamente o desempenho dos LLMs em tarefas de seguir instruções [5][8].

No entanto, é crucial reconhecer as limitações e potenciais armadilhas desta abordagem. A qualidade dos prompts preenchidos depende fortemente da qualidade dos templates e dos dados de entrada, e um design inadequado pode introduzir vieses ou limitar a flexibilidade do modelo [9]. Técnicas avançadas, como geração automática e seleção baseada em similaridade, oferecem caminhos promissores para superar algumas dessas limitações [10][11].

À medida que o campo continua a evoluir, é provável que vejamos desenvolvimentos ainda mais sofisticados nas técnicas de preenchimento de prompts, possivelmente incorporando métodos de aprendizado de máquina para otimizar automaticamente os templates e os processos de seleção de demonstrações.

### Perguntas Avançadas

1. Como você projetaria um experimento para comparar a eficácia de prompts preenchidos gerados manualmente versus aqueles gerados automaticamente por um LLM em uma tarefa complexa de raciocínio?

2. Discuta as implicações éticas e práticas de usar prompts preenchidos para ajustar LLMs em tarefas que envolvem informações sensíveis ou potencialmente prejudiciais.

3. Proponha uma arquitetura de sistema que combine prompts preenchidos com técnicas de aprendizado por reforço para melhorar continuamente a qualidade dos prompts gerados.

### Referências

[1] "Um prompt é uma string de texto que um usuário emite para um modelo de linguagem para fazer o modelo fazer algo útil." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[2] "Muitos datasets de instrução tuning enormes foram criados, cobrindo muitas tarefas e idiomas." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[3] "Por instrução, temos em mente uma descrição em linguagem natural de uma tarefa a ser realizada, combinada com demonstrações rotuladas de tarefas." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[4] "Considere os seguintes templates para uma variedade de tarefas:" (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[5] "Para gerar dados de instrução-tuning, esses campos e os rótulos verdadeiros são extraídos dos dados de treinamento, codificados como pares chave/valor e inseridos em templates (Fig. 12.7) para produzir instruções instanciadas." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[6] "Não gostei do serviço que me foi prestado, quando entrei no hotel. Também não gostei da área, em que o hotel estava localizado. Muito barulho e eventos acontecendo para eu me sentir relaxado. Em resumo, nossa estadia foi" (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[7] "Beyoncé Giselle Knowles-Carter (nascida em 4 de setembro de 1981) é uma cantora, compositora, produtora musical e atriz americana. Nascida e criada em Houston, Texas, ela se apresentou em várias competições de canto e dança quando criança, e alcançou fama no final dos anos 1990 como vocalista principal do grupo feminino de R&B Destiny's Child." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[8] "Desenvolver dados de treinamento supervisionados de alta qualidade dessa maneira é demorado e caro. Uma abordagem mais comum faz uso das copiosas quantidades de dados de treinamento supervisionados que foram curados ao longo dos anos para uma ampla gama de tarefas de linguagem natural." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[9] "Para abordar essa questão, grandes datasets de instrução-tuning são particionados em clusters com base na similaridade da tarefa." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[10] "Uma maneira final de gerar datasets de instrução-tuning que está se tornando mais comum é usar modelos de linguagem para ajudar em cada estágio." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)

[11] "Por exemplo, usar demonstrações que são semelhantes à entrada atual parece melhorar o desempenho. Portanto, pode ser útil recuperar demonstrações dinamicamente para cada entrada, com base em sua similaridade com o exemplo atual (por exemplo, comparando o embedding da entrada atual com embeddings de cada exemplo do conjunto de treinamento para encontrar o melhor top-T)." (Excerpt from Chapter 12: Model Alignment, Prompting, and In-Context Learning)