## Tokens Especiais [CLS] e [SEP] em Modelos de Linguagem Pré-treinados

<image: Um diagrama mostrando uma sequência de tokens de entrada, com [CLS] no início, [SEP] separando duas frases, e tokens de palavras entre eles. Setas apontando para representações vetoriais acima dos tokens especiais.>

### Introdução

Os tokens especiais [CLS] e [SEP] desempenham um papel crucial na estruturação e representação de entradas em modelos de linguagem pré-treinados como BERT (Bidirectional Encoder Representations from Transformers) [1]. Estes tokens são fundamentais para tarefas como classificação de sequências e previsão da próxima frase (Next Sentence Prediction - NSP), permitindo que os modelos processem eficientemente pares de sentenças e extraiam informações relevantes para várias tarefas de processamento de linguagem natural (NLP) [2].

### Conceitos Fundamentais

| Conceito                           | Explicação                                                   |
| ---------------------------------- | ------------------------------------------------------------ |
| **[CLS] Token**                    | Token especial adicionado ao início de todas as sequências de entrada, usado para representar a sequência inteira para tarefas de classificação. [1] |
| **[SEP] Token**                    | Token especial usado para separar pares de sentenças em uma única sequência de entrada, crucial para tarefas que envolvem relações entre sentenças. [2] |
| **Next Sentence Prediction (NSP)** | Tarefa de treinamento onde o modelo prevê se duas sentenças são consecutivas no texto original, utilizando os tokens [CLS] e [SEP]. [3] |

> ✔️ **Ponto de Destaque**: A introdução dos tokens [CLS] e [SEP] permite que modelos como BERT processem eficientemente pares de sentenças e realizem tarefas complexas de NLP com uma única arquitetura de modelo.

### Papel e Funcionamento dos Tokens Especiais

#### [CLS] Token

O token [CLS] (abreviação de "classification") é prepended a todas as sequências de entrada e serve como um agregador de informações para toda a sequência [1]. 

<image: Diagrama mostrando uma sequência de tokens com [CLS] no início, seguido por tokens de palavras, e uma seta apontando do [CLS] para um vetor de classificação no topo.>

Funcionamento:
1. É adicionado ao início de cada sequência de entrada.
2. Durante o processamento, acumula informações de toda a sequência através das camadas de atenção.
3. A representação final do [CLS] no último layer é usada como entrada para tarefas de classificação de sequência [4].

#### [SEP] Token

O token [SEP] (abreviação de "separator") é usado para separar pares de sentenças em uma única sequência de entrada [2].

<image: Diagrama mostrando duas sentenças separadas por [SEP], com [CLS] no início e [SEP] no final.>

Funcionamento:
1. É inserido entre duas sentenças quando elas são combinadas em uma única sequência.
2. Ajuda o modelo a distinguir entre diferentes segmentos da entrada.
3. É crucial para tarefas que envolvem relações entre sentenças, como NSP e inferência de linguagem natural [5].

### Aplicação em Next Sentence Prediction (NSP)

A tarefa de NSP é uma parte importante do treinamento de modelos como BERT, utilizando os tokens [CLS] e [SEP] para estruturar a entrada [3].

Processo:
1. Duas sentenças são selecionadas do corpus de treinamento.
2. A sequência de entrada é formada como: [CLS] Sentença A [SEP] Sentença B [SEP]
3. O modelo é treinado para prever se a Sentença B é a continuação real da Sentença A no texto original.

Matematicamente, a previsão é feita da seguinte forma:

$$
y_i = \text{softmax}(W_{NSP}h_i)
$$

Onde:
- $h_i$ é a representação final do token [CLS]
- $W_{NSP}$ é uma matriz de pesos aprendida
- $y_i$ é a distribuição de probabilidade sobre as duas classes (próxima sentença ou não)

> ⚠️ **Nota Importante**: A tarefa de NSP ajuda o modelo a aprender relações de longo alcance entre sentenças, crucial para muitas aplicações downstream de NLP.

#### Questões Técnicas/Teóricas

1. Como o token [CLS] contribui para tarefas de classificação de sequência em modelos como BERT?
2. Explique como o token [SEP] facilita o processamento de pares de sentenças em tarefas como inferência de linguagem natural.

### Impacto na Representação e Fine-tuning

A inclusão dos tokens [CLS] e [SEP] tem um impacto significativo na forma como os modelos pré-treinados são fine-tuned para tarefas específicas.

#### Classificação de Sequência

Para tarefas de classificação, como análise de sentimento, o vetor de saída correspondente ao token [CLS] é usado como entrada para um classificador [6]:

$$
p = \text{softmax}(W_C z_{CLS})
$$

Onde:
- $z_{CLS}$ é o vetor de saída do último layer para o token [CLS]
- $W_C$ é uma matriz de pesos específica da tarefa
- $p$ é a distribuição de probabilidade sobre as classes

#### Classificação de Pares de Sequências

Em tarefas que envolvem pares de sentenças, como detecção de paráfrase ou inferência textual, a estrutura [CLS] Sentença A [SEP] Sentença B [SEP] é mantida durante o fine-tuning [7].

### Vantagens e Considerações

| 👍 Vantagens                                                  | 👎 Considerações                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Permite processamento eficiente de pares de sentenças [8]    | Pode introduzir overhead em tarefas que não requerem representação de pares [9] |
| Facilita transfer learning para diversas tarefas de NLP [8]  | O token [CLS] pode não capturar perfeitamente toda a informação da sequência em alguns casos [10] |
| Simplifica a arquitetura do modelo para múltiplas tarefas [8] | A eficácia do NSP como tarefa de pré-treinamento tem sido questionada em alguns estudos recentes [11] |

### Conclusão

Os tokens especiais [CLS] e [SEP] são componentes fundamentais na arquitetura de modelos de linguagem pré-treinados como BERT. Eles permitem uma representação estruturada de sequências e pares de sequências, facilitando o treinamento em tarefas como NSP e o fine-tuning para diversas aplicações de NLP. Embora sua introdução tenha revolucionado o campo, pesquisas contínuas exploram maneiras de otimizar ainda mais seu uso e impacto.

### Questões Avançadas

1. Como você adaptaria o uso dos tokens [CLS] e [SEP] para uma tarefa de classificação multi-sentença, onde mais de duas sentenças precisam ser consideradas simultaneamente?
2. Discuta as implicações de remover a tarefa de NSP do pré-treinamento, como feito em alguns modelos posteriores ao BERT. Como isso afetaria o uso e a eficácia dos tokens [CLS] e [SEP]?
3. Proponha uma modificação na arquitetura ou no processo de treinamento que poderia melhorar a capacidade do token [CLS] de capturar informações globais da sequência para tarefas de classificação.

### Referências

[1] "A unique vocabulary token [CLS] is added to the vocabulary and is prepended to the start of all input sequences, both during pretraining and encoding." (Trecho de Fine-Tuning and Masked Language Models)

[2] "To facilitate NSP training, BERT introduces two new tokens to the input representation (tokens that will prove useful for fine-tuning as well). After tokenizing the input with the subword model, the token [CLS] is prepended to the input sentence pair, and the token [SEP] is placed between the sentences and after the final token of the second sentence." (Trecho de Fine-Tuning and Masked Language Models)

[3] "In this task, the model is presented with pairs of sentences and is asked to predict whether each pair consists of an actual pair of adjacent sentences from the training corpus or a pair of unrelated sentences." (Trecho de Fine-Tuning and Masked Language Models)

[4] "The output vector in the final layer of the model for the [CLS] input represents the entire input sequence and serves as the input to a classifier head, a logistic regression or neural network classifier that makes the relevant decision." (Trecho de Fine-Tuning and Masked Language Models)

[5] "Finally, embeddings representing the first and second segments of the input are added to the word and positional embeddings to allow the model to more easily distinguish the input sentences." (Trecho de Fine-Tuning and Masked Language Models)

[6] "Classification of unseen documents proceeds by passing the input text through the pretrained language model to generate z_CLS, multiplying it by W_C, and finally passing the resulting vector through a softmax." (Trecho de Fine-Tuning and Masked Language Models)

[7] "During fine-tuning, pairs of labeled sentences from the supervised training data are presented to the model, and run through all the layers of the model to produce the z outputs for each input token." (Trecho de Fine-Tuning and Masked Language Models)

[8] "The focus of bidirectional encoders is instead on computing contextualized representations of the input tokens." (Trecho de Fine-Tuning and Masked Language Models)

[9] "As with RNNs, a greedy approach, where the argmax tag for each token is taken as a likely answer, can be used to generate the final output tag sequence." (Trecho de Fine-Tuning and Masked Language Models)

[10] "Note that only the tokens in M play a role in learning; the other words play no role in the loss function, so in that sense BERT and its descendents are inefficient; only 15% of the input samples in the training data are actually used for training weights." (Trecho de Fine-Tuning and Masked Language Models)

[11] "Some models, like the RoBERTa model, drop the next sentence prediction objective, and therefore change the training regime a bit." (Trecho de Fine-Tuning and Masked Language Models)