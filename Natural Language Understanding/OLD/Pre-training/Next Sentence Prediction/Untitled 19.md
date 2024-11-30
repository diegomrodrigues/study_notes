## Amostragem de Pares de Sentenças em Modelos de Linguagem Pré-treinados

<image: Uma ilustração mostrando duas colunas: à esquerda, pares de sentenças conectadas por setas, representando a amostragem com NSP; à direita, uma sequência contínua de sentenças, representando a amostragem sem NSP.>

### Introdução

A amostragem de pares de sentenças é um componente crucial no treinamento de modelos de linguagem pré-treinados, especialmente aqueles que incorporam objetivos de aprendizado específicos para capturar relações entre sentenças. Este resumo explora detalhadamente as técnicas de amostragem utilizadas em modelos que empregam a Previsão da Próxima Sentença (Next Sentence Prediction - NSP) e as contrasta com abordagens que utilizam amostragem de sentenças contíguas, sem NSP [1][2].

### Conceitos Fundamentais

| Conceito                              | Explicação                                                   |
| ------------------------------------- | ------------------------------------------------------------ |
| **Next Sentence Prediction (NSP)**    | Objetivo de aprendizado onde o modelo é treinado para prever se duas sentenças são consecutivas no texto original ou não. [2] |
| **Amostragem com NSP**                | Processo de seleção de pares de sentenças para treinamento, onde 50% são pares consecutivos reais e 50% são pares aleatórios não relacionados. [2] |
| **Amostragem de Sentenças Contíguas** | Abordagem alternativa que utiliza sequências contínuas de texto, sem a necessidade de pares de sentenças explícitos ou classificação de relação entre sentenças. [5] |
| **Tokenização de Subpalavras**        | Técnica de segmentação do texto em unidades menores que palavras, utilizada em modelos como BERT e seus descendentes. [1] |

> ⚠️ **Nota Importante**: A escolha entre amostragem com NSP e amostragem de sentenças contíguas pode impactar significativamente o desempenho do modelo em tarefas específicas.

### Amostragem com Next Sentence Prediction (NSP)

<image: Diagrama mostrando o processo de seleção de pares de sentenças para NSP, com exemplos de pares positivos e negativos.>

A amostragem com NSP é uma técnica fundamental no treinamento de modelos como BERT (Bidirectional Encoder Representations from Transformers) [2]. O processo de amostragem segue estas etapas:

1. **Seleção de Pares Positivos**: 
   - Escolhe-se uma sentença A do corpus de treinamento.
   - A sentença B é selecionada como a sentença que segue imediatamente A no texto original.
   - Este par (A, B) é rotulado como um exemplo positivo.

2. **Seleção de Pares Negativos**:
   - Para criar um exemplo negativo, mantém-se a sentença A.
   - A sentença B é substituída por uma sentença aleatória de outro documento do corpus.
   - Este par (A, B') é rotulado como um exemplo negativo.

3. **Balanceamento**:
   - O conjunto de treinamento é construído com 50% de exemplos positivos e 50% de exemplos negativos.

4. **Preparação para Input**:
   - Os pares de sentenças são tokenizados usando um modelo de subpalavras.
   - Tokens especiais são adicionados:
     - [CLS] no início do par
     - [SEP] entre as sentenças e ao final

A formalização matemática do objetivo NSP pode ser representada como:

$$
L_{NSP} = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1-y_i) \log(1-p_i)]
$$

Onde:
- $N$ é o número total de pares de sentenças
- $y_i$ é o rótulo verdadeiro (1 para pares consecutivos, 0 para pares aleatórios)
- $p_i$ é a probabilidade prevista pelo modelo de que o par seja consecutivo

> ✔️ **Ponto de Destaque**: A inclusão do objetivo NSP visa capturar relações semânticas e de coerência entre sentenças, potencialmente melhorando o desempenho em tarefas que requerem compreensão de contexto mais amplo.

#### Questões Técnicas/Teóricas

1. Como a proporção de 50/50 entre exemplos positivos e negativos no NSP pode influenciar o aprendizado do modelo? Discuta possíveis vantagens e desvantagens desta abordagem.

2. Considerando a fórmula do $L_{NSP}$, como você interpretaria o impacto de um desequilíbrio significativo entre pares positivos e negativos no conjunto de treinamento?

### Amostragem de Sentenças Contíguas

<image: Ilustração de um fluxo contínuo de texto sendo segmentado em sequências de comprimento fixo, sem separação explícita entre sentenças.>

Modelos posteriores, como RoBERTa, abandonaram o objetivo NSP em favor de uma abordagem de amostragem de sentenças contíguas [5]. Esta técnica apresenta as seguintes características:

1. **Seleção de Sequências**:
   - Sequências contínuas de texto são selecionadas do corpus, sem distinção explícita entre sentenças.
   - O comprimento total é limitado a um número fixo de tokens (por exemplo, 512).

2. **Preenchimento Dinâmico**:
   - Se uma sequência termina antes de atingir o limite de tokens, um token separador especial é adicionado.
   - O restante é preenchido com texto do próximo documento até atingir o comprimento desejado.

3. **Tokenização**:
   - A sequência completa é tokenizada usando um modelo de subpalavras, sem adição de tokens especiais entre sentenças.

4. **Mascaramento**:
   - Tokens são mascarados aleatoriamente ao longo de toda a sequência para o objetivo de Masked Language Modeling (MLM).

A função de perda para esta abordagem se concentra apenas no MLM:

$$
L_{MLM} = - \frac{1}{|M|} \sum_{i \in M} \log P(x_i|z_i)
$$

Onde:
- $M$ é o conjunto de tokens mascarados
- $x_i$ é o token original
- $z_i$ é a representação contextual do token mascarado

> ❗ **Ponto de Atenção**: A eliminação do NSP e a adoção de sequências contíguas mais longas podem resultar em um treinamento mais eficiente e em melhor captura de dependências de longo alcance.

#### Vantagens e Desvantagens

| 👍 Vantagens da Amostragem Contígua                           | 👎 Desvantagens da Amostragem Contígua                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Captura melhor dependências de longo alcance [5]             | Pode perder informações específicas sobre relações entre sentenças [2] |
| Treinamento mais eficiente com menos overhead [5]            | Potencial redução no desempenho em tarefas que requerem compreensão explícita de pares de sentenças [2] |
| Simplifica o processo de pré-processamento e treinamento [5] | Pode ser menos eficaz em tarefas como detecção de paráfrase ou inferência textual [2] |

#### Questões Técnicas/Teóricas

1. Como a ausência de um objetivo explícito de relação entre sentenças (como NSP) pode afetar o desempenho do modelo em tarefas de inferência textual? Proponha uma abordagem para mitigar possíveis limitações.

2. Considerando a função de perda $L_{MLM}$, como você poderia modificá-la para incorporar algum tipo de informação sobre a estrutura das sentenças, sem reintroduzir completamente o NSP?

### Implementação Prática

A implementação da amostragem de pares de sentenças ou sentenças contíguas geralmente é realizada durante o pré-processamento dos dados. Aqui está um exemplo simplificado de como isso poderia ser feito usando Python e a biblioteca Hugging Face Transformers:

```python
from transformers import BertTokenizer
import random

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def sample_sentence_pair(corpus, use_nsp=True):
    if use_nsp:
        # Amostragem com NSP
        doc1, doc2 = random.sample(corpus, 2)
        sent1 = random.choice(doc1)
        if random.random() < 0.5:
            # Par positivo
            sent2 = doc1[doc1.index(sent1) + 1] if doc1.index(sent1) < len(doc1) - 1 else random.choice(doc1)
            is_next = 1
        else:
            # Par negativo
            sent2 = random.choice(doc2)
            is_next = 0
        
        tokens = tokenizer.encode_plus(sent1, sent2, max_length=512, truncation=True, padding='max_length')
        tokens['next_sentence_label'] = is_next
    else:
        # Amostragem contígua
        doc = random.choice(corpus)
        text = ' '.join(doc)
        tokens = tokenizer.encode_plus(text, max_length=512, truncation=True, padding='max_length')
    
    return tokens

# Uso
corpus = [["Sentence 1", "Sentence 2", "Sentence 3"], ["Sentence A", "Sentence B", "Sentence C"]]
sample_with_nsp = sample_sentence_pair(corpus, use_nsp=True)
sample_without_nsp = sample_sentence_pair(corpus, use_nsp=False)
```

> 💡 **Dica**: Na prática, a implementação seria mais complexa, lidando com questões como balanceamento do dataset, tratamento de documentos curtos, e otimização de performance para grandes corpora.

### Conclusão

A escolha entre amostragem com NSP e amostragem de sentenças contíguas representa um trade-off importante no design de modelos de linguagem pré-treinados [2][5]. Enquanto a abordagem NSP oferece benefícios potenciais para tarefas que requerem compreensão explícita de relações entre sentenças, a amostragem contígua simplifica o treinamento e pode capturar melhor dependências de longo alcance. A decisão deve ser baseada nos objetivos específicos do modelo e nas tarefas downstream previstas.

### Questões Avançadas

1. Dado um cenário onde você precisa pré-treinar um modelo de linguagem para uma tarefa específica de detecção de coerência textual em documentos longos, como você modificaria a estratégia de amostragem para otimizar o desempenho nesta tarefa, considerando as limitações de memória e computação?

2. Considerando as diferenças entre amostragem com NSP e amostragem contígua, proponha uma abordagem híbrida que tente capturar os benefícios de ambas as técnicas. Como você avaliaria a eficácia desta abordagem em comparação com as técnicas existentes?

3. Analise criticamente o impacto da tokenização de subpalavras na eficácia da amostragem de pares de sentenças. Como as escolhas de tokenização podem afetar a capacidade do modelo de capturar relações semânticas entre sentenças, e que modificações você sugeriria para mitigar possíveis problemas?

### Referências

[1] "An English-only subword vocabulary consisting of 30,000 tokens generated using the WordPiece algorithm (Schuster and Nakajima, 2012)." (Trecho de Fine-Tuning and Masked Language Models)

[2] "To facilitate NSP training, BERT introduces two new tokens to the input representation (tokens that will prove useful for fine-tuning as well). After tokenizing the input with the subword model, the token [CLS] is prepended to the input sentence pair, and the token [SEP] is placed between the sentences and after the final token of the second sentence." (Trecho de Fine-Tuning and Masked Language Models)

[3] "In BERT, 50% of the training pairs consisted of positive pairs, and in the other 50% the second sentence of a pair was randomly selected from elsewhere in the corpus. The NSP loss is based on how well the model can distinguish true pairs from random pairs." (Trecho de Fine-Tuning and Masked Language Models)

[4] "Cross entropy is used to compute the NSP loss for each sentence pair presented to the model." (Trecho de Fine-Tuning and Masked Language Models)

[5] "Some models, like the RoBERTa model, drop the next sentence prediction objective, and therefore change the training regime a bit. Instead of sampling pairs of sentence, the input is simply a series of contiguous sentences. If the document runs out before 512 tokens are reached, an extra separator token is added, and sentences from the next document are packed in, until we reach a total of 512 tokens." (Trecho de Fine-Tuning and Masked Language Models)