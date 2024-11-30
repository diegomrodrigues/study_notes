## Subword Tokenization: WordPiece e SentencePiece para Tokenização e suas Implicações

<image: Um diagrama mostrando um texto sendo dividido em subpalavras, com destaque para os tokens gerados pelos algoritmos WordPiece e SentencePiece, e setas apontando para diferentes tarefas de NLP downstream>

### Introdução

A tokenização de subpalavras é um componente crucial nos modelos de linguagem modernos, especialmente em arquiteturas como BERT e seus descendentes. Este resumo explora em profundidade o uso dos algoritmos WordPiece e SentencePiece para tokenização e analisa suas implicações para tarefas downstream em processamento de linguagem natural (NLP) [1][2].

> ✔️ **Ponto de Destaque**: A tokenização de subpalavras é fundamental para lidar com vocabulários extensos e palavras desconhecidas em modelos de linguagem.

### Conceitos Fundamentais

| Conceito                 | Explicação                                                   |
| ------------------------ | ------------------------------------------------------------ |
| **Subword Tokenization** | Processo de dividir palavras em unidades menores (subpalavras) para reduzir o vocabulário e lidar com palavras desconhecidas [1] |
| **WordPiece**            | Algoritmo de tokenização desenvolvido pela Google, usado no BERT original [1] |
| **SentencePiece**        | Algoritmo de tokenização mais recente, que inclui o modelo Unigram LM, usado em modelos multilíngues como XLM-RoBERTa [2] |

### WordPiece: O Pioneiro na Tokenização de Subpalavras

<image: Um fluxograma detalhando o processo de tokenização WordPiece, mostrando a divisão de palavras em subpalavras e a criação do vocabulário>

O algoritmo WordPiece, desenvolvido pela Google, foi utilizado no modelo BERT original e é fundamental para entender a evolução da tokenização de subpalavras [1].

#### Funcionamento do WordPiece

1. Inicializa o vocabulário com caracteres individuais.
2. Iterativamente, encontra o par de tokens mais frequente e o mescla.
3. Continua até atingir o tamanho desejado do vocabulário ou um limite predefinido.

> ⚠️ **Nota Importante**: O vocabulário WordPiece do BERT original consiste em 30.000 tokens, um número significativamente menor que o vocabulário total de palavras em inglês [1].

#### Vantagens do WordPiece

- Reduz significativamente o tamanho do vocabulário.
- Lida eficientemente com palavras desconhecidas (OOV - Out of Vocabulary).
- Melhora a representação de palavras raras e compostas.

#### Desvantagens do WordPiece

- Pode fragmentar excessivamente palavras comuns.
- A segmentação pode não ser linguisticamente motivada em todos os casos.

### SentencePiece: Evolução na Tokenização Multilíngue

<image: Um diagrama comparativo entre WordPiece e SentencePiece, destacando as diferenças no tratamento de espaços e a abordagem language-agnostic do SentencePiece>

O SentencePiece, especificamente seu algoritmo Unigram LM, representa uma evolução na tokenização de subpalavras, especialmente para modelos multilíngues como o XLM-RoBERTa [2].

#### Características do SentencePiece Unigram LM

1. Trata espaços como símbolos normais, permitindo uma tokenização language-agnostic.
2. Utiliza um modelo probabilístico para determinar a segmentação ótima.
3. Permite um vocabulário muito maior, de até 250.000 tokens para modelos multilíngues [2].

> ❗ **Ponto de Atenção**: O SentencePiece é especialmente eficaz para línguas sem espaços claros entre palavras, como o chinês ou o japonês.

#### Formulação Matemática do Unigram LM

O algoritmo Unigram LM do SentencePiece busca maximizar a verossimilhança dos dados de treinamento:

$$
\mathcal{L} = \sum_{s \in \mathcal{S}} \log P(s)
$$

Onde $\mathcal{S}$ é o conjunto de sentenças de treinamento e $P(s)$ é a probabilidade de uma sentença $s$.

A probabilidade de uma sentença é calculada como:

$$
P(s) = \prod_{i=1}^{m} P(x_i)
$$

Onde $x_1, ..., x_m$ é a segmentação de $s$ em subpalavras e $P(x_i)$ é a probabilidade de cada subpalavra.

#### Vantagens do SentencePiece

- Melhor desempenho em tarefas multilíngues.
- Tratamento uniforme de diferentes sistemas de escrita.
- Capacidade de lidar com vocabulários muito maiores.

#### Desvantagens do SentencePiece

- Pode ser computacionalmente mais intensivo.
- A interpretabilidade dos tokens pode ser menor em alguns casos.

#### Questões Técnicas/Teóricas

1. Como o algoritmo Unigram LM do SentencePiece lida com a ambiguidade na segmentação de palavras?
2. Qual é o impacto do tamanho do vocabulário na qualidade das representações de subpalavras em modelos como BERT e XLM-RoBERTa?

### Implicações para Tarefas Downstream

A escolha do método de tokenização tem implicações significativas para várias tarefas de NLP downstream [3][4].

#### Named Entity Recognition (NER)

- **Desafio**: Alinhar tokens de subpalavras com anotações BIO em nível de palavra [4].
- **Solução**: Atribuir a tag BIO da palavra original a todos os tokens de subpalavra derivados dela durante o treinamento [4].

#### Exemplo de Implementação para NER com Subpalavras

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")

text = "Mt. Sanitas is in Sunshine Canyon"
labels = ["B-LOC", "I-LOC", "O", "O", "B-LOC", "I-LOC", "O"]

# Tokenização
tokens = tokenizer.tokenize(text)
word_ids = tokenizer.word_ids(tokens)

# Alinhamento de labels
aligned_labels = []
for word_id in word_ids:
    if word_id is None:
        aligned_labels.append(-100)  # Ignorar tokens especiais
    else:
        aligned_labels.append(labels[word_id])

# Codificação e inferência
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Decodificação
predictions = torch.argmax(outputs.logits, dim=2)
predicted_labels = [model.config.id2label[t.item()] for t in predictions[0]]

# Mapeamento de volta para palavras
final_labels = []
for word_id, label in zip(word_ids, predicted_labels):
    if word_id is not None and (len(final_labels) == 0 or word_id != len(final_labels) - 1):
        final_labels.append(label)

print(f"Palavras: {text.split()}")
print(f"Labels previstas: {final_labels}")
```

> ✔️ **Ponto de Destaque**: A implementação acima demonstra como lidar com o desalinhamento entre tokens de subpalavras e labels em nível de palavra para NER.

#### Question Answering e Span-based Tasks

- **Desafio**: Identificar spans precisos quando as respostas não se alinham com os limites dos tokens [3].
- **Solução**: Utilizar técnicas de span boundary detection que consideram a granularidade das subpalavras [3].

#### Análise Sintática e Semântica

- **Implicação**: A fragmentação de palavras pode afetar a identificação de estruturas sintáticas e relações semânticas [3].
- **Abordagem**: Desenvolver métodos que agreguem informações de subpalavras para reconstruir representações em nível de palavra quando necessário [3].

### Adaptações e Otimizações

Para lidar com os desafios impostos pela tokenização de subpalavras, várias adaptações e otimizações foram desenvolvidas:

1. **Span-based Masking**: Técnica que mascara spans inteiros de tokens durante o pré-treinamento, melhorando a captura de dependências de longo alcance [5].

2. **Contextual Embeddings**: Utilização de embeddings contextuais que consideram a posição e o contexto de cada subpalavra [6].

$$
e_i = \text{Embedding}(x_i) + \text{PositionalEncoding}(i)
$$

Onde $e_i$ é o embedding final do token $i$, $x_i$ é o token de entrada, e $\text{PositionalEncoding}(i)$ é o encoding posicional.

3. **Adaptive Tokenization**: Métodos que ajustam dinamicamente a granularidade da tokenização baseado na tarefa ou domínio específico [7].

> ❗ **Ponto de Atenção**: A escolha entre WordPiece e SentencePiece deve considerar o equilíbrio entre eficiência computacional e cobertura linguística, especialmente em cenários multilíngues.

#### Questões Técnicas/Teóricas

1. Como a técnica de span-based masking afeta a capacidade do modelo em capturar dependências de longo alcance em comparação com o masking tradicional de tokens individuais?
2. Quais são as considerações ao projetar um esquema de tokenização adaptativo que possa se ajustar a diferentes domínios ou tarefas?

### Conclusão

A tokenização de subpalavras, implementada através de algoritmos como WordPiece e SentencePiece, revolucionou o processamento de linguagem natural, permitindo modelos mais eficientes e linguisticamente flexíveis [1][2]. Enquanto o WordPiece foi fundamental para o sucesso inicial de modelos como BERT, o SentencePiece, com seu algoritmo Unigram LM, expandiu as capacidades para cenários multilíngues [2].

As implicações para tarefas downstream são significativas, exigindo adaptações cuidadosas em áreas como NER, question answering e análise sintática [3][4]. A comunidade de NLP continua a desenvolver técnicas inovadoras para aproveitar ao máximo a granularidade oferecida pela tokenização de subpalavras, ao mesmo tempo em que aborda os desafios inerentes a essa abordagem.

À medida que avançamos, a otimização desses métodos de tokenização e o desenvolvimento de técnicas para lidar eficientemente com suas peculiaridades continuarão sendo áreas cruciais de pesquisa e desenvolvimento em NLP.

### Questões Avançadas

1. Como podemos projetar um sistema de tokenização que balance eficientemente a cobertura de vocabulário, a capacidade de lidar com palavras desconhecidas e a preservação de informações morfológicas em um contexto multilíngue?

2. Considerando as limitações atuais da tokenização de subpalavras em tarefas que dependem fortemente de estruturas sintáticas, como podemos desenvolver modelos que integrem melhor o conhecimento linguístico estrutural com as representações de subpalavras?

3. Dado o trade-off entre o tamanho do vocabulário e a eficiência computacional, como podemos determinar o tamanho ótimo de vocabulário para diferentes aplicações de NLP, especialmente em cenários de recursos computacionais limitados?

### Referências

[1] "An English-only subword vocabulary consisting of 30,000 tokens generated using the WordPiece algorithm (Schuster and Nakajima, 2012)." (Trecho de Fine-Tuning and Masked Language Models)

[2] "A multilingual subword vocabulary with 250,000 tokens generated using the SentencePiece Unigram LM algorithm (Kudo and Richardson, 2018)." (Trecho de Fine-Tuning and Masked Language Models)

[3] "For many purposes, a pretrained multilingual model is more practical than a monolingual model, since it avoids the need to build many (100!) separate monolingual models. And multilingual models can improve performance on low-resourced languages by leveraging linguistic information from a similar language in the training data that happens to have more resources." (Trecho de Fine-Tuning and Masked Language Models)

[4] "To deal with this misalignment, we need a way to assign BIO tags to subword tokens during training and a corresponding way to recover word-level tags from subwords during decoding. For training, we can just assign the gold-standard tag associated with each word to all of the subword tokens derived from it." (Trecho de Fine-Tuning and Masked Language Models)

[5] "Once a span is chosen for masking, all the tokens within the span are substituted according to the same regime used in BERT: 80% of the time the span elements are substituted with the [MASK] token, 10% of the time they are replaced by randomly sampled tokens from the vocabulary, and 10% of the time they are left as is." (Trecho de Fine-Tuning and Masked Language Models)

[6] "Just as we used static embeddings like word2vec in Chapter 6 to represent the meaning of words, we can use contextual embeddings as representations of word meanings in context for any task that might require a model of word meaning." (Trecho de Fine-Tuning and Masked Language Models)

[7] "Multilingual models similarly use webtext and multilingual Wikipedia. For example the XLM-R model was trained on about 300 billion tokens in 100 languages, taken from the web via Common Crawl (https://commoncrawl.org/)." (Trecho de Fine-Tuning and Masked Language Models)