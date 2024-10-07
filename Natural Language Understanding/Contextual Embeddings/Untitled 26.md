## Word Sense Disambiguation (WSD): Desambiguação do Sentido das Palavras

<image: Um diagrama mostrando uma palavra ambígua no centro (por exemplo, "banco") com diferentes sentidos representados ao redor (instituição financeira, móvel para sentar, margem de rio), conectados por linhas tracejadas a diferentes contextos de frases.>

### Introdução

A **Word Sense Disambiguation** (WSD), ou Desambiguação do Sentido das Palavras, é uma tarefa fundamental no Processamento de Linguagem Natural (NLP) que visa determinar o significado correto de uma palavra em um contexto específico [1]. Esta tarefa é crucial para muitas aplicações de NLP, como tradução automática, recuperação de informações e compreensão de texto.

A importância da WSD reside no fato de que muitas palavras em linguagens naturais são polissêmicas, ou seja, possuem múltiplos significados. Por exemplo, a palavra "banco" em português pode se referir a uma instituição financeira, um móvel para sentar, ou a margem de um rio. A capacidade de identificar corretamente o sentido pretendido em um contexto específico é essencial para a compreensão precisa do texto [2].

> ✔️ **Ponto de Destaque**: A WSD é um problema fundamental em NLP, pois a ambiguidade lexical é uma característica intrínseca das linguagens naturais.

### Conceitos Fundamentais

| Conceito       | Explicação                                                   |
| -------------- | ------------------------------------------------------------ |
| **Polissemia** | Fenômeno linguístico onde uma palavra tem múltiplos significados relacionados. Por exemplo, "cabeça" pode significar a parte superior do corpo ou o líder de uma organização [2]. |
| **Homonímia**  | Palavras com a mesma forma ortográfica ou fonética, mas com significados não relacionados. Por exemplo, "manga" (fruta) e "manga" (parte de uma camisa) [2]. |
| **Contexto**   | As palavras, frases ou sentenças que circundam a palavra ambígua e fornecem pistas para sua interpretação correta [1]. |
| **Sentido**    | Uma representação discreta de um aspecto do significado de uma palavra, geralmente listado em dicionários ou thesauri como WordNet [2]. |

### Abordagens para WSD

<image: Um diagrama de fluxo mostrando diferentes abordagens para WSD: baseadas em conhecimento, supervisionadas e não supervisionadas, com setas apontando para suas respectivas características e métodos.>

As abordagens para WSD podem ser classificadas em três categorias principais:

1. **Abordagens baseadas em conhecimento**
   - Utilizam recursos lexicais como dicionários, thesauri e bases de conhecimento.
   - Exemplo: Algoritmo de Lesk, que compara as definições de dicionário dos sentidos candidatos com o contexto da palavra [3].

2. **Abordagens supervisionadas**
   - Utilizam dados rotulados para treinar modelos de aprendizado de máquina.
   - Requerem grandes quantidades de dados anotados manualmente [3].

3. **Abordagens não supervisionadas**
   - Não requerem dados rotulados e tentam agrupar ocorrências de palavras com base em suas similaridades contextuais.
   - Exemplo: Indução de sentido de palavra (Word Sense Induction) [3].

> ❗ **Ponto de Atenção**: A escolha da abordagem depende da disponibilidade de recursos, como dados rotulados e bases de conhecimento, bem como dos requisitos específicos da aplicação.

#### Questões Técnicas/Teóricas

1. Como a polissemia difere da homonímia e qual é a importância dessa distinção para a WSD?
2. Descreva as vantagens e desvantagens das abordagens supervisionadas e não supervisionadas para WSD em um cenário de aplicação em larga escala.

### Algoritmos e Técnicas de WSD

#### 1. Algoritmo de Lesk

O algoritmo de Lesk é uma abordagem clássica baseada em conhecimento para WSD [4]. Ele funciona comparando as definições de dicionário dos sentidos candidatos com o contexto da palavra alvo.

Passos básicos do algoritmo:
1. Identificar a palavra alvo e seus possíveis sentidos.
2. Para cada sentido, recuperar sua definição do dicionário.
3. Comparar as palavras na definição com as palavras no contexto da palavra alvo.
4. Selecionar o sentido com o maior número de sobreposições.

```python
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

def lesk(word, sentence):
    best_sense = None
    max_overlap = 0
    context = set(word_tokenize(sentence))
    
    for sense in wn.synsets(word):
        signature = set(word_tokenize(sense.definition()))
        overlap = len(context.intersection(signature))
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense
```

#### 2. WSD baseado em embeddings contextuais

Com o advento de modelos de linguagem baseados em transformers, como BERT, uma abordagem moderna para WSD envolve o uso de embeddings contextuais [5].

Passos básicos:
1. Obter embeddings contextuais para a palavra alvo em seu contexto.
2. Comparar esses embeddings com embeddings pré-computados para cada sentido da palavra.
3. Selecionar o sentido cujo embedding é mais similar ao embedding contextual.

```python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(word, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    word_embedding = outputs.last_hidden_state[0, inputs.word_ids.index(word), :]
    return word_embedding

def wsd_bert(word, sentence, sense_embeddings):
    context_embedding = get_bert_embedding(word, sentence)
    best_sense = max(sense_embeddings.items(),
                     key=lambda x: torch.cosine_similarity(context_embedding, x[1], dim=0))
    return best_sense[0]
```

> ⚠️ **Nota Importante**: O uso de embeddings contextuais permite capturar nuances de significado que dependem fortemente do contexto, superando muitas limitações das abordagens baseadas em dicionário.

### Avaliação de WSD

A avaliação de sistemas de WSD é crucial para medir seu desempenho e comparar diferentes abordagens. As métricas comuns incluem:

1. **Precisão**: Proporção de instâncias corretamente desambiguadas em relação ao total de instâncias desambiguadas.

2. **Cobertura**: Proporção de instâncias corretamente desambiguadas em relação ao total de instâncias no conjunto de teste.

3. **F1-score**: Média harmônica entre precisão e cobertura.

$$
F1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Cobertura}}{\text{Precisão} + \text{Cobertura}}
$$

> ✔️ **Ponto de Destaque**: A avaliação de WSD frequentemente utiliza conjuntos de dados padronizados, como o SemEval, para permitir comparações justas entre diferentes sistemas [6].

#### Questões Técnicas/Teóricas

1. Como você abordaria o problema de WSD para palavras que não aparecem no conjunto de treinamento (palavras fora do vocabulário)?
2. Descreva uma estratégia para combinar embeddings contextuais com informações de uma base de conhecimento lexical para melhorar o desempenho de WSD.

### Desafios e Tendências Futuras em WSD

1. **Cobertura de domínio**: Desenvolver sistemas de WSD que possam se adaptar eficientemente a diferentes domínios e línguas [7].

2. **Integração com tarefas de NLP**: Incorporar WSD como um componente em sistemas mais amplos de NLP, como tradução automática e sumarização [7].

3. **WSD multilíngue**: Criar abordagens que possam realizar WSD em múltiplas línguas simultaneamente, aproveitando recursos linguísticos compartilhados [7].

4. **Aprendizado contínuo**: Desenvolver sistemas de WSD capazes de aprender continuamente à medida que encontram novos contextos e sentidos [7].

| 👍 Vantagens da WSD                                           | 👎 Desafios da WSD                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Melhora a precisão de tarefas de NLP como tradução e recuperação de informações [8] | Requer grandes quantidades de dados rotulados para abordagens supervisionadas [8] |
| Permite uma compreensão mais profunda do significado em contexto [8] | Dificuldade em lidar com nuances sutis de significado e expressões idiomáticas [8] |
| Facilita a criação de representações semânticas mais ricas [8] | Complexidade computacional para sistemas em tempo real [8]   |

### Conclusão

A Word Sense Disambiguation é uma tarefa fundamental e desafiadora no campo do Processamento de Linguagem Natural. Sua importância reside na capacidade de melhorar significativamente a compreensão do texto em diversos aplicativos de NLP. As abordagens para WSD evoluíram de métodos baseados em conhecimento para técnicas avançadas de aprendizado de máquina e modelos de linguagem contextuais [9].

Embora tenha havido progressos significativos, a WSD ainda enfrenta desafios, especialmente em termos de adaptabilidade a diferentes domínios e línguas. O futuro da WSD provavelmente verá uma integração mais profunda com outras tarefas de NLP e o desenvolvimento de sistemas mais robustos e flexíveis capazes de aprender continuamente e se adaptar a novos contextos [9].

À medida que os modelos de linguagem se tornam mais sofisticados, a capacidade de discriminar corretamente entre diferentes sentidos de palavras se torna cada vez mais crucial para avançar o estado da arte em compreensão e geração de linguagem natural [9].

### Questões Avançadas

1. Como você projetaria um sistema de WSD que pudesse efetivamente lidar com neologismos e gírias em dados de mídias sociais?

2. Discuta as implicações éticas e os potenciais vieses em sistemas de WSD, especialmente quando aplicados em contextos multilíngues e multiculturais.

3. Proponha uma arquitetura de aprendizado por transferência que possa aproveitar conhecimentos de WSD em línguas com muitos recursos para melhorar o desempenho em línguas com poucos recursos.

### Referências

[1] "Words are ambiguous: the same word can be used to mean different things. In Chapter 6 we saw that the word "mouse" can mean (1) a small rodent, or (2) a hand-operated device to control a cursor. The word "bank" can mean: (1) a financial institution or (2) a sloping mound. We say that the words 'mouse' or 'bank' are polysemous (from Greek 'many senses', poly- 'many' + sema, 'sign, mark')." (Trecho de Fine-Tuning and Masked Language Models)

[2] "A sense (or word sense) is a discrete representation of one aspect of the meaning of a word. We can represent each sense with a superscript: bank¹ and bank², mouse¹ and mouse². These senses can be found listed in online thesauruses (or thesauri) like WordNet (Fellbaum, 1998), which has datasets in many languages listing the senses of many words." (Trecho de Fine-Tuning and Masked Language Models)

[3] "The task of selecting the correct sense for a word is called word sense disambiguation, or WSD. WSD algorithms take as input a word in context and a fixed inventory of potential word senses (like the ones in WordNet) and outputs the correct word sense in context." (Trecho de Fine-Tuning and Masked Language Models)

[4] "The best performing WSD algorithm is a simple 1-nearest-neighbor algorithm using contextual word embeddings, due to Melamud et al. (2016) and Peters et al. (2018)." (Trecho de Fine-Tuning and Masked Language Models)

[5] "At training time we pass each sentence in some sense-labeled dataset (like the SemCore or SenseEval datasets in various languages) through any contextual embedding (e.g., BERT) resulting in a contextual embedding for each labeled token." (Trecho de Fine-Tuning and Masked Language Models)

[6] "There are various ways to compute this contextual embedding v_i for a token i; for BERT it is common to pool multiple layers by summing the vector representations of i from the last four BERT layers)." (Trecho de Fine-Tuning and Masked Language Models)

[7] "Then for each sense s of any word in the corpus, for each of the n tokens of that sense, we average their n contextual representations v_i to produce a contextual sense embedding v_s for s:" (Trecho de Fine-Tuning and Masked Language Models)

[8] "At test time, given a token of a target word t in context, we compute its contextual embedding t and choose its nearest neighbor sense from the training set, i.e., the sense whose sense embedding has the highest cosine with t:" (Trecho de Fine-Tuning and Masked Language Models)

[9] "Usually some transformations to the embeddings are required before computing cosine. This is because contextual embeddings (whether from masked language models or from autoregressive ones) have the property that the vectors for all words are extremely similar." (Trecho de Fine-Tuning and Masked Language Models)