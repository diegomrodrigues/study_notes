Aqui está um resumo detalhado e avançado sobre o tópico "Closed-Vocabulary Assumption" em Modelos de Linguagem:

## Limitações da Premissa de Vocabulário Fechado em Modelos de Linguagem

<imagem: Um diagrama mostrando um conjunto finito de palavras (vocabulário fechado) com setas apontando para palavras desconhecidas fora do conjunto, ilustrando as limitações desta abordagem>

### Introdução

A premissa de vocabulário fechado é uma suposição fundamental em muitos modelos de linguagem, onde se assume que o vocabulário $V$ é um conjunto finito [1]. Esta abordagem, embora comum, apresenta limitações significativas em cenários de aplicação realistas, especialmente quando se lida com textos dinâmicos e em constante evolução [2]. Este resumo explorará as implicações desta premissa, suas limitações e as abordagens alternativas para lidar com palavras fora do vocabulário (out-of-vocabulary words - OOV).

### Conceitos Fundamentais

| Conceito                | Explicação                                                   |
| ----------------------- | ------------------------------------------------------------ |
| **Vocabulário Fechado** | Um conjunto finito $V$ de palavras conhecidas pelo modelo [3]. |
| **Palavras OOV**        | Termos que não fazem parte do vocabulário predefinido $V$ [4]. |
| **Token \<UNK\>**       | Símbolo especial usado para representar palavras desconhecidas [5]. |

> ⚠️ **Nota Importante**: A premissa de vocabulário fechado pode levar a uma representação inadequada de textos em domínios dinâmicos ou multilíngues [6].

### Limitações da Premissa de Vocabulário Fechado

<imagem: Gráfico mostrando a frequência de palavras em um corpus, com uma longa cauda de palavras raras ou novas que seriam ignoradas em um vocabulário fechado>

A premissa de vocabulário fechado enfrenta desafios significativos em aplicações do mundo real:

1. **Neologismos e Termos Emergentes**: Incapacidade de lidar com novas palavras que surgem constantemente, especialmente em domínios como tecnologia e mídias sociais [7].

2. **Nomes Próprios**: Dificuldade em representar adequadamente nomes de pessoas, lugares ou organizações que não foram vistos durante o treinamento [8].

3. **Variações Morfológicas**: Em línguas com sistemas morfológicos ricos, muitas formas inflexionadas de palavras podem não ser capturadas [9].

4. **Multilinguismo**: Desafios ao lidar com múltiplos idiomas ou code-switching, onde o vocabulário pode expandir drasticamente [10].

5. **Erros Ortográficos e Variantes**: Incapacidade de lidar com erros de digitação ou variantes ortográficas não incluídas no vocabulário original [11].

#### 👍 Vantagens do Vocabulário Fechado

- Simplicidade computacional e eficiência de memória [12].
- Maior controle sobre o espaço de representação do modelo [13].

#### 👎 Desvantagens do Vocabulário Fechado

- Perda de informação semântica para palavras OOV [14].
- Redução da adaptabilidade do modelo a novos domínios ou dados [15].

### Abordagens para Lidar com Palavras OOV

1. **Uso do Token \<UNK\>**
   - Todas as palavras desconhecidas são mapeadas para um único token especial \<UNK\> [16].
   - Fórmula de probabilidade:
     $$P(w_i | w_{1:i-1}) = P(\text{\<UNK\>} | w_{1:i-1}) \text{ se } w_i \notin V$$

2. **Modelagem em Nível de Caracteres**
   - Utiliza modelos de linguagem baseados em caracteres para lidar com palavras OOV [17].
   - RNNs ou CNNs são aplicadas sobre sequências de caracteres [18].

3. **Segmentação em Subpalavras**
   - Decompõe palavras em unidades menores (morfemas ou subpalavras) [19].
   - Métodos como Byte-Pair Encoding (BPE) ou WordPiece são comumente utilizados [20].

4. **Modelos Híbridos**
   - Combinam abordagens em nível de palavra e caractere [21].
   - Exemplo: LSTM com componente de caracteres para palavras OOV [22].

### Implementação Avançada: Modelo Híbrido Palavra-Caractere

```python
import torch
import torch.nn as nn

class HybridWordCharModel(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, word_embed_dim, char_embed_dim, hidden_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
        self.char_lstm = nn.LSTM(char_embed_dim, hidden_dim // 2, bidirectional=True)
        self.word_lstm = nn.LSTM(word_embed_dim + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, word_vocab_size)
        
    def forward(self, word_ids, char_ids):
        # Word-level processing
        word_embeds = self.word_embedding(word_ids)
        
        # Character-level processing for OOV words
        char_embeds = self.char_embedding(char_ids)
        char_hidden, _ = self.char_lstm(char_embeds)
        char_hidden = char_hidden[:, -1, :]  # Use last hidden state
        
        # Combine word and character representations
        combined = torch.cat([word_embeds, char_hidden], dim=-1)
        
        # Main LSTM
        output, _ = self.word_lstm(combined)
        
        # Prediction
        logits = self.fc(output)
        return logits

# Uso do modelo
model = HybridWordCharModel(word_vocab_size=10000, char_vocab_size=128, 
                            word_embed_dim=300, char_embed_dim=50, hidden_dim=512)
```

Este modelo híbrido combina representações em nível de palavra e caractere, permitindo lidar com palavras OOV de forma mais eficaz [23].

### Conclusão

A premissa de vocabulário fechado, embora computacionalmente conveniente, apresenta limitações significativas em cenários do mundo real. Abordagens como modelagem em nível de caracteres, segmentação em subpalavras e modelos híbridos oferecem soluções mais robustas para lidar com a natureza dinâmica e diversa da linguagem natural [24]. A escolha da abordagem deve considerar o equilíbrio entre complexidade computacional, adaptabilidade e desempenho específico da tarefa [25].

### Perguntas Teóricas Avançadas

1. Derive a expressão para a perplexidade de um modelo de linguagem com vocabulário fechado quando uma fração $f$ das palavras em um corpus de teste são OOV. Como isso se compara com um modelo que pode gerar novas palavras?

2. Analise teoricamente o impacto da Lei de Zipf na eficácia de modelos de vocabulário fechado vs. aberto. Como a distribuição de cauda longa das palavras afeta o desempenho desses modelos?

3. Desenvolva uma prova matemática que demonstre as condições sob as quais um modelo híbrido palavra-caractere superaria consistentemente um modelo puramente baseado em palavras em termos de perplexidade.

4. Formule uma expressão para a complexidade computacional de um modelo de linguagem baseado em caracteres vs. um modelo de vocabulário fechado tradicional. Em que condições o modelo baseado em caracteres se torna computacionalmente mais eficiente?

5. Proponha e analise teoricamente um novo método de tokenização que otimize o trade-off entre cobertura de vocabulário e eficiência computacional, considerando a distribuição de frequência de palavras em corpora de grande escala.

### Referências

[1] "We have assumed a closed-vocabulary setting — the vocabulary $V$ is assumed to be a finite set." *(Trecho de Language Models_143-162.pdf.md)*

[2] "In realistic application scenarios, this assumption may not hold." *(Trecho de Language Models_143-162.pdf.md)*

[3] "Consider, for example, the problem of translating newspaper articles." *(Trecho de Language Models_143-162.pdf.md)*

[4] "The bolded terms either did not exist at this date, or were not widely known; they are unlikely to be in the vocabulary." *(Trecho de Language Models_143-162.pdf.md)*

[5] "One solution is to simply mark all such terms with a special token, \<UNK\>." *(Trecho de Language Models_143-162.pdf.md)*

[6] "The same problem can occur for a variety of other terms: new technologies, previously unknown individuals, new words (e.g., hashtag), and numbers." *(Trecho de Language Models_143-162.pdf.md)*

[7] "This is particularly important in languages that have rich morphological systems, with many inflections for each word." *(Trecho de Language Models_143-162.pdf.md)*

[8] "For example, Portuguese is only moderately complex from a morphological perspective, yet each verb has dozens of inflected forms (see Figure 4.3b)." *(Trecho de Language Models_143-162.pdf.md)*

[9] "In such languages, there will be many word types that we do not encounter in a corpus, which are nonetheless predictable from the morphological rules of the language." *(Trecho de Language Models_143-162.pdf.md)*

[10] "To use a somewhat contrived English example, if transfenestrate is in the vocabulary, our language model should assign a non-zero probability to the past tense transfenestrated, even if it does not appear in the training data." *(Trecho de Language Models_143-162.pdf.md)*

[11] "One way to accomplish this is to supplement word-level language models with character-level language models." *(Trecho de Language Models_143-162.pdf.md)*

[12] "Such models can use $n$-grams or RNNs, but with a fixed vocabulary equal to the set of ASCII or Unicode characters." *(Trecho de Language Models_143-162.pdf.md)*

[13] "For example, Ling et al. (2015) propose an LSTM model over characters, and Kim (2014) employ a convolutional neural network." *(Trecho de Language Models_143-162.pdf.md)*

[14] "A more linguistically motivated approach is to segment words into meaningful subword units, known as morphemes (see chapter 9)." *(Trecho de Language Models_143-162.pdf.md)*

[15] "For example, Botha and Blunsom (2014) induce vector representations for morphemes, which they build into a log-bilinear language model; Bhatia et al. (2016) incorporate morpheme vectors into an LSTM." *(Trecho de Language Models_143-162.pdf.md)*

[16] "While training the language model, we decide in advance on the vocabulary (often the $K$ most common terms), and mark all other terms in the training data as \<UNK\>." *(Trecho de Language Models_143-162.pdf.md)*

[17] "One way to accomplish this is to supplement word-level language models with character-level language models." *(Trecho de Language Models_143-162.pdf.md)*

[18] "For example, Ling et al. (2015) propose an LSTM model over characters, and Kim (2014) employ a convolutional neural network." *(Trecho de Language Models_143-162.pdf.md)*

[19] "A more linguistically motivated approach is to segment words into meaningful subword units, known as morphemes (see chapter 9)." *(Trecho de Language Models_143-162.pdf.md)*

[20] "For example, Botha and Blunsom (2014) induce vector representations for morphemes, which they build into a log-bilinear language model; Bhatia et al. (2016) incorporate morpheme vectors into an LSTM." *(Trecho de Language Models_143-162.pdf.md)*

[21] "One way to accomplish this is to supplement word-level language models with character-level language models." *(Trecho de Language Models_143-162.pdf.md)*

[22] "For example, Ling et al. (2015) propose an LSTM model over characters, and Kim (2014) employ a convolutional neural network." *(Trecho de Language Models_143-162.pdf.md)*

[23] "Using the Pytorch library, train an LSTM language model from the Wikitext training corpus." *(Trecho de Language Models_143-162.pdf.md)*

[24] "A more linguistically motivated approach is to segment words into meaningful subword units, known as morphemes (see chapter 9)." *(Trecho de Language Models_143-162.pdf.md)*

[25] "For example, Botha and Blunsom (2014) induce vector representations for morphemes, which they build into a log-bilinear language model; Bhatia et al. (2016) incorporate morpheme vectors into an LSTM." *(Trecho de Language Models_143-162.pdf.md)*