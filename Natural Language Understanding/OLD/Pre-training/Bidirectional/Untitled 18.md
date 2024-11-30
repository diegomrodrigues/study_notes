## Dados de Treinamento e Tamanho: Visão Geral dos Datasets Usados para Treinar Codificadores Bidirecionais

<image: Uma representação visual de diferentes fontes de dados (como Wikipedia, textos da web, recursos multilíngues) fluindo para um modelo de codificador bidirecional, com ícones representando a diversidade linguística e o volume de dados.>

### Introdução

Os codificadores bidirecionais, como o BERT (Bidirectional Encoder Representations from Transformers) e seus descendentes, revolucionaram o processamento de linguagem natural ao fornecer representações contextualizadas profundas de texto. A eficácia desses modelos está intrinsecamente ligada à qualidade e quantidade dos dados de treinamento utilizados [1]. Este resumo fornece uma visão geral abrangente dos datasets empregados no treinamento de codificadores bidirecionais, explorando suas características, tamanhos e impactos no desempenho dos modelos.

### Conceitos Fundamentais

| Conceito                     | Explicação                                                   |
| ---------------------------- | ------------------------------------------------------------ |
| **Codificador Bidirecional** | Modelo de linguagem que processa o contexto em ambas as direções (esquerda para direita e direita para esquerda) para gerar representações contextualizadas de tokens. [1] |
| **Web Text**                 | Corpus de texto coletado da internet, geralmente filtrado para qualidade e diversidade. [2] |
| **Wikipedia**                | Enciclopédia online multilíngue, frequentemente usada como fonte de texto de alta qualidade para treinamento de modelos de linguagem. [2] |
| **Recursos Multilíngues**    | Datasets que abrangem múltiplos idiomas, permitindo o treinamento de modelos com capacidades linguísticas diversas. [3] |

> ⚠️ **Nota Importante**: A qualidade e diversidade dos dados de treinamento são cruciais para o desempenho e a generalização dos codificadores bidirecionais.

### Datasets Principais para Treinamento

#### Wikipedia e BooksCorpus

O BERT original foi treinado em aproximadamente 3,3 bilhões de palavras, combinando o English Wikipedia com o BooksCorpus [1]. 

> ✔️ **Ponto de Destaque**: O uso de Wikipedia proporciona um corpus de texto de alta qualidade e bem estruturado, cobrindo uma ampla gama de tópicos.

No entanto, é importante notar que o BooksCorpus não é mais utilizado devido a questões de propriedade intelectual [1]. Esta mudança destaca a importância de considerar não apenas a qualidade dos dados, mas também suas implicações legais e éticas no treinamento de modelos de linguagem.

#### Web Text e Common Crawl

Modelos mais recentes de linguagem mascarada expandiram significativamente o tamanho e a diversidade dos dados de treinamento, incorporando grandes volumes de texto da web. Por exemplo:

- O XLM-R foi treinado em aproximadamente 300 bilhões de tokens em 100 idiomas, utilizando dados do Common Crawl [3].

> ❗ **Ponto de Atenção**: O uso de dados da web requer filtragem cuidadosa para remover conteúdo de baixa qualidade ou inadequado.

A utilização de Common Crawl e outras fontes de texto da web permite aos modelos acesso a um corpus massivo e diversificado, refletindo o uso contemporâneo da linguagem em diversos contextos e domínios.

### Estratégias de Amostragem para Datasets Multilíngues

Para modelos multilíngues, a estratégia de amostragem dos dados de treinamento é crucial para equilibrar a representação de diferentes idiomas. O XLM-R utiliza uma abordagem sofisticada para ajustar as probabilidades de seleção de sentenças de cada idioma [3]:

$$
q_i = \frac{p_i^\alpha}{\sum_{j=1}^N p_j^\alpha}, \text{ onde } p_i = \frac{n_i}{\sum_{k=1}^N n_k}
$$

Onde:
- $q_i$ é a probabilidade ajustada de selecionar uma sentença do idioma $i$
- $p_i$ é a proporção original de sentenças do idioma $i$ no corpus
- $\alpha$ é um parâmetro de ajuste (tipicamente 0.3)
- $N$ é o número total de idiomas

Esta fórmula permite dar maior peso a idiomas menos representados, mitigando o viés para idiomas com maior volume de dados disponíveis, como o inglês.

> 💡 **Insight**: Um valor de $\alpha = 0.3$ foi empiricamente determinado como eficaz para melhorar a inclusão de idiomas raros na tokenização, resultando em melhor desempenho multilíngue geral [3].

#### Questões Técnicas/Teóricas

1. Como a escolha de $\alpha = 0.3$ na fórmula de amostragem afeta a representação de idiomas com poucos recursos em comparação com idiomas dominantes como o inglês?
2. Quais são as implicações práticas de usar Common Crawl como fonte de dados em termos de qualidade do modelo e viés potencial?

### Tokenização e Vocabulário

A escolha do método de tokenização e o tamanho do vocabulário são aspectos críticos que influenciam diretamente a capacidade do modelo de lidar com diferentes idiomas e domínios.

#### Tokenização Subword

Modelos como BERT e XLM-RoBERTa utilizam algoritmos de tokenização subword, que permitem lidar eficientemente com palavras desconhecidas e morfologia complexa:

- BERT: Utiliza o algoritmo WordPiece com um vocabulário de 30.000 tokens [1].
- XLM-RoBERTa: Emprega o algoritmo SentencePiece Unigram LM com um vocabulário de 250.000 tokens [3].

> ✔️ **Ponto de Destaque**: A tokenização subword é crucial para a eficácia de modelos multilíngues, permitindo a decomposição de palavras em unidades menores compartilhadas entre idiomas.

A diferença significativa no tamanho do vocabulário entre BERT e XLM-RoBERTa reflete a necessidade de um vocabulário maior para cobrir efetivamente múltiplos idiomas.

### Impacto do Tamanho dos Dados no Desempenho do Modelo

O volume de dados de treinamento tem um impacto direto na qualidade e capacidade de generalização dos modelos de linguagem. Comparando BERT e XLM-RoBERTa:

| Modelo      | Tamanho dos Dados       | Número de Idiomas | Tamanho do Modelo |
| ----------- | ----------------------- | ----------------- | ----------------- |
| BERT        | 3,3 bilhões de palavras | 1 (Inglês)        | ~100M parâmetros  |
| XLM-RoBERTa | 300 bilhões de tokens   | 100               | ~550M parâmetros  |

Esta comparação ilustra a tendência de aumentar drasticamente o volume de dados e a complexidade do modelo para melhorar o desempenho e a cobertura linguística.

> ❗ **Ponto de Atenção**: O aumento no tamanho dos dados e do modelo traz desafios computacionais significativos, exigindo estratégias eficientes de treinamento e infraestrutura.

### Desafios e Considerações

#### Maldição da Multilingualidade

À medida que o número de idiomas cobertos por um modelo aumenta, observa-se um fenômeno chamado "maldição da multilingualidade" [3]:

- O desempenho em cada idioma individual tende a degradar em comparação com modelos treinados em menos idiomas.
- Isso cria um trade-off entre cobertura linguística e desempenho por idioma.

#### Viés Linguístico

Modelos multilíngues podem exibir um "sotaque" em idiomas com menos recursos [3]:

- Estruturas gramaticais de idiomas com mais recursos (frequentemente o inglês) podem "sangrar" para representações de idiomas com menos recursos.
- Isso resulta em representações para idiomas com poucos recursos que são ligeiramente mais "inglesas" do que o ideal.

> ⚠️ **Nota Importante**: O viés linguístico pode afetar a equidade e a precisão do modelo em aplicações multilingues, exigindo estratégias de mitigação cuidadosas.

#### Questões Técnicas/Teóricas

1. Como podemos quantificar o trade-off entre a cobertura linguística e o desempenho por idioma em modelos multilíngues? Proponha uma métrica ou metodologia.
2. Considerando o fenômeno de "sotaque" em modelos multilíngues, quais estratégias poderiam ser implementadas durante o treinamento para mitigar este efeito?

### Estratégias de Treinamento

O treinamento eficaz de codificadores bidirecionais em grandes volumes de dados multilíngues requer estratégias sofisticadas:

1. **Mascaramento de Linguagem (MLM)**:
   - 15% dos tokens de entrada são selecionados para aprendizado [1].
   - Destes, 80% são substituídos por [MASK], 10% por tokens aleatórios, e 10% permanecem inalterados.

2. **Next Sentence Prediction (NSP)**:
   - Utilizado no BERT original, mas abandonado em alguns modelos subsequentes como o RoBERTa [2].
   - 50% dos pares de treinamento são pares de sentenças adjacentes reais, 50% são pares aleatórios.

3. **Empacotamento de Documentos**:
   - RoBERTa e modelos similares empacotam sentenças contíguas até atingir o limite de tokens (geralmente 512) [2].
   - Um token separador extra é adicionado entre documentos.

4. **Tamanhos de Lote Grandes**:
   - Lotes entre 8K e 32K tokens são comumente usados [2].
   - Facilita o treinamento eficiente em grandes volumes de dados.

> 💡 **Insight**: O abandono do NSP em modelos como RoBERTa sugere que a tarefa pode não ser crucial para o desempenho em downstream tasks, simplificando o processo de treinamento.

A implementação dessas estratégias em PyTorch para um modelo simplificado pode ser esboçada da seguinte forma:

```python
import torch
import torch.nn as nn

class BidirectionalEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            num_layers=num_layers
        )
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.mlm_head(x)

def mlm_loss(predictions, targets, mask):
    loss = nn.CrossEntropyLoss(ignore_index=-100)
    return loss(predictions.view(-1, predictions.size(-1)), targets.view(-1))

# Exemplo de uso
model = BidirectionalEncoder(vocab_size=30000, hidden_size=768, num_layers=12)
optimizer = torch.optim.Adam(model.parameters())

# Assume-se que input_ids, masked_lm_labels e attention_mask são tensores apropriados
for epoch in range(num_epochs):
    for input_ids, masked_lm_labels, attention_mask in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids, mask=attention_mask)
        loss = mlm_loss(outputs, masked_lm_labels, attention_mask)
        loss.backward()
        optimizer.step()
```

Este código simplificado ilustra a estrutura básica de um codificador bidirecional e como implementar o treinamento MLM. Na prática, modelos como BERT e XLM-RoBERTa são significativamente mais complexos e otimizados.

### Conclusão

O treinamento de codificadores bidirecionais envolve a utilização de datasets massivos e diversos, com estratégias sofisticadas para lidar com múltiplos idiomas e domínios. A evolução de modelos como BERT para XLM-RoBERTa demonstra uma tendência clara de aumento no volume de dados, complexidade do modelo e diversidade linguística [1][2][3].

Enquanto essa abordagem tem produzido modelos com capacidades impressionantes, ela também apresenta desafios significativos, incluindo a maldição da multilingualidade e potenciais vieses linguísticos. A pesquisa contínua nesta área provavelmente se concentrará em estratégias para mitigar esses desafios, possivelmente através de técnicas de treinamento mais avançadas e seleção de dados mais sofisticada.

O futuro dos codificadores bidirecionais pode envolver um equilíbrio mais refinado entre a amplitude da cobertura linguística e a profundidade do entendimento em cada idioma, potencialmente levando a modelos que são verdadeiramente multilingues sem sacrificar o desempenho em idiomas individuais.

### Questões Avançadas

1. Considerando a maldição da multilingualidade, proponha uma arquitetura ou método de treinamento que possa potencialmente mitigar este efeito sem aumentar drasticamente o tamanho do modelo ou dos dados de treinamento.

2. Como você abordaria o problema de avaliar a qualidade e representatividade de um dataset multilíngue massivo como o usado no XLM-RoBERTa? Quais métricas e metodologias você proporia para garantir uma cobertura adequada e balanceada entre idiomas e domínios?

3. Dado o fenômeno de "sotaque" observado em modelos multilíngues, onde estruturas de idiomas dominantes influenciam a representação de idiomas com menos recursos, como você projetaria um experimento para quantificar e visualizar este efeito? Que insights isso poderia fornecer para melhorar o treinamento de futuros modelos multilíngues?

### Referências

[1] "BERT and other early transformer-based language models were trained on about 3.3 billion words (a combination of English Wikipedia and a corpus of book texts called BooksCorpus (Zhu et al., 2015) that is no longer used for intellectual property reasons)." (Trecho de Fine-Tuning and Masked Language Models)

[2] "Modern masked language models are now trained on much larger datasets of web text, filtered a bit, and augmented by higher-quality data like Wikipedia, the same as those we discussed for the causal large language models of Chapter 10." (Trecho de Fine-Tuning and Masked Language Models)

[3] "Multilingual models similarly use webtext and multilingual Wikipedia. For example the XLM-R model was trained on about 300 billion tokens in 100 languages, taken from the web via Common Crawl (https://commoncrawl.org/)." (Trecho de Fine-Tuning and Masked Language Models)

[4] "To train the original BERT models, pairs of text segments were selected from the training corpus according to the next sentence prediction 50/50 scheme. Pairs were sampled so that their combined length was less than the 512 token input." (Trecho de Fine-Tuning and Masked Language Models)

[5] "Some models, like the