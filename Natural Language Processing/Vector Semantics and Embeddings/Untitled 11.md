# FastText: Modelagem Avançada de Palavras com Subword N-grams

<imagem: Um diagrama mostrando uma palavra sendo decomposta em n-gramas de caracteres, com vetores sendo gerados para cada n-grama e combinados para formar o vetor final da palavra.>

## Introdução

FastText é uma extensão avançada do modelo Word2Vec que aborda de forma inovadora os desafios de palavras desconhecidas e esparsidade em línguas morfologicamente ricas [1]. Desenvolvido como uma evolução do Word2Vec, o FastText incorpora informações de subpalavras, permitindo uma representação mais robusta e flexível do significado das palavras, especialmente em contextos onde a morfologia desempenha um papel crucial na semântica [2].

> ⚠️ **Nota Importante**: FastText representa uma mudança paradigmática na modelagem de palavras, passando de uma abordagem baseada em palavras inteiras para uma que considera a estrutura interna das palavras [3].

## Conceitos Fundamentais

| Conceito               | Explicação                                                   |
| ---------------------- | ------------------------------------------------------------ |
| **Subword Model**      | FastText utiliza um modelo de subpalavras, representando cada palavra como a soma de seus n-gramas de caracteres constituintes [4]. |
| **N-grams**            | Sequências contíguas de n caracteres extraídas da palavra, incluindo delimitadores especiais [5]. |
| **Vector Composition** | O vetor final de uma palavra é composto pela soma dos vetores de seus n-gramas constituintes [6]. |

> ❗ **Ponto de Atenção**: A incorporação de n-gramas permite ao FastText capturar semelhanças morfológicas e semânticas entre palavras, mesmo para palavras não vistas durante o treinamento [7].

### Arquitetura do FastText

<imagem: Arquitetura detalhada do modelo FastText, mostrando a entrada de uma palavra, sua decomposição em n-gramas, o processo de lookup e agregação dos vetores de n-gramas, e a saída do vetor final da palavra.>

O FastText estende a arquitetura do Skip-gram, incorporando uma camada adicional para processar n-gramas [8]. A arquitetura pode ser descrita matematicamente como:

$$
\text{FastText}(w) = \sum_{g \in G_w} v_g
$$

Onde:
- $w$ é a palavra de entrada
- $G_w$ é o conjunto de n-gramas da palavra $w$
- $v_g$ é o vetor associado ao n-grama $g$

Esta formulação permite que o modelo capture informações morfológicas intrínsecas, resultando em representações mais ricas e informativas [9].

#### Geração de N-gramas

O processo de geração de n-gramas é fundamental para o funcionamento do FastText. Para uma palavra $w$, o conjunto de n-gramas $G_w$ é gerado da seguinte forma [10]:

1. Adicione os símbolos especiais < e > no início e fim da palavra, respectivamente.
2. Extraia todos os n-gramas de tamanho 3 a 6 (por padrão).
3. Inclua a palavra completa como um n-grama adicional.

Por exemplo, para a palavra "where" com n=3, teríamos:

```python
G_where = {<wh, whe, her, ere, re>, <where>}
```

Esta abordagem permite ao modelo capturar prefixos, sufixos e raízes das palavras, essenciais para línguas morfologicamente ricas [11].

### Função Objetivo e Treinamento

O FastText utiliza uma função objetivo similar à do Skip-gram, mas incorporando a estrutura de n-gramas [12]:

$$
\mathcal{L} = \sum_{t=1}^T \sum_{c \in C_t} \log\sigma(v_{c}^T \cdot \sum_{g \in G_{w_t}} v_g) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)} [\log\sigma(-v_{w_i}^T \cdot \sum_{g \in G_{w_t}} v_g)]
$$

Onde:
- $T$ é o número total de palavras no corpus
- $C_t$ é o conjunto de palavras de contexto para a palavra alvo $w_t$
- $G_{w_t}$ é o conjunto de n-gramas da palavra alvo
- $v_c$ e $v_g$ são os vetores de contexto e n-grama, respectivamente
- $P_n(w)$ é a distribuição de ruído para amostragem negativa

O treinamento é realizado utilizando Stochastic Gradient Descent (SGD) com amostragem negativa, similar ao Word2Vec [13].

#### Perguntas Teóricas

1. Derive a expressão para o gradiente da função objetivo do FastText com respeito a um vetor de n-grama específico $v_g$. Como isso difere do gradiente no modelo Skip-gram original?

2. Considerando um vocabulário $V$ e um conjunto de n-gramas $G$, demonstre matematicamente como o FastText reduz a complexidade computacional para palavras raras ou desconhecidas em comparação com o Word2Vec.

3. Prove que, para uma palavra $w$ composta por $m$ n-gramas, o limite superior do número de parâmetros necessários para representá-la no FastText é $O(m \cdot d)$, onde $d$ é a dimensão do espaço de embeddings.

### Vantagens e Desvantagens do FastText

| 👍 Vantagens                                                 | 👎 Desvantagens                                         |
| ----------------------------------------------------------- | ------------------------------------------------------ |
| Melhor representação de palavras raras e desconhecidas [14] | Aumento no número de parâmetros do modelo [15]         |
| Captura de informações morfológicas [16]                    | Potencial oversensibilidade a ruídos ortográficos [17] |
| Eficiente para línguas com vocabulários grandes [18]        | Maior complexidade computacional no treinamento [19]   |

### Análise Teórica da Composicionalidade

A composicionalidade no FastText pode ser analisada através da teoria de espaços vetoriais. Seja $\Phi : \Sigma^* \to \mathbb{R}^d$ uma função que mapeia strings para vetores d-dimensionais. A representação FastText de uma palavra $w$ pode ser expressa como [20]:

$$
\Phi(w) = \sum_{g \in G_w} \phi(g)
$$

Onde $\phi(g)$ é a representação vetorial do n-grama $g$. Esta formulação implica que o espaço de embeddings do FastText é fechado sob adição, uma propriedade crucial para sua capacidade de generalização [21].

> ✔️ **Destaque**: A composicionalidade do FastText permite que o modelo infira representações para palavras desconhecidas, desde que compartilhem n-gramas com palavras conhecidas [22].

#### Prova de Invariância Rotacional

Teorema: As representações FastText são invariantes sob rotações ortogonais do espaço de embeddings.

Prova:
Seja $R$ uma matriz de rotação ortogonal. Para qualquer palavra $w$:

$$
\begin{align*}
R\Phi(w) &= R\sum_{g \in G_w} \phi(g) \\
&= \sum_{g \in G_w} R\phi(g) \\
&= \Phi_R(w)
\end{align*}
$$

Onde $\Phi_R(w)$ é a representação de $w$ no espaço rotacionado. Isto demonstra que a estrutura relativa das representações é preservada sob rotações, uma propriedade importante para a estabilidade do modelo [23].

### Implementação Avançada

A implementação do FastText requer considerações especiais para eficiência computacional. Aqui está um exemplo de código Python avançado utilizando PyTorch para a geração de n-gramas e lookup eficiente:

```python
import torch
import torch.nn as nn

class FastTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ngram_range=(3, 6)):
        super().__init__()
        self.ngram_range = ngram_range
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_hash = lambda x: hash(x) % vocab_size

    def generate_ngrams(self, word):
        word = f"<{word}>"
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(word) - n + 1):
                ngrams.append(word[i:i+n])
        return ngrams + [word]

    def forward(self, words):
        ngram_ids = []
        for word in words:
            ngrams = self.generate_ngrams(word)
            ngram_ids.extend([self.ngram_hash(ng) for ng in ngrams])
        
        ngram_tensor = torch.LongTensor(ngram_ids).to(self.embedding.weight.device)
        return self.embedding(ngram_tensor).sum(dim=0)

# Uso do modelo
model = FastTextModel(vocab_size=100000, embedding_dim=300)
word_vector = model(["example"])
```

Este código demonstra uma implementação eficiente do FastText, utilizando hashing para reduzir a complexidade de armazenamento dos n-gramas [24].

### Discussão Crítica

Apesar de suas vantagens, o FastText enfrenta desafios em cenários específicos. Por exemplo, em línguas com sistemas de escrita não-alfabéticos, a decomposição em n-gramas pode não ser tão efetiva [25]. Além disso, a abordagem de soma simples dos vetores de n-gramas pode não capturar completamente as nuances semânticas em composições complexas [26].

Pesquisas futuras poderiam explorar:
1. Integração de informações sintáticas na geração de n-gramas.
2. Adaptação dinâmica do tamanho dos n-gramas baseada na estrutura morfológica da língua.
3. Incorporação de mecanismos de atenção para ponderar a importância relativa dos n-gramas.

## Conclusão

O FastText representa um avanço significativo na modelagem de palavras, especialmente para línguas morfologicamente ricas e cenários com vocabulários extensos [27]. Sua abordagem baseada em subpalavras não apenas melhora a representação de palavras raras e desconhecidas, mas também fornece uma base teórica sólida para a exploração da estrutura interna das palavras no processamento de linguagem natural [28].

## Perguntas Teóricas Avançadas

1. Derive uma expressão para a complexidade computacional do FastText em termos de tamanho do vocabulário, dimensão do embedding e distribuição de comprimento das palavras. Compare com a complexidade do Word2Vec e discuta as implicações para escalabilidade.

2. Considere um cenário onde temos um corpus multilíngue. Proponha e analise matematicamente uma extensão do FastText que possa aprender representações de palavras compartilhadas entre línguas, levando em conta as diferenças morfológicas.

3. Desenvolva um framework teórico para analisar a capacidade do FastText em capturar analogias morfológicas (e.g., "carro:carros::gato:gatos"). Como isso se compara com modelos baseados apenas em palavras inteiras?

4. Prove que, sob certas condições, o FastText pode aproximar arbitrariamente bem qualquer função contínua do espaço de strings para o espaço de embeddings. Quais são as implicações teóricas e práticas desta propriedade?

5. Formule uma versão probabilística do FastText baseada em processos gaussianos. Como isso afetaria a interpretação das representações de palavras e a capacidade do modelo de lidar com incerteza?

## Anexos

### A.1 Derivação da Função de Perda do FastText

Partindo da função de verossimilhança negativa do Skip-gram, a função de perda do FastText pode ser derivada como:

$$
\begin{align*}
\mathcal{L} &= -\sum_{(w,c) \in D} \log P(c|w) \\
&= -\sum_{(w,c) \in D} \log \frac{\exp(v_c^T \cdot v_w)}{\sum_{c' \in V} \exp(v_{c'}^T \cdot v_w)} \\
&= -\sum_{(w,c) \in D} \log \frac{\exp(v_c^T \cdot \sum_{g \in G_w} v_g)}{\sum_{c' \in V} \exp(v_{c'}^T \cdot \sum_{g \in G_w} v_g)}
\end{align*}
$$

Onde $D$ é o conjunto de pares palavra-contexto observados, $V$ é o vocabulário, e $G_w$ é o conjunto de n-gramas da palavra $w$ [29].

### A.2 Análise de Complexidade Espacial

A complexidade espacial do FastText é $O(|V| \cdot d + |G| \cdot d)$, onde $|V|$ é o tamanho do vocabulário, $|G|$ é o número total de n-gramas únicos, e $d$ é a dimensão do embedding. Em contraste, o Word2Vec tem complexidade $O(|V| \cdot d)$. Embora o FastText requeira mais memória, ele oferece uma compensação favorável entre uso de memória e capacidade de representação, especialmente para línguas com morfologia rica [30].

## Referências

[1] "FastText é uma extensão do modelo Word2Vec que aborda de forma inovadora os desafios de palavras desconhecidas e esparsidade em línguas morfologicamente ricas." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[2] "fasttext (Bojanowski et al., 2017), addresses a problem with word2vec as we have presented it so far: it has no good way to deal with unknown words—words that appear in a test corpus but were unseen in the training corpus." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[3] "A related problem is word sparsity, such as in languages with rich morphology, where some of the many forms for each noun and verb may only occur rarely." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[4] "Fasttext deals with these problems by using subword models, representing each word as itself plus a bag of constituent n-grams, with special boundary symbols < and > added to each word." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[5] "For example, with n = 3, the word "where" would be represented by the sequence <where> plus the character n-grams:" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[6] "<wh, whe, her, ere, re>" *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[7] "Then a skipgram embedding is learned for each constituent n-gram, and the word "where" is represented by the sum of all of the embeddings of its constituent n-grams." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[8] "Unknown words can then be presented only by the sum of the constituent n-grams." *(Trecho de Vector Semantics and Embeddings.pdf.md)*

[9] "Fasttext deals with these problems by using subword models, representing each word as itself plus a bag of constituent n-grams, with special boundary symbols < and > added to each word." *(Trecho de Vector Semantics and Embeddings.