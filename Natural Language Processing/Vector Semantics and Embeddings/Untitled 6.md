# Self-Supervision: Aprendizagem a partir de Texto não Rotulado

<imagem: Uma representação visual de palavras em um espaço vetorial, com setas indicando como palavras semanticamente relacionadas se agrupam. No centro, destaque para o processo de aprendizagem do Word2Vec, mostrando como ele extrai informações de contexto do texto não rotulado.>

---

## Introdução

A *self-supervision* é uma abordagem revolucionária no campo do Processamento de Linguagem Natural (PLN) que permite o aprendizado de representações semânticas ricas a partir de dados de texto não rotulados. Este conceito é fundamental para entender como modelos como o **Word2Vec** são capazes de capturar relações semânticas complexas entre palavras sem a necessidade de rotulação manual extensa [1].

O paradigma da *self-supervision* representa uma mudança significativa na forma como abordamos o aprendizado de máquina em PLN. Em vez de depender de conjuntos de dados laboriosamente rotulados, esta técnica aproveita a estrutura inerente da linguagem natural para criar sinais de supervisão implícitos [2]. Este método não apenas reduz drasticamente a necessidade de intervenção humana no processo de treinamento, mas também permite a utilização de vastos corpora de texto não estruturado disponíveis na internet.

⚠️ **Nota Importante**: A *self-supervision* não é apenas uma técnica de treinamento, mas uma mudança fundamental na filosofia do aprendizado de máquina, permitindo que os modelos "compreendam" a linguagem de uma forma mais análoga à aprendizagem humana [3].

---

## Conceitos Fundamentais

| **Conceito**         | **Explicação**                                               |
| -------------------- | ------------------------------------------------------------ |
| **Self-Supervision** | Técnica de aprendizado onde o modelo gera seus próprios rótulos a partir dos dados de entrada, eliminando a necessidade de anotação manual [4]. |
| **Embedding**        | Representação vetorial densa de palavras em um espaço multidimensional, capturando relações semânticas [5]. |
| **Contexto**         | Conjunto de palavras que cercam uma palavra-alvo, usado para prever ou entender o significado da palavra [6]. |

❗ **Ponto de Atenção**: A qualidade dos *embeddings* gerados por modelos *self-supervisionados* depende criticamente da riqueza e diversidade do corpus de treinamento [7].

---

## Fundamentos Teóricos da Self-Supervision

A *self-supervision* no contexto do PLN baseia-se no princípio da **Hipótese Distribucional**, formulada por linguistas como Harris (1954) e Firth (1957). Esta hipótese postula que palavras que ocorrem em contextos similares tendem a ter significados semelhantes [8]. Matematicamente, podemos expressar esta ideia considerando a similaridade semântica entre duas palavras como proporcional à similaridade entre suas distribuições de contexto.

==Uma forma de quantificar essa similaridade é utilizando medidas de divergência entre distribuições de probabilidade, como a **divergência de Kullback-Leibler (KL)**. Para palavras $w_i$ e $w_j$, podemos definir:==

$$
\text{Similaridade}(w_i, w_j) \propto -D_{\text{KL}}(P(C|w_i) \parallel P(C|w_j))
$$

Onde:

- $P(C|w_i)$ é a distribuição de probabilidade do contexto $C$ dado a palavra $w_i$.
- $D_{\text{KL}}(P \parallel Q)$ é a divergência KL entre as distribuições $P$ e $Q$.

Essa formulação captura a ideia de que, se duas palavras têm distribuições de contexto similares, elas devem estar próximas no espaço semântico [9].

---

## Revisão da Literatura

A evolução da *self-supervision* em PLN pode ser traçada desde os trabalhos seminais de Bengio et al. (2003) e Collobert et al. (2011), que demonstraram que redes neurais poderiam aprender representações úteis de palavras como parte de tarefas de predição [10]. O avanço crucial veio com Mikolov et al. (2013a, 2013b), que introduziram o **Word2Vec**, simplificando o processo de treinamento e tornando possível o aprendizado eficiente de *embeddings* a partir de grandes corpora de texto [11].

✔️ **Destaque**: O **Word2Vec** não apenas tornou o treinamento de *embeddings* mais eficiente, mas também revelou propriedades algébricas surpreendentes nas representações aprendidas, como a capacidade de capturar analogias semânticas [12].

---

## Aplicações Avançadas

A *self-supervision* tem encontrado aplicações além do PLN tradicional:

- **Visão Computacional**: Técnicas inspiradas no **Word2Vec** têm sido adaptadas para aprender representações de imagens sem rótulos [13].
- **Sistemas de Recomendação**: *Embeddings* de usuários e itens podem ser aprendidos de forma *self-supervisionada* a partir de históricos de interações [14].
- **Bioinformática**: Sequências de proteínas podem ser representadas como "frases", permitindo a aplicação de técnicas de PLN para prever estruturas e funções [15].

---

## O Modelo Word2Vec

O **Word2Vec**, introduzido por Mikolov et al. (2013), é um exemplo paradigmático de aprendizado *self-supervisionado* em PLN. Existem duas arquiteturas principais: **Skip-gram** e **Continuous Bag of Words (CBOW)** [16].

### Skip-gram com Negative Sampling (SGNS)

O SGNS é uma variante do Skip-gram que se tornou particularmente influente devido à sua eficiência computacional e qualidade dos *embeddings* produzidos [17].

A função objetivo do SGNS pode ser expressa como:

$$
J = \sum_{(w, c) \in D} \left[ \log \sigma(\mathbf{v}_w^\top \mathbf{v}_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_c)] \right]
$$

Onde:

- $D$ é o conjunto de pares palavra-contexto observados.
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ é a função sigmoide.
- $\mathbf{v}_w$ e $\mathbf{v}_c$ são os vetores da palavra e do contexto, respectivamente.
- $P_n(w)$ é a distribuição de amostragem negativa.
- $K$ é o número de amostras negativas.

Esta função maximiza a similaridade entre palavras e seus contextos verdadeiros enquanto minimiza a similaridade com contextos negativos amostrados aleatoriamente [18].

💡 **Insights**: A *negative sampling* é crucial para a eficiência do SGNS, permitindo que o modelo aprenda distinguindo entre contextos reais e falsos, em vez de tentar prever explicitamente todo o vocabulário [19].

---

## Perguntas Teóricas

1. **Derive a expressão para o gradiente da função de perda do SGNS com respeito aos vetores de palavra e contexto. Como esta formulação facilita o aprendizado eficiente de *embeddings*?**

2. **Considerando a hipótese distribucional, como você explicaria matematicamente a capacidade do **Word2Vec** de capturar relações semânticas como analogias (e.g., rei - homem + mulher ≈ rainha)?**

3. **Analise teoricamente como a escolha do tamanho da janela de contexto afeta as propriedades dos *embeddings* aprendidos pelo **Word2Vec**. Como isso se relaciona com a captura de diferentes tipos de relações semânticas (sintagmáticas vs. paradigmáticas)?**

---

## Propriedades Semânticas dos Embeddings

Os *embeddings* aprendidos através de métodos *self-supervisionados* como o **Word2Vec** exibem propriedades semânticas fascinantes que vão além da simples similaridade de palavras [20].

### Analogias e Similaridade Relacional

Uma das descobertas mais intrigantes é a capacidade dos *embeddings* de capturar relações analógicas. Isso é frequentemente demonstrado através do "modelo de paralelogramo" [21]:

$$
\mathbf{v}_{\text{rei}} - \mathbf{v}_{\text{homem}} + \mathbf{v}_{\text{mulher}} \approx \mathbf{v}_{\text{rainha}}
$$

Esta propriedade sugere que os *embeddings* codificam não apenas similaridades, mas também diferenças semânticas de maneira estruturada [22].

### Visualização de Embeddings

Para entender melhor as estruturas semânticas capturadas, técnicas de redução de dimensionalidade como **t-SNE** são frequentemente aplicadas [23]. Isso permite visualizar como palavras semanticamente relacionadas se agrupam no espaço de *embeddings*:

<imagem: Um gráfico 2D mostrando clusters de palavras após aplicação de t-SNE em embeddings Word2Vec. Destaque para grupos semânticos como países, profissões e conceitos abstratos.>

---

## Discussão Crítica

Apesar dos sucessos impressionantes, é crucial reconhecer as limitações dos *embeddings* estáticos como os produzidos pelo **Word2Vec**:

- **Polissemia**: Palavras com múltiplos significados são representadas por um único vetor, perdendo nuances contextuais [24].
- **Viés**: Os *embeddings* podem perpetuar e amplificar preconceitos presentes nos dados de treinamento [25].
- **Interpretabilidade**: As dimensões individuais dos vetores de *embedding* geralmente carecem de interpretação semântica clara [26].

⚠️ **Desafio Futuro**: Desenvolver métodos de *self-supervision* que possam abordar estas limitações, possivelmente incorporando informações de estrutura sintática ou conhecimento de mundo externo, permanece uma área ativa de pesquisa [27].

---

## Conclusão

A *self-supervision*, exemplificada pelo **Word2Vec**, representa um avanço fundamental na forma como abordamos o aprendizado de representações semânticas em PLN. Ao aproveitar a estrutura inerente da linguagem como sinal de supervisão, estes métodos permitem o aprendizado de *embeddings* ricos e úteis a partir de vastos corpora de texto não rotulado [28].

A capacidade de capturar relações semânticas complexas sem supervisão explícita não apenas reduziu drasticamente a necessidade de anotação manual, mas também abriu novas possibilidades para compreender e modelar a linguagem de formas mais análogas ao aprendizado humano [29].

À medida que o campo evolui, podemos esperar que os princípios de *self-supervision* continuem a desempenhar um papel crucial no desenvolvimento de modelos de linguagem cada vez mais sofisticados e capazes [30].

---

## Perguntas Teóricas Avançadas

1. **Formule matematicamente como o princípio de máxima verossimilhança se aplica ao treinamento do modelo Skip-gram. Como isso se relaciona com a função de perda do SGNS? Derive a conexão entre estas formulações.**

2. **Considere a propriedade de aditividade composicional dos *embeddings* **Word2Vec** (e.g., $\mathbf{v}_{\text{Paris}} - \mathbf{v}_{\text{França}} + \mathbf{v}_{\text{Alemanha}} \approx \mathbf{v}_{\text{Berlim}}$). Proponha e analise um framework teórico que possa explicar por que os *embeddings* aprendidos de forma *self-supervisionada* exibem esta propriedade.**

3. **O Teorema de Johnson-Lindenstrauss sugere que projeções aleatórias podem preservar distâncias relativas em espaços de alta dimensão. Como isso se relaciona com a eficácia dos *embeddings* de baixa dimensão aprendidos pelo **Word2Vec**? Desenvolva uma prova ou argumento formal para esta relação.**

4. **Derive uma expressão para a complexidade amostral do **Word2Vec** em termos do tamanho do vocabulário e da dimensão dos *embeddings*. Como isso se compara com métodos baseados em fatoração de matriz para aprender *embeddings*?**

5. **Analise teoricamente como a distribuição de frequência das palavras no corpus de treinamento afeta a qualidade dos *embeddings* aprendidos. Proponha e justifique matematicamente uma estratégia de amostragem ou ponderação que possa mitigar possíveis vieses introduzidos por esta distribuição.**

---

## Anexos

### A.1 Prova da Convergência do SGD para SGNS

Aqui, apresentamos uma prova esboçada da convergência do **Stochastic Gradient Descent (SGD)** para o modelo Skip-gram com Negative Sampling (SGNS):

Seja $\theta$ o conjunto de parâmetros do modelo (os *embeddings* de palavras e contextos). A função objetivo do SGNS pode ser escrita como:

$$
J(\theta) = \sum_{(w, c) \in D} \left[ \log \sigma(\mathbf{v}_w^\top \mathbf{v}_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} \left[ \log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_c) \right] \right]
$$

Para provar a convergência, precisamos mostrar que:

1. **Convexidade Local**: Embora a função objetivo não seja globalmente convexa, podemos analisar sua convexidade local em torno de mínimos locais.

2. **Gradiente Estocástico Não Viciado**: O gradiente estocástico é um estimador não enviesado do gradiente verdadeiro da função de perda.

3. **Taxa de Aprendizado**: A taxa de aprendizado $\eta_t$ satisfaz as condições de Robbins-Monro, ou seja, $\sum_{t} \eta_t = \infty$ e $\sum_{t} \eta_t^2 < \infty$.

Sob estas condições, o SGD converge para um ponto crítico da função de perda [31].

---

## Referências

[1] "A self-supervision representa uma mudança significativa na forma como abordamos o aprendizado de máquina em PLN."

[2] "Em vez de depender de conjuntos de dados laboriosamente rotulados, esta técnica aproveita a estrutura inerente da linguagem natural para criar sinais de supervisão implícitos."

[3] "A self-supervision não é apenas uma técnica de treinamento, mas uma mudança fundamental na filosofia do aprendizado de máquina, permitindo que os modelos 'compreendam' a linguagem de uma forma mais análoga à aprendizagem humana."

[4] "Self-Supervision: Técnica de aprendizado onde o modelo gera seus próprios rótulos a partir dos dados de entrada, eliminando a necessidade de anotação manual."

[5] "Embedding: Representação vetorial densa de palavras em um espaço multidimensional, capturando relações semânticas."

[6] "Contexto: Conjunto de palavras que cercam uma palavra-alvo, usado para prever ou entender o significado da palavra."

[7] "A qualidade dos embeddings gerados por modelos self-supervisionados depende criticamente da riqueza e diversidade do corpus de treinamento."

[8] "Esta hipótese postula que palavras que ocorrem em contextos similares tendem a ter significados semelhantes."

[9] "Essa formulação captura a ideia de que, se duas palavras têm distribuições de contexto similares, elas devem estar próximas no espaço semântico."

[10] "A evolução da self-supervision em PLN pode ser traçada desde os trabalhos seminais de Bengio et al. (2003) e Collobert et al. (2011)."

[11] "O avanço crucial veio com Mikolov et al. (2013a, 2013b), que introduziram o Word2Vec."

[12] "O Word2Vec revelou propriedades algébricas surpreendentes nas representações aprendidas."

[13] "Técnicas inspiradas no Word2Vec têm sido adaptadas para aprender representações de imagens sem rótulos."

[14] "Embeddings de usuários e itens podem ser aprendidos de forma self-supervisionada a partir de históricos de interações."

[15] "Sequências de proteínas podem ser representadas como 'frases', permitindo a aplicação de técnicas de PLN."

[16] "Existem duas arquiteturas principais: Skip-gram e Continuous Bag of Words (CBOW)."

[17] "O SGNS é uma variante do Skip-gram que se tornou particularmente influente."

[18] "Esta função maximiza a similaridade entre palavras e seus contextos verdadeiros."

[19] "A negative sampling é crucial para a eficiência do SGNS."

[20] "Os embeddings aprendidos exibem propriedades semânticas fascinantes."

[21] "Isso é frequentemente demonstrado através do 'modelo de paralelogramo'."

[22] "Esta propriedade sugere que os embeddings codificam diferenças semânticas."

[23] "Técnicas de redução de dimensionalidade como t-SNE são frequentemente aplicadas."

[24] "Palavras com múltiplos significados são representadas por um único vetor."

[25] "Os embeddings podem perpetuar e amplificar preconceitos presentes nos dados."

[26] "As dimensões individuais dos vetores de embedding geralmente carecem de interpretação."

[27] "Desenvolver métodos que possam abordar estas limitações permanece uma área ativa de pesquisa."

[28] "Estes métodos permitem o aprendizado de embeddings ricos e úteis."

[29] "Abriu novas possibilidades para compreender e modelar a linguagem."

[30] "Os princípios de self-supervision continuam a desempenhar um papel crucial."

[31] "Sob estas condições, o SGD converge para um ponto crítico da função de perda."