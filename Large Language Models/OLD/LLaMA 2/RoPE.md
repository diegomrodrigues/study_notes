## RoFormer: Transformador Aprimorado com Rotary Position Embedding

![image-20240917120717832](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240917120717832.png)

### Introdução

O artigo "RoFormer: Enhanced Transformer with Rotary Position Embedding" apresenta uma inovação significativa na arquitetura Transformer, focando na melhoria do encoding de informações posicionais [1]. Os autores abordam uma limitação fundamental dos modelos Transformer: ==a falta de sensibilidade à ordem sequencial das palavras, que é crucial para a compreensão da linguagem natural [2].==

O objetivo principal do estudo é introduzir um novo método de embedding posicional, denominado Rotary Position Embedding (RoPE), que ==codifica informações de posição absoluta usando uma matriz de rotação e, simultaneamente, incorpora dependência de posição relativa explícita na formulação de self-attention [3].== Este método visa superar as limitações das abordagens existentes de encoding posicional, oferecendo ==propriedades valiosas como flexibilidade de comprimento de sequência e decaimento da dependência inter-token com o aumento das distâncias relativas [4].==

> 💡 **Contribuição Chave**: O RoFormer propõe uma nova forma de incorporar informações posicionais em modelos Transformer, potencialmente melhorando o desempenho em tarefas de processamento de linguagem natural, especialmente para textos longos.

### Revisão da Literatura

O artigo posiciona o RoFormer no contexto de pesquisas existentes sobre encoding posicional em arquiteturas Transformer. Os autores identificam duas principais abordagens na literatura:

1. **Encoding de posição absoluta**:
   - Gerado por funções pré-definidas [5]
   - ==Encoding de posição absoluta treinável [6]==

2. **Encoding de posição relativa**:
   - ==Incorporação de informações de posição relativa no mecanismo de atenção [7]==

Os autores argumentam que, apesar da eficácia dessas abordagens, ==elas geralmente adicionam informações de posição à representação do contexto, tornando-as inadequadas para arquiteturas de self-attention linear [8].==

| Abordagem                    | Vantagens                                      | Limitações                                        |
| ---------------------------- | ---------------------------------------------- | ------------------------------------------------- |
| Encoding de posição absoluta | Simplicidade, fácil implementação              | Pode não capturar relações de longo alcance       |
| Encoding de posição relativa | Captura melhor relações entre tokens distantes | Complexidade computacional, difícil implementação |
| RoPE (proposto)              | Flexibilidade, eficácia em textos longos       | A ser determinado pela pesquisa futura            |

==O RoFormer se distingue ao propor uma abordagem que codifica tanto a posição absoluta quanto a relativa, mantendo a compatibilidade com self-attention linear [9].==

### Metodologia

#### Modelos Teóricos e Conceituais:

1. **Rotary Position Embedding (RoPE)**:

   O RoPE é a inovação central do RoFormer. ==Ele codifica a posição absoluta com uma matriz de rotação e incorpora dependência de posição relativa explícita na formulação de self-attention [10].==

   $$
   f_{q,k}(x_m, m) = R_{\Theta,m}^d W_{q,k}x_m
   $$

   Onde:
   - $f_{q,k}$ são as funções de transformação para queries e keys
   - $x_m$ é o embedding da palavra na posição $m$
   - $R_{\Theta,m}^d$ é a matriz de rotação
   - $W_{q,k}$ são as matrizes de projeção para queries e keys

2. **Matriz de Rotação**:

   A matriz de rotação $R_{\Theta,m}^d$ é definida como [11]:

   $$
   R_{\Theta,m}^d = \begin{pmatrix}
   \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
   \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
   0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
   0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
   \vdots & \vdots & \vdots & \vdots & \ddots
   \end{pmatrix}
   $$

   Com $\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, ..., d/2]\}$

> ⚠️ **Ponto Crucial**: ==A escolha dos valores de $\theta_i$ é fundamental para garantir a propriedade de decaimento de longo prazo do RoPE.==

#### Procedimentos Experimentais:

Os autores conduziram experimentos em várias tarefas para validar a eficácia do RoFormer:

1. **Tradução Automática**: Utilizando o conjunto de dados WMT 2014 English-German [12].
2. **Pré-treinamento de Modelo de Linguagem**: Usando BookCorpus e Wikipedia Corpus [13].
3. **Fine-tuning em tarefas GLUE**: Incluindo MRPC, SST-2, QNLI, STS-B, QQP e MNLI [14].
4. **Performer com RoPE**: Implementação do RoPE no modelo Performer [15].
5. **Avaliação em Dados Chineses**: Usando o conjunto de dados CAIL2019-SCM [16].

### RoPE com Atenção Linear

A incorporação do **Rotary Position Embedding (RoPE)** em arquiteturas de atenção linear representa um avanço significativo na evolução dos modelos Transformer. ==O RoPE oferece uma maneira eficiente de incorporar informações posicionais através de rotações no espaço vetorial, permitindo que o modelo capture relações posicionales de forma contínua e sem depender de embeddings posicionais explícitos.== Nesta seção, exploraremos em profundidade como o RoPE pode ser integrado aos mecanismos de atenção linear, mantendo suas propriedades benéficas enquanto reduz a complexidade computacional associada à atenção tradicional.

#### Formulação Geral da Atenção

A atenção em modelos Transformer pode ser generalizada pela seguinte equação:

$$
\text{Atenção}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_m = \frac{\sum_{n=1}^N \text{sim}(\mathbf{q}_m, \mathbf{k}_n) \mathbf{v}_n}{\sum_{n=1}^N \text{sim}(\mathbf{q}_m, \mathbf{k}_n)}
$$

Onde:

- $\mathbf{q}_m \in \mathbb{R}^d$ é a **query** na posição $m$.
- $\mathbf{k}_n \in \mathbb{R}^d$ é a **key** na posição $n$.
- $\mathbf{v}_n \in \mathbb{R}^d$ é o **value** na posição $n$.
- $\text{sim}(\mathbf{q}_m, \mathbf{k}_n)$ é uma função de **similaridade** que quantifica a compatibilidade entre $\mathbf{q}_m$ e $\mathbf{k}_n$.

==Na atenção tradicional, a função de similaridade é definida como:==
$$
\text{sim}(\mathbf{q}_m, \mathbf{k}_n) = \exp\left(\frac{\mathbf{q}_m^\top \mathbf{k}_n}{\sqrt{d}}\right)
$$

Esta formulação resulta em uma complexidade computacional de **$\mathcal{O}(N^2)$**, pois calcula-se a interação entre todas as pares possíveis de queries e keys em uma sequência de comprimento $N$.

#### Atenção Linear

Para superar a limitação de complexidade quadrática, a atenção linear reformula o mecanismo de atenção de modo a permitir computações mais eficientes. ==A ideia central é decompor a função de similaridade em produtos internos de funções não-negativas, permitindo a reordenação das operações:==

$$
\text{Atenção}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_m = \frac{\phi(\mathbf{q}_m)^\top \left( \sum_{n=1}^N \varphi(\mathbf{k}_n) \odot \mathbf{v}_n \right)}{\phi(\mathbf{q}_m)^\top \left( \sum_{n=1}^N \varphi(\mathbf{k}_n) \right)}
$$

Onde:

- $\phi(\cdot)$ e $\varphi(\cdot)$ são ==funções de mapeamento não-negativas== aplicadas às queries e keys, respectivamente.
- ==$\odot$ representa o **produto de Hadamard** (produto elemento a elemento).==

**Abordagens Comuns:**

| Abordagem            | $\phi(\cdot)$                                       | $\varphi(\cdot)$                             | Características                                              |
| -------------------- | --------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| Katharopoulos et al. | $\phi(x) = \varphi(x) = \text{elu}(x) + 1$          | $\phi(x) = \varphi(x) = \text{elu}(x) + 1$   | Aproveita a propriedade associativa para eficiência computacional |
| Shen et al.          | $\phi(\mathbf{q}_i) = \text{softmax}(\mathbf{q}_i)$ | $\varphi(\mathbf{k}_j) = \exp(\mathbf{k}_j)$ | Normaliza queries e keys separadamente para estabilidade numérica |

==Essas funções permitem que o cálculo da atenção seja reescrito de forma que a complexidade computacional seja reduzida para **$\mathcal{O}(N)$**.==

#### Integração do RoPE com Atenção Linear

A integração do RoPE em atenção linear é realizada aplicando as matrizes de rotação às saídas das funções não-negativas $\phi(\cdot)$ e $\varphi(\cdot)$. A fórmula modificada da atenção linear com RoPE torna-se:

$$
\text{Atenção}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_m = \frac{\left( R_{\Theta, m} \phi(\mathbf{q}_m) \right)^\top \left( \sum_{n=1}^N R_{\Theta, n} \varphi(\mathbf{k}_n) \odot \mathbf{v}_n \right)}{\left( R_{\Theta, m} \phi(\mathbf{q}_m) \right)^\top \left( \sum_{n=1}^N R_{\Theta, n} \varphi(\mathbf{k}_n) \right)}
$$

Onde:

- $R_{\Theta, m}$ é a **matriz de rotação** gerada pelo RoPE para a posição $m$.
- As rotações incorporam informações posicionais diretamente nas representações de queries e keys.

**Detalhes Importantes:**

- **Aplicação das Rotações:** As matrizes de rotação são aplicadas **após** as funções $\phi(\cdot)$ e $\varphi(\cdot)$, garantindo que as propriedades não-negativas dessas funções sejam mantidas.
- **Estabilidade Numérica:** O denominador é cuidadosamente tratado para evitar divisão por zero e manter a estabilidade durante o treinamento.

#### Análise Teórica

1. **Preservação da Norma:**
   - ==As matrizes de rotação $R_{\Theta, m}$ são **ortogonais**, o que significa que elas preservam a norma dos vetores.==
   - **Importância:** A preservação da norma é crucial para evitar mudanças abruptas na escala dos vetores, o que poderia levar a instabilidades no treinamento.

2. **Compatibilidade com Funções Não-Negativas:**
   - Ao aplicar as rotações após $\phi(\cdot)$ e $\varphi(\cdot)$, asseguramos que as saídas permaneçam compatíveis com os requisitos da atenção linear.
   - **Resultado:** Mantém-se a interpretabilidade dos pesos de atenção e a eficiência computacional.

3. **Complexidade Computacional:**
   - A integração do RoPE não altera a complexidade linear da atenção.
   - **Justificativa:** As operações de rotação têm complexidade $\mathcal{O}(N)$ e podem ser implementadas de forma altamente eficiente.

#### Implicações Práticas

1. **Flexibilidade de Implementação:**
   - O RoPE pode ser facilmente integrado em modelos existentes com modificações mínimas.
   - **Benefício:** Facilita a adoção em diversos cenários sem a necessidade de reestruturar significativamente o modelo.

2. **Eficiência em Processamento de Sequências Longas:**
   - A combinação de RoPE com atenção linear é especialmente útil para tarefas que envolvem sequências longas, como processamento de texto longo ou séries temporais.
   - **Vantagem:** Permite treinamento e inferência eficientes onde a atenção tradicional seria computacionalmente inviável.

3. **Interpretação dos Pesos de Atenção:**
   - Embora os pesos resultantes não sejam probabilísticos no sentido estrito (não somam exatamente 1), eles ainda fornecem uma medida relativa da importância de cada valor $\mathbf{v}_n$ para a query $\mathbf{q}_m$.
   - **Consideração:** Pode ser necessário aplicar técnicas adicionais de normalização para certas aplicações que requerem pesos probabilísticos.

#### Considerações Adicionais

- **Propriedades do RoPE:**
  - O RoPE permite que o modelo capture relações posicionales além das adjacências imediatas, graças à natureza contínua das rotações.
  - **Efeito:** Melhora a capacidade do modelo de generalizar padrões posicionales em diferentes contextos.

- **Comparação com Embeddings Posicionais Absolutos:**
  - Enquanto embeddings posicionais absolutos atribuem um vetor fixo a cada posição, o RoPE incorpora a posição através de transformações aplicadas aos embeddings existentes.
  - **Benefício:** Reduz a necessidade de aprender ou armazenar embeddings adicionais, economizando recursos.

- **Desafios Potenciais:**
  - **Saturação das Funções Não-Negativas:** Em alguns casos, as funções $\phi(\cdot)$ e $\varphi(\cdot)$ podem saturar, levando a gradientes pequenos.
    - **Solução:** Ajustes nas funções de ativação ou nos hiperparâmetros podem mitigar esse problema.
  - **Inicialização dos Parâmetros:** A inicialização adequada das matrizes de rotação e dos pesos do modelo é essencial para um treinamento estável.

#### Exemplificação Matemática

Para ilustrar a eficácia da integração do RoPE com atenção linear, considere um exemplo simplificado onde:

- $\phi(\mathbf{q}_m) = \mathbf{q}_m$ (identidade).
- $\varphi(\mathbf{k}_n) = \mathbf{k}_n$ (identidade).
- As matrizes de rotação $R_{\Theta, m}$ são definidas por uma função senoidal baseada na posição.

Neste caso, a atenção pode ser computada de forma eficiente, e as rotações introduzem informações posicionales que permitem ao modelo diferenciar entre elementos em diferentes posições, mesmo quando os conteúdos são semelhantes.

### Resultados

Os resultados dos experimentos demonstram a eficácia do RoFormer em várias tarefas:

1. **Tradução Automática**:
   
   | Modelo      | BLEU Score |
   | ----------- | ---------- |
   | Transformer | 27.3       |
   | RoFormer    | 27.5       |

   O RoFormer superou o Transformer base na tarefa de tradução English-to-German [17].

2. **Pré-treinamento de Modelo de Linguagem**:

   <imagem: Um gráfico mostrando a perda de treinamento do BERT e do RoFormer ao longo do tempo, com o RoFormer convergindo mais rapidamente.>

   O RoFormer demonstrou convergência mais rápida durante o pré-treinamento em comparação com o BERT [18].

3. **Fine-tuning em tarefas GLUE**:

   | Modelo   | MRPC | SST-2 | QNLI | STS-B | QQP  | MNLI(m/mm) |
   | -------- | ---- | ----- | ---- | ----- | ---- | ---------- |
   | BERT     | 88.9 | 93.5  | 90.5 | 85.8  | 71.2 | 84.6/83.4  |
   | RoFormer | 89.5 | 90.7  | 88.0 | 87.0  | 86.4 | 80.2/79.8  |

   O RoFormer superou o BERT em três das seis tarefas avaliadas [19].

4. **Performer com RoPE**:

   A incorporação do RoPE no Performer levou a uma convergência mais rápida e menor perda durante o pré-treinamento [20].

5. **Avaliação em Dados Chineses**:

   | Modelo        | Validação | Teste  |
   | ------------- | --------- | ------ |
   | BERT-512      | 64.13%    | 67.77% |
   | WoBERT-512    | 64.07%    | 68.10% |
   | RoFormer-512  | 64.13%    | 68.29% |
   | RoFormer-1024 | 66.07%    | 69.79% |

   O RoFormer demonstrou desempenho superior em textos longos, superando modelos como BERT e WoBERT [21].

> ✔️ **Resultado Significativo**: O RoFormer mostrou melhorias consistentes em tarefas que envolvem textos longos, demonstrando a eficácia do RoPE em capturar dependências de longo alcance.

### Proposições, Teoremas e Provas

#### Proposição 1: Formulação do RoPE em 2D

**Enunciado**: Em um espaço bidimensional, o RoPE pode ser formulado como [22]:

$$
f_q(x_m, m) = (W_q x_m)e^{im\theta}
$$
$$
f_k(x_n, n) = (W_k x_n)e^{in\theta}
$$

**Prova**:

1. Considere a decomposição de funções em componentes radiais e angulares [23]:

   $$
   \begin{aligned}
   f_q(x_q, m) &= R_q(x_q, m)e^{i\Theta_q(x_q,m)} \\
   f_k(x_k, n) &= R_k(x_k, n)e^{i\Theta_k(x_k,n)} \\
   g(x_q, x_k, n - m) &= R_g(x_q, x_k, n - m)e^{i\Theta_g(x_q,x_k,n-m)}
   \end{aligned}
   $$

2. Aplicando as condições iniciais e as relações derivadas [24]:

   $$
   \begin{aligned}
   R_q(x_q, m) &= R_q(x_q, 0) = \|q\| \\
   R_k(x_k, n) &= R_k(x_k, 0) = \|k\| \\
   \Theta_f(x_{q,k}, m) &= \phi(m) + \theta_{q,k}
   \end{aligned}
   $$

3. Considerando $\phi(m) = m\theta + \gamma$, onde $\theta, \gamma \in \mathbb{R}$ são constantes [25].

4. Finalmente, definindo $q = W_q x_m$ e $k = W_k x_n$ [26], obtemos:

   $$
   \begin{aligned}
   f_q(x_m, m) &= (W_q x_m)e^{im\theta} \\
   f_k(x_n, n) &= (W_k x_n)e^{in\theta}
   \end{aligned}
   $$

> ❗ **Ponto de Atenção**: Esta formulação em 2D serve como base para a generalização do RoPE em dimensões superiores.

#### Teorema 1: Decaimento de Longo Prazo do RoPE

![image-20240917123312506](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240917123312506.png)

**Enunciado**: O RoPE possui a propriedade de decaimento de longo prazo, ==onde a influência entre tokens diminui com o aumento da distância relativa [27].==

**Prova**:

1. Considere o produto interno de RoPE em dimensão d [28]:

   $$
   (R_{\theta,m}^d W_q x_m)^T(R_{\theta,n}^d W_k x_n) = \text{Re}\left[\sum_{i=0}^{d/2-1} q_{[2i:2i+1]}k_{[2i:2i+1]}^*e^{i(m-n)\theta_i}\right]
   $$

2. Usando a transformação de Abel e definindo $h_i = q_{[2i:2i+1]}k_{[2i:2i+1]}^*$ e $S_j = \sum_{i=0}^{j-1} e^{i(m-n)\theta_i}$ [29]:

   $$
   \left|\sum_{i=0}^{d/2-1} q_{[2i:2i+1]}k_{[2i:2i+1]}^*e^{i(m-n)\theta_i}\right| \leq \left(\max_i |h_{i+1} - h_i|\right) \sum_{i=0}^{d/2-1} |S_{i+1}|
   $$

3. Com $\theta_i = 10000^{-2i/d}$, demonstra-se que $\frac{1}{d/2} \sum_{i=1}^{d/2} |S_i|$ decai com o aumento da distância relativa $m - n$ [30].

> 💡 **Insight Crucial**: Esta propriedade de decaimento é fundamental para a capacidade do RoFormer em modelar eficientemente dependências de longo alcance.

### Discussão

O RoFormer apresenta avanços significativos em relação a abordagens anteriores de encoding posicional:

1. **Flexibilidade de Comprimento de Sequência**: Ao contrário de métodos de posição absoluta, o RoPE não é limitado por um comprimento máximo de sequência predefinido [31].

2. **Compatibilidade com Self-Attention Linear**: O RoPE pode ser facilmente incorporado em arquiteturas de self-attention linear, como o Performer [32].

3. **Desempenho em Textos Longos**: Os resultados em tarefas de textos longos, como o CAIL2019-SCM, demonstram a eficácia do RoFormer em capturar dependências de longo alcance [33].

**Comparação com Trabalhos Anteriores**:

| Aspecto                            | RoFormer [34]           | Transformers Tradicionais [35]   |
| ---------------------------------- | ----------------------- | -------------------------------- |
| Encoding Posicional                | Rotação de embeddings   | Adição de embeddings posicionais |
| Captura de Posição Relativa        | Intrínseca à formulação | Requer modificações adicionais   |
| Desempenho em Textos Longos        | Superior                | Limitado pelo comprimento máximo |
| Compatibilidade com Atenção Linear | Compatível              | Geralmente incompatível          |

**Limitações e Perspectivas Futuras**:

1. **Complexidade Computacional**: Embora o RoPE seja eficiente, a implementação em larga escala pode requerer otimizações adicionais [36].
2. **Generalização para Outras Modalidades**: A eficácia do RoPE em tarefas além do processamento de texto natural ainda precisa ser investigada [37].
3. **Interpretabilidade**: Uma análise mais aprofundada da representação interna aprendida pelo RoFormer pode fornecer insights valiosos sobre seu funcionamento [38].

### Conclusão

O RoFormer, com sua inovadora abordagem de Rotary Position Embedding, representa um avanço significativo na arquitetura Transformer [39]. Ao codificar informações de posição através de rotações em vez de adições, o RoFormer demonstra melhorias consistentes em várias tarefas de processamento