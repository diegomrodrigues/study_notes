# Exemplo Numérico: Classificação de Texto com Multinomial Naïve Bayes

Neste exemplo, iremos explorar os principais conceitos do **Multinomial Naïve Bayes** aplicados à classificação de textos. Utilizaremos um pequeno conjunto de dados para ilustrar passo a passo o processo de treinamento e predição do modelo.

## Conjunto de Dados

Temos um conjunto de documentos pertencentes a duas classes: **Esporte** e **Política**.

### Documentos de Treinamento

| ID   | Classe   | Documento                               |
| ---- | -------- | --------------------------------------- |
| D1   | Esporte  | "jogo de futebol emocionante"           |
| D2   | Esporte  | "torcida celebra vitória no campeonato" |
| D3   | Política | "debate político acirrado no congresso" |
| D4   | Política | "eleição presidencial próxima"          |

### Construção do Vocabulário

Extraímos todas as palavras dos documentos para formar o vocabulário.

**Vocabulário (V = 16):**

1.  jogo
2.  de
3.  futebol
4.  emocionante
5.  torcida
6.  celebra
7.  vitória
8.  no
9.  campeonato
10. debate
11. político
12. acirrado
13. congresso
14. eleição
15. presidencial
16. próxima

## Representação Bag-of-Words

Convertemos cada documento em um vetor de contagem de palavras (bag-of-words) com base no vocabulário.

### Vetores de Contagem

| Palavra      | Índice | D1   | D2   | D3   | D4   |
| ------------ | ------ | ---- | ---- | ---- | ---- |
| jogo         | 1      | 1    | 0    | 0    | 0    |
| de           | 2      | 1    | 0    | 0    | 0    |
| futebol      | 3      | 1    | 0    | 0    | 0    |
| emocionante  | 4      | 1    | 0    | 0    | 0    |
| torcida      | 5      | 0    | 1    | 0    | 0    |
| celebra      | 6      | 0    | 1    | 0    | 0    |
| vitória      | 7      | 0    | 1    | 0    | 0    |
| no           | 8      | 0    | 1    | 1    | 0    |
| campeonato   | 9      | 0    | 1    | 0    | 0    |
| debate       | 10     | 0    | 0    | 1    | 0    |
| político     | 11     | 0    | 0    | 1    | 0    |
| acirrado     | 12     | 0    | 0    | 1    | 0    |
| congresso    | 13     | 0    | 0    | 1    | 0    |
| eleição      | 14     | 0    | 0    | 0    | 1    |
| presidencial | 15     | 0    | 0    | 0    | 1    |
| próxima      | 16     | 0    | 0    | 0    | 1    |

## Cálculo das Probabilidades a Priori $p(y)$

Calculamos as probabilidades a priori de cada classe com base no número de documentos de cada classe.

- Número total de documentos: $N = 4$
- Número de documentos da classe **Esporte**: $N_{\text{Esporte}} = 2$
- Número de documentos da classe **Política**: $N_{\text{Política}} = 2$

As probabilidades a priori são:

$$
p(\text{Esporte}) = \frac{N_{\text{Esporte}}}{N} = \frac{2}{4} = 0{,}5 \\
p(\text{Política}) = \frac{N_{\text{Política}}}{N} = \frac{2}{4} = 0{,}5
$$

## Estimativa dos Parâmetros $\phi_{y,j}$ com Suavização de Laplace

### Contagem de Palavras por Classe

#### Classe **Esporte**

Somamos as contagens das palavras nos documentos D1 e D2.

| Palavra     | Índice | Contagem na Classe Esporte |
| ----------- | ------ | -------------------------- |
| jogo        | 1      | 1                          |
| de          | 2      | 1                          |
| futebol     | 3      | 1                          |
| emocionante | 4      | 1                          |
| torcida     | 5      | 1                          |
| celebra     | 6      | 1                          |
| vitória     | 7      | 1                          |
| no          | 8      | 1                          |
| campeonato  | 9      | 1                          |
| Outras      | 10-16  | 0                          |

- Total de palavras na classe Esporte: $N_{\text{Esporte}}^{\text{palavras}} = 9$

#### Classe **Política**

Somamos as contagens das palavras nos documentos D3 e D4.

| Palavra      | Índice | Contagem na Classe Política |
| ------------ | ------ | --------------------------- |
| no           | 8      | 1                           |
| debate       | 10     | 1                           |
| político     | 11     | 1                           |
| acirrado     | 12     | 1                           |
| congresso    | 13     | 1                           |
| eleição      | 14     | 1                           |
| presidencial | 15     | 1                           |
| próxima      | 16     | 1                           |
| Outras       | 1-7,9  | 0                           |

- Total de palavras na classe Política: $N_{\text{Política}}^{\text{palavras}} = 8$

### Aplicação da Suavização de Laplace ($\alpha = 1$)

Calculamos os parâmetros $\phi_{y,j}$ com a suavização:

$$
\phi_{y,j} = \frac{\alpha + \text{contagem}(w_j, y)}{V \alpha + N_{y}^{\text{palavras}}}
$$

Onde:

- $V = 16$ é o tamanho do vocabulário.
- $N_{y}^{\text{palavras}}$ é o total de palavras na classe $y$.

#### Denominadores:

- Para Esporte: $16 \times 1 + 9 = 25$
- Para Política: $16 \times 1 + 8 = 24$

#### Parâmetros para a Classe **Esporte**

Para palavras com contagem $\text{contagem}(w_j, \text{Esporte}) = 1$:

$$
\phi_{\text{Esporte},j} = \frac{1 + 1}{25} = \frac{2}{25} = 0{,}08
$$

Para palavras com contagem $0$:

$$
\phi_{\text{Esporte},j} = \frac{1 + 0}{25} = \frac{1}{25} = 0{,}04
$$

#### Parâmetros para a Classe **Política**

Para palavras com contagem $1$:

$$
\phi_{\text{Política},j} = \frac{1 + 1}{24} = \frac{2}{24} \approx 0{,}0833
$$

Para palavras com contagem $0$:

$$
\phi_{\text{Política},j} = \frac{1 + 0}{24} = \frac{1}{24} \approx 0{,}0417
$$

### Tabela de Parâmetros $\phi_{y,j}$

| Palavra      | Índice | $\phi_{\text{Esporte},j}$ | $\phi_{\text{Política},j}$ |
| ------------ | ------ | ------------------------- | -------------------------- |
| jogo         | 1      | 0,08                      | 0,0417                     |
| de           | 2      | 0,08                      | 0,0417                     |
| futebol      | 3      | 0,08                      | 0,0417                     |
| emocionante  | 4      | 0,08                      | 0,0417                     |
| torcida      | 5      | 0,08                      | 0,0417                     |
| celebra      | 6      | 0,08                      | 0,0417                     |
| vitória      | 7      | 0,08                      | 0,0417                     |
| no           | 8      | 0,08                      | 0,0833                     |
| campeonato   | 9      | 0,08                      | 0,0417                     |
| debate       | 10     | 0,04                      | 0,0833                     |
| político     | 11     | 0,04                      | 0,0833                     |
| acirrado     | 12     | 0,04                      | 0,0833                     |
| congresso    | 13     | 0,04                      | 0,0833                     |
| eleição      | 14     | 0,04                      | 0,0833                     |
| presidencial | 15     | 0,04                      | 0,0833                     |
| próxima      | 16     | 0,04                      | 0,0833                     |

## Classificação de um Novo Documento

Vamos classificar o seguinte documento:

- **Documento D5**: "vitória no congresso"

### Vetor de Contagem do Documento D5

| Palavra   | Índice | Contagem em D5 |
| --------- | ------ | -------------- |
| vitória   | 7      | 1              |
| no        | 8      | 1              |
| congresso | 13     | 1              |
| Outras    |        | 0              |

### Cálculo dos Scores para Cada Classe

Utilizamos a fórmula:

$$
\text{score}(y) = \log p(y) + \sum_{j=1}^V x_j \log \phi_{y,j}
$$

Onde:

- $x_j$ é a contagem da palavra $w_j$ no documento D5.
- $\log p(y)$ é o logaritmo da probabilidade a priori da classe $y$.

#### Logaritmos das Probabilidades a Priori

$$
\log p(\text{Esporte}) = \log 0{,}5 \approx -0{,}6931 \\
\log p(\text{Política}) = \log 0{,}5 \approx -0{,}6931
$$

#### Logaritmos dos Parâmetros $\phi_{y,j}$

Calculamos os logaritmos das probabilidades condicionais para as palavras presentes em D5.

##### Classe **Esporte**

- $\log \phi_{\text{Esporte},7} = \log 0{,}08 \approx -2{,}5257$
- $\log \phi_{\text{Esporte},8} = \log 0{,}08 \approx -2{,}5257$
- $\log \phi_{\text{Esporte},13} = \log 0{,}04 \approx -3{,}2189$

##### Classe **Política**

- $\log \phi_{\text{Política},7} = \log 0{,}0417 \approx -3{,}1781$
- $\log \phi_{\text{Política},8} = \log 0{,}0833 \approx -2{,}4849$
- $\log \phi_{\text{Política},13} = \log 0{,}0833 \approx -2{,}4849$

#### Cálculo dos Scores

##### Score para a Classe **Esporte**

$$
\text{score}(\text{Esporte}) = -0{,}6931 + 1 \times (-2{,}5257) + 1 \times (-2{,}5257) + 1 \times (-3{,}2189)
$$

$$
\text{score}(\text{Esporte}) = -0{,}6931 - 2{,}5257 - 2{,}5257 - 3{,}2189 = -8{,}9634
$$

##### Score para a Classe **Política**

$$
\text{score}(\text{Política}) = -0{,}6931 + 1 \times (-3{,}1781) + 1 \times (-2{,}4849) + 1 \times (-2{,}4849)
$$

$$
\text{score}(\text{Política}) = -0{,}6931 - 3{,}1781 - 2{,}4849 - 2{,}4849 = -8{,}8410
$$

### Decisão de Classificação

Comparando os scores:

- $\text{score}(\text{Esporte}) = -8{,}9634$
- $\text{score}(\text{Política}) = -8{,}8410$

Como $\text{score}(\text{Política}) > \text{score}(\text{Esporte})$ (menos negativo), classificamos o documento D5 como pertencente à classe **Política**.

## Interpretação dos Resultados

Apesar de a palavra "vitória" estar associada à classe **Esporte**, a presença das palavras "congresso" e "no" (que tem maior probabilidade na classe **Política** após a suavização) influenciou a classificação final. Isso ilustra como o modelo considera a combinação das evidências das palavras para tomar a decisão.

## Conclusão do Exemplo

Este exemplo numérico demonstra:

- **Construção do vocabulário** e representação dos documentos usando **bag-of-words**.
- **Cálculo das probabilidades a priori** $p(y)$.
- **Estimativa dos parâmetros** $\phi_{y,j}$ com **suavização de Laplace** para evitar probabilidades zero.
- **Cálculo dos scores** para cada classe usando os **logaritmos das probabilidades**, o que facilita os cálculos e evita problemas numéricos.
- **Classificação de um novo documento** com base nos scores calculados.

Este processo reflete os principais conceitos do **Multinomial Naïve Bayes** e mostra como o modelo utiliza as informações das frequências de palavras nos documentos de treinamento para realizar predições em novos dados.

---

**Nota:** Este exemplo simplificado ilustra o funcionamento do Multinomial Naïve Bayes em um contexto controlado. Em aplicações reais, o vocabulário e os conjuntos de dados são muito maiores, e técnicas adicionais de pré-processamento, como remoção de stop words e stemming, são comumente aplicadas.