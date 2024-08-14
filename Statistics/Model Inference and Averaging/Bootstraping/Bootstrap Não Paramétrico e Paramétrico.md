## Bootstrap Não Paramétrico e Paramétrico: Técnicas de Reamostragem para Inferência Estatística

![image-20240811111909834](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240811111909834.png)

O bootstrap é uma técnica estatística poderosa que permite realizar inferências sobre parâmetros populacionais e construir intervalos de confiança com base em uma única amostra de dados. Este resumo aborda duas variantes principais do bootstrap: o não paramétrico e o paramétrico, focando em suas aplicações, vantagens e limitações no contexto da inferência estatística e modelagem de dados.

### Conceitos Fundamentais

| Conceito                      | Explicação                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Bootstrap Não Paramétrico** | Técnica de reamostragem que gera novas amostras através da amostragem com reposição dos dados originais, sem assumir uma distribuição específica. [1] |
| **Bootstrap Paramétrico**     | Método que gera novas amostras a partir de um modelo paramétrico ajustado aos dados originais, assumindo uma forma específica de distribuição. [6] |
| **Inferência Estatística**    | Processo de fazer conclusões sobre uma população com base em uma amostra, utilizando técnicas como o bootstrap para estimar a variabilidade das estimativas. [1] |

> ✔️ **Ponto de Destaque**: O bootstrap permite estimar a distribuição amostral de estatísticas complexas sem a necessidade de fórmulas analíticas, tornando-o uma ferramenta versátil para inferência estatística.

### Bootstrap Não Paramétrico

O bootstrap não paramétrico é uma técnica fundamental na estatística moderna, introduzida por Bradley Efron em 1979. Este método permite realizar inferências sobre parâmetros populacionais sem fazer suposições sobre a distribuição subjacente dos dados.

#### Procedimento do Bootstrap Não Paramétrico

1. Seja $Z = \{z_1, z_2, ..., z_N\}$ a amostra original de tamanho $N$.
2. Crie $B$ amostras bootstrap $Z^{*1}, Z^{*2}, ..., Z^{*B}$, cada uma de tamanho $N$, amostrando com reposição de $Z$.
3. Para cada amostra bootstrap, calcule a estatística de interesse $\hat{\theta}^{*b}$.
4. Use a distribuição empírica de $\hat{\theta}^{*b}$ para fazer inferências sobre $\theta$.

A distribuição empírica dos $\hat{\theta}^{*b}$ aproxima a verdadeira distribuição amostral de $\hat{\theta}$, permitindo a construção de intervalos de confiança e testes de hipóteses. [1]

> ⚠️ **Nota Importante**: O bootstrap não paramétrico assume que a amostra original é representativa da população. Se a amostra for enviesada, as inferências baseadas no bootstrap também serão.

#### Aplicações do Bootstrap Não Paramétrico

1. **Estimação de Erro Padrão**: O erro padrão de $\hat{\theta}$ pode ser estimado pela fórmula:

   $$se_B(\hat{\theta}) = \sqrt{\frac{1}{B-1} \sum_{b=1}^B (\hat{\theta}^{*b} - \bar{\theta}^*)^2}$$

   onde $\bar{\theta}^* = \frac{1}{B} \sum_{b=1}^B \hat{\theta}^{*b}$. [2]

2. **Intervalos de Confiança**: O método percentil simples usa os percentis da distribuição bootstrap para construir intervalos de confiança. Para um intervalo de 95%, use os percentis 2,5 e 97,5 dos $\hat{\theta}^{*b}$. [3]

3. **Correção de Viés**: O bootstrap pode ser usado para estimar e corrigir o viés de estimadores:

   $$\text{bias}_{\text{boot}} = \bar{\theta}^* - \hat{\theta}$$

   O estimador corrigido é então $\hat{\theta}_{\text{corr}} = \hat{\theta} - \text{bias}_{\text{boot}}$. [4]

#### Vantagens e Desvantagens do Bootstrap Não Paramétrico

| 👍 Vantagens                                                  | 👎 Desvantagens                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Não requer suposições sobre a distribuição dos dados         | Pode ser computacionalmente intensivo para grandes conjuntos de dados |
| Aplicável a uma ampla variedade de estatísticas              | Sensível a outliers e amostras não representativas           |
| Fornece estimativas de erro padrão e intervalos de confiança para estatísticas complexas | Pode subestimar a variabilidade em amostras pequenas         |

#### Questões Técnicas/Teóricas

1. Como o tamanho da amostra original afeta a precisão das estimativas bootstrap? Discuta as implicações para amostras pequenas versus grandes.

2. Em um cenário onde você suspeita de heterogeneidade nos dados (por exemplo, mistura de populações), como o bootstrap não paramétrico pode ser adaptado? Quais são as potenciais armadilhas?

### Bootstrap Paramétrico

O bootstrap paramétrico é uma variante que assume um modelo probabilístico específico para os dados. Este método é particularmente útil quando temos conhecimento prévio sobre a distribuição dos dados ou quando queremos fazer inferências baseadas em um modelo paramétrico ajustado.

#### Procedimento do Bootstrap Paramétrico

1. Ajuste um modelo paramétrico $F_{\hat{\theta}}$ aos dados originais $Z$.
2. Gere $B$ amostras bootstrap $Z^{*1}, Z^{*2}, ..., Z^{*B}$ simulando a partir de $F_{\hat{\theta}}$.
3. Para cada amostra bootstrap, recalcule o parâmetro de interesse $\hat{\theta}^{*b}$.
4. Use a distribuição empírica de $\hat{\theta}^{*b}$ para inferência.

> ❗ **Ponto de Atenção**: O bootstrap paramétrico é sensível à escolha do modelo. Um modelo mal especificado pode levar a inferências incorretas.

#### Exemplo: Bootstrap Paramétrico para Regressão Linear

Considere um modelo de regressão linear simples:

$$y_i = \beta_0 + \beta_1x_i + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)$$

O bootstrap paramétrico para este modelo seguiria os seguintes passos:

1. Ajuste o modelo aos dados originais, obtendo $\hat{\beta}_0$, $\hat{\beta}_1$, e $\hat{\sigma}^2$.
2. Gere novas respostas $y_i^*$ adicionando ruído Gaussiano aos valores preditos:

   $$y_i^* = \hat{\beta}_0 + \hat{\beta}_1x_i + \varepsilon_i^*, \quad \varepsilon_i^* \sim N(0, \hat{\sigma}^2)$$

3. Reajuste o modelo a cada conjunto de dados bootstrap $(x_i, y_i^*)$.
4. Repita os passos 2-3 $B$ vezes.

Este procedimento gera uma distribuição bootstrap dos parâmetros $\beta_0$ e $\beta_1$, permitindo a construção de intervalos de confiança e testes de hipóteses. [6]

#### Comparação entre Bootstrap Não Paramétrico e Paramétrico

| Aspecto       | Bootstrap Não Paramétrico               | Bootstrap Paramétrico                        |
| ------------- | --------------------------------------- | -------------------------------------------- |
| Suposições    | Mínimas sobre a distribuição dos dados  | Assume um modelo probabilístico específico   |
| Flexibilidade | Alta, aplicável a diversas estatísticas | Limitada ao modelo assumido                  |
| Eficiência    | Pode requerer mais amostras bootstrap   | Geralmente mais eficiente com menos amostras |
| Viés          | Menos propenso a viés de modelo         | Pode ser enviesado se o modelo for incorreto |

> 💡 **Dica**: O bootstrap paramétrico é particularmente útil quando temos um bom entendimento do processo gerador de dados e queremos fazer inferências dentro desse modelo específico.

#### Questões Técnicas/Teóricas

1. Em um cenário de regressão logística com dados desbalanceados, como você compararia o desempenho do bootstrap não paramétrico versus o paramétrico para estimar os intervalos de confiança dos coeficientes?

2. Discuta as implicações de usar o bootstrap paramétrico quando o modelo assumido é uma aproximação razoável, mas não exata, da verdadeira distribuição dos dados. Como isso afeta as inferências?

### Aplicações Avançadas do Bootstrap

#### Bootstrap para Séries Temporais

Para dados com dependência temporal, técnicas especiais de bootstrap são necessárias:

1. **Block Bootstrap**: Reamostra blocos de observações consecutivas para preservar a estrutura de dependência.
2. **Sieve Bootstrap**: Ajusta um modelo autorregressivo e aplica o bootstrap aos resíduos.

#### Bootstrap para Dados Espaciais

Para dados com dependência espacial:

1. **Spatial Block Bootstrap**: Similar ao block bootstrap, mas em duas ou três dimensões.
2. **Variogram-Based Bootstrap**: Usa o variograma empírico para gerar amostras bootstrap que preservam a estrutura de correlação espacial.

#### Bootstrap em Aprendizado de Máquina

1. **Bagging (Bootstrap Aggregating)**: Usa múltiplas amostras bootstrap para treinar um conjunto de modelos, melhorando a estabilidade e reduzindo o overfitting.

2. **Random Forests**: Estende o bagging incorporando seleção aleatória de features em cada split de árvore.

> ✔️ **Ponto de Destaque**: O bootstrap não é apenas uma ferramenta estatística, mas também um componente fundamental em muitos algoritmos de ensemble em aprendizado de máquina.

### Conclusão

O bootstrap, tanto em sua forma não paramétrica quanto paramétrica, é uma ferramenta estatística poderosa e versátil para inferência e modelagem. O bootstrap não paramétrico oferece grande flexibilidade e robustez, enquanto o paramétrico pode fornecer estimativas mais precisas quando as suposições do modelo são válidas. A escolha entre as duas abordagens depende do contexto específico do problema, do conhecimento prévio sobre os dados e dos objetivos da análise. Em muitos casos, a combinação de ambas as abordagens pode fornecer insights valiosos e uma compreensão mais completa da incerteza associada às estimativas estatísticas.

### Questões Avançadas

1. Considere um cenário de aprendizado profundo onde você está treinando uma rede neural convolucional para classificação de imagens. Como você poderia incorporar técnicas de bootstrap para quantificar a incerteza nas previsões do modelo? Discuta as vantagens e limitações dessa abordagem em comparação com métodos bayesianos.

2. Em um estudo ecológico, você tem dados de contagem de espécies em diferentes locais geográficos, com possível dependência espacial e temporal. Proponha uma estratégia de bootstrap que leve em conta ambas as fontes de dependência e discuta como você validaria a eficácia dessa abordagem.

3. No contexto de seleção de modelos em regressão de alto dimensão (p >> n), como você poderia usar o bootstrap para avaliar a estabilidade da seleção de variáveis? Discuta as implicações para a inferência e interpretação do modelo final.

### Referências

[1] "O bootstrap método provê uma direta maneira computacional de avaliar incerteza, por amostragem dos dados de treinamento. Aqui nós ilustramos o bootstrap em um simples problema de suavização unidimensional, e mostramos sua conexão com máxima verossimilhança." (Trecho de ESL II)

[2] "Seja H a matriz N × 7 com elemento ij h_j(x_i). A estimativa usual de β, obtida minimizando o erro quadrático sobre o conjunto de treinamento, é dada por β̂ = (H^T H)^{-1}H^T y." (Trecho de ESL II)

[3] "A matriz de covariância estimada de β̂ é Var(β̂) = (H^T H)^{-1}σ̂^2, onde estimamos a variância do ruído por σ̂^2 = ∑^N_{i=1}(y_i - μ̂(x_i))^2/N." (Trecho de ESL II)

[4] "O painel inferior direito da Figura 8.2 mostra μ̂(x) ± 1.96 · se[μ̂(x)]. Como 1.96 é o ponto 97.5% da distribuição normal padrão, estes representam bandas de confiança pontual aproximadas de 100 − 2 × 2.5% = 95% para μ(x)." (Trecho de ESL II)

[5] "Aqui está como poderíamos aplicar o bootstrap neste exemplo. Desenhamos B conjuntos de dados cada um de tamanho N = 50 com reposição de nossos dados de treinamento, sendo a unidade de amostragem o par z_i = (x_i, y_i)." (Trecho de ESL II)

[6] "Considere uma variação do bootstrap, chamada bootstrap paramétrico, na qual simulamos novas respostas adicionando ruído Gaussiano aos valores preditos: y^*_i = μ̂(x_i) + ε^*_i; ε^*_i ∼ N(0, σ̂^2); i = 1, 2, . . . , N." (Trecho de ESL II)

[7] "Este processo é repetido B vezes, onde B = 200 digamos. Os conjuntos de dados bootstrap resultantes têm a forma (x_1, y^*_1), . . . , (x_N, y^*_N) e recomputamos a suavização B-spline em cada um." (Trecho de ESL II)

[8] "As bandas de confiança deste método serão exatamente iguais às bandas de mínimos quadrados no painel superior direito, à medida que o número de amostras bootstrap vai para o infinito." (Trecho de ESL II)