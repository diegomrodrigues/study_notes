## Capítulo 4: Aplicações Linguísticas da Classificação

**4.1 Análise de Sentimento e Opinião**

- 
- **Classificação de Sentimento como Classificação de Documentos:** Conceito de classificação binária ou ternária (positivo, negativo, neutro) de documentos, utilizando técnicas de classificação de documentos.  Contexto: Aplicação direta de métodos de classificação previamente apresentados no livro.
- **Aquisição de Rótulos:**  Métodos para obter rótulos de sentimento, incluindo uso de emoticons, avaliações numéricas (estrelas) e posicionamentos políticos em votações. Contexto: Diferentes abordagens para criação de conjuntos de dados rotulados para análise de sentimento.
- **Modelo Bag-of-Words para Análise de Sentimento:**  Eficácia do modelo bag-of-words para documentos longos e suas limitações para documentos curtos. Contexto: Aplicação e limitações do modelo bag-of-words em análise de sentimento em diferentes tamanhos de texto.
- **Negation e Irrealis:**  Impacto da negação e de expressões irreais na classificação baseada em bag-of-words. Contexto: Desafio de lidar com nuances linguísticas que afetam a interpretação direta do sentimento.
- **Modelagem n-gramas:**  Utilizando bigramas e trigramas para capturar contextos maiores e lidar com negação. Contexto: Abordagem para melhorar a precisão ao considerar sequências de palavras adjacentes.
- **Abordagens Mais Sofisticadas:**  Consideração de estrutura sintática e uso de classificadores mais complexos (CNNs) para lidar com casos complexos de negação e expressões irreais. Contexto: Limitações das abordagens simples e a necessidade de modelos mais robustos.

**4.1.1 Problemas Relacionados**

- 
- **Detecção de Subjetividade:**  Identificação de partes subjetivas em textos, incluindo especulação e conteúdo hipotético. Contexto: Problema relacionado à análise de sentimento, focando na identificação da subjetividade da informação.
- **Classificação de Postura:**  Identificação da posição do autor em um argumento (a favor ou contra). Contexto: Classificação de textos baseado na postura do autor em relação a um determinado tema.
- **Análise de Sentimento Direcionado:**  Identificação do sentimento em relação a entidades específicas em um texto. Contexto: Análise de sentimento que busca refinar a granularidade, associando sentimentos a entidades específicas.
- **Mineração de Opiniões Baseada em Aspectos:**  Identificação do sentimento em relação a aspectos predefinidos (preço, serviço, etc.). Contexto: Método para focar em opiniões sobre características específicas de um produto ou serviço.
- **Classificação de Emoções:**  Classificação de textos com base em categorias emocionais multifacetadas. Contexto: Expansão do conceito de análise de sentimento para categorias emocionais mais complexas que apenas positivo e negativo.

**4.1.2 Abordagens Alternativas para Análise de Sentimento**

- 
- **Regressão:**  Utilização de regressão (linear e ridge) para determinar uma classificação numérica contínua do sentimento. Contexto: Abordagem que gera um score numérico do sentimento, ao invés de uma classificação categórica.
- **Classificação Ordinal:**  Classificação em escalas ordinais discretas (ex: 1-5 estrelas). Contexto: Modelagem de problemas onde as classes possuem uma ordem intrínseca.
- **Classificação Baseada em Léxico:**  Classificação baseada em listas de palavras associadas a cada categoria de sentimento. Contexto: Método que usa listas de palavras pré-definidas para determinar o sentimento de um texto, sem aprendizado de máquina.

**4.2 Desambiguização de Sentido de Palavras**

- 
- **Desambiguização de Sentido como Problema de Classificação:**  Utilização de técnicas de classificação para identificar o sentido correto de uma palavra em um contexto específico. Contexto:  Aplicação de métodos de classificação a um problema de processamento de linguagem natural.
- **Número de Sentidos de Palavras:**  A complexidade da polisemia e a definição de sentidos de palavras em WordNet. Contexto: Exploração da complexidade do problema de desambiguização, considerando a variedade de sentidos para uma única palavra.
- **Outras Relações Semânticas Lexicais:**  Relações entre palavras como sinonímia, antonímia, hiponímia, hiperonímia, meronímia e holonímia. Contexto:  Exploração de diferentes relações semânticas entre palavras.
- **Representação de Contexto para Desambiguização:**  Uso de bag-of-words e recursos de colocação para representar o contexto de uma palavra ambígua. Contexto:  Criando recursos para classificar o sentido das palavras baseado em seu contexto.
- **Algoritmo Lesk:**  Seleção do sentido de palavra baseado na sobreposição entre sua definição no dicionário e o contexto local. Contexto: Abordagem baseada em léxico para desambiguização sem aprendizado de máquina.
- **Concordância Semântica e Dados Rotulados:**  Uso de corpora rotulados (SemCor) para aprendizado supervisionado de desambiguização de sentidos. Contexto:  Necessidade de dados rotulados para aplicar aprendizado supervisionado ao problema.
- **Métodos Não Supervisionados e Semi-Supervisionados:**  Importância de métodos não supervisionados e semi-supervisionados para lidar com a escassez de dados rotulados. Contexto:  Abordagem para lidar com a escassez de dados rotulados na desambiguização.

**4.3 Decisões de Design para Classificação de Texto**

**4.3.1 O que é uma palavra?**

- 
- **Tokenização:**  Conversão de texto de sequência de caracteres para sequência de tokens de palavras. Contexto:  Primeira etapa no pré-processamento de texto para classificação.
- **Normalização de Texto:**  Transformações de strings que removem distinções irrelevantes para aplicações posteriores (conversão de caixa, normalização de números e datas). Contexto:  Métodos para limpar e padronizar o texto antes da classificação.
- **Stemming e Lematização:**  Eliminação de afixos flexionais (stemming) e identificação do lema subjacente (lemmatização). Contexto:  Técnicas para reduzir a dimensionalidade do espaço de features, agrupando palavras com significado similar.
- **Cobertura de Tokens:**  Trade-off entre cobertura de tokens e tamanho do vocabulário. Contexto:  Otimização da seleção de palavras para a classificação.

**4.3.2 Quantas palavras?**

- 
- **Limitação do Tamanho do Vetor de Características:**  Métodos para limitar o tamanho do vocabulário (palavras mais frequentes, remoção de stop words, feature hashing). Contexto:  Otimização do tamanho do modelo para eficiência e generalização.

**4.3.3 Contagem ou Binário?**

- 
- **Representação Binária vs. Contagem de Palavras:**  Comparação entre representações binárias (presença/ausência) e contagens de palavras em vetores de características. Contexto:  Comparação entre duas abordagens de representação de features.

**4.4 Avaliando Classificadores**

- 
- **Precisão, Recall e F-Measure:**  Métricas para avaliar o desempenho de classificadores, especialmente em situações de desequilíbrio de classes. Contexto:  Métricas que capturam a precisão e o poder de detecção do modelo.
- **Métricas Livres de Limiar:**  Curva ROC e AUC como métricas independentes do limiar de classificação. Contexto:  Métricas que permitem avaliar o modelo sem definir um limiar de decisão.
- **Comparação de Classificadores e Significância Estatística:**  Testes estatísticos (teste binomial, teste de randomização) para determinar a significância estatística de diferenças entre classificadores. Contexto:  Métodos para determinar se as diferenças de performance entre modelos são significativas estatisticamente.
- **Múltiplas Comparações:**  Ajustes de p-valor (Bonferroni, Benjamini-Hochberg) para lidar com múltiplas comparações. Contexto:  Métodos para ajustar a significância estatística quando múltiplos testes são realizados.

**4.5 Construindo Conjuntos de Dados**

- 
- **Metadados como Rótulos:**  Utilização de metadados existentes para obter rótulos de treinamento. Contexto:  Método para criar conjuntos de dados rotulados de forma mais eficiente.
- **Rótulos Manuais:**  Processo de rotulagem manual de dados, incluindo a definição de um protocolo de anotação, a medição de concordância entre anotadores (Kappa de Cohen, Alfa de Krippendorff), e a utilização de crowdsourcing. Contexto:  Método tradicional e desafiador para construção de conjuntos de dados rotulados.

Espero que esta lista seja útil para seus estudos.  Lembre-se que esta é uma análise teórica dos conceitos apresentados, sem a inclusão de passos práticos ou de código.