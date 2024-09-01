## Recapitulação de Modelos Baseados em Verossimilhança: De Autorregressivos a Fluxos Normalizadores

<image: Um diagrama mostrando a progressão e relações entre modelos autorregressivos, VAEs e fluxos normalizadores, enfatizando sua base comum em treinamento baseado em verossimilhança>

### Introdução

No domínio da modelagem generativa, abordagens baseadas em verossimilhança têm sido a pedra angular para muitas famílias de modelos poderosos. Esta recapitulação aprofunda-se nos conceitos-chave, pontos fortes e limitações dos modelos autorregressivos, Autoencoders Variacionais (VAEs) e fluxos normalizadores. Ao entender essas abordagens fundamentais, podemos melhor apreciar a motivação por trás dos métodos independentes de verossimilhança, como as Redes Adversariais Generativas (GANs) [1].

### Conceitos Fundamentais

| Conceito                                | Explicação                                                   |
| --------------------------------------- | ------------------------------------------------------------ |
| **Estimação de Máxima Verossimilhança** | O princípio de maximizar a verossimilhança dos dados observados sob os parâmetros do modelo. Forma a base para o treinamento da maioria dos modelos generativos [1]. |
| **Modelos de Variáveis Latentes**       | Modelos que introduzem variáveis não observadas (latentes) para capturar distribuições de dados complexas. VAEs são um exemplo principal [2]. |
| **Estimação de Densidade Tratável**     | A capacidade de computar diretamente a densidade de probabilidade dos pontos de dados sob o modelo. Esta é uma característica-chave dos fluxos normalizadores [3]. |

> ⚠️ **Nota Importante**: Embora os modelos baseados em verossimilhança tenham sido altamente bem-sucedidos, eles podem nem sempre se correlacionar perfeitamente com a qualidade das amostras, especialmente em espaços de alta dimensão [1].

### Modelos Autorregressivos

<image: Uma representação gráfica de um modelo autorregressivo, mostrando como cada variável depende das variáveis anteriores na sequência>

Modelos autorregressivos decompõem a probabilidade conjunta de um ponto de dados em um produto de probabilidades condicionais:

$$
p(x) = \prod_{i=1}^{n} p(x_i | x_{1:i-1})
$$

Onde $x_i$ representa a i-ésima dimensão do ponto de dados [1].

#### 👍 Vantagens
* Cálculo de verossimilhança tratável
* Modelagem poderosa de dados sequenciais

#### 👎 Desvantagens
* A geração pode ser lenta devido à natureza sequencial
* Pode ter dificuldades com dependências de longo alcance

#### Questões Técnicas/Teóricas

1. Como a escolha da ordenação em modelos autorregressivos afeta seu desempenho e quais estratégias existem para mitigar a dependência de ordem?
2. Em quais cenários um modelo autorregressivo pode ser preferível a um VAE ou fluxo normalizador, e por quê?

### Autoencoders Variacionais (VAEs)

<image: Um diagrama da arquitetura de um VAE, destacando os componentes do codificador, espaço latente e decodificador>

VAEs introduzem um modelo de variável latente com uma posterior aproximada $q_\phi(z|x)$ e um modelo generativo $p_\theta(x|z)$. O objetivo é maximizar o limite inferior da evidência (ELBO) [2]:

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) || p(z))
$$

Onde:
- $q_\phi(z|x)$ é o codificador (modelo de inferência)
- $p_\theta(x|z)$ é o decodificador (modelo generativo)
- $p(z)$ é a distribuição prior sobre variáveis latentes

> ✔️ **Destaque**: O ELBO fornece um objetivo de otimização tratável que equilibra a qualidade da reconstrução com a regularização do espaço latente.

#### 👍 Vantagens
* Aprende representações latentes significativas
* Permite amostragem eficiente e interpolação no espaço latente

#### 👎 Desvantagens
* Frequentemente produz amostras borradas devido ao uso de modelos de verossimilhança simples (por exemplo, Gaussiano)
* A posterior verdadeira pode ser mais complexa do que a posterior aproximada pode capturar

#### Questões Técnicas/Teóricas

1. Como a escolha da distribuição prior $p(z)$ em um VAE afeta as representações aprendidas e as amostras geradas?
2. Descreva um cenário onde o problema de colapso posterior pode ocorrer em VAEs e proponha uma solução potencial.

### Fluxos Normalizadores

<image: Uma série de transformações invertíveis ilustrando o conceito de fluxos normalizadores, transformando uma distribuição base simples em uma distribuição alvo complexa>

Fluxos normalizadores definem uma sequência de transformações invertíveis $f_1, ..., f_K$ que mapeiam uma distribuição base simples $p_Z(z)$ para uma distribuição alvo complexa $p_X(x)$ [3]:

$$
x = f_K \circ ... \circ f_1(z), \quad z \sim p_Z(z)
$$

A fórmula de mudança de variáveis permite o cálculo exato da verossimilhança:

$$
\log p_X(x) = \log p_Z(z) - \sum_{k=1}^K \log |\det \frac{\partial f_k}{\partial z_{k-1}}|
$$

> ❗ **Ponto de Atenção**: A invertibilidade das transformações é crucial tanto para amostragem quanto para cálculo de verossimilhança em fluxos normalizadores.

#### 👍 Vantagens
* Cálculo exato de verossimilhança
* Processo de amostragem eficiente
* Modelagem expressiva de distribuições complexas

#### 👎 Desvantagens
* A restrição de transformações invertíveis pode limitar a expressividade
* Pode requerer um grande número de transformações para distribuições complexas

#### Questões Técnicas/Teóricas

1. Como a escolha da distribuição base e das funções de transformação em um modelo de fluxo normalizador impacta sua expressividade e eficiência computacional?
2. Descreva um cenário onde um modelo de fluxo normalizador pode ser preferível a um VAE ou GAN, considerando aspectos teóricos e práticos.

### Limitações dos Modelos Baseados em Verossimilhança

Embora os modelos baseados em verossimilhança tenham mostrado notável sucesso, eles enfrentam vários desafios:

1. **Qualidade da Amostra vs. Verossimilhança**: Alta verossimilhança nem sempre corresponde a alta qualidade de amostra, especialmente em espaços de alta dimensão [1].

2. **Complexidade Computacional**: O cálculo exato de verossimilhança pode ser computacionalmente caro para modelos complexos [3].

3. **Cobertura de Modos**: Esses modelos podem ter dificuldades em capturar todos os modos de distribuições complexas e multimodais [2].

4. **Maldição da Dimensionalidade**: À medida que a dimensionalidade dos dados aumenta, o volume do espaço cresce exponencialmente, tornando desafiador estimar densidades com precisão [1].

> 💡 **Insight Chave**: As limitações dos modelos baseados em verossimilhança motivam a exploração de objetivos de treinamento alternativos e arquiteturas de modelo, como aquelas empregadas em GANs e outras abordagens independentes de verossimilhança [1].

### Conclusão

Modelos baseados em verossimilhança, incluindo modelos autorregressivos, VAEs e fluxos normalizadores, têm sido instrumentais no avanço do campo da modelagem generativa. Cada abordagem oferece pontos fortes únicos e enfrenta desafios específicos. Entender esses modelos fundamentais e suas limitações fornece um contexto crucial para apreciar as motivações por trás de abordagens mais recentes e independentes de verossimilhança, como as GANs. À medida que avançamos no campo da modelagem generativa, é essencial considerar tanto os pontos fortes dos métodos baseados em verossimilhança quanto os benefícios potenciais de objetivos e arquiteturas alternativos.

### Questões Avançadas

1. Compare e contraste as representações do espaço latente aprendidas por VAEs e os espaços latentes implícitos em GANs. Como essas diferenças impactam tarefas como interpolação e desemaranhamento?

2. Proponha um modelo híbrido que combine elementos de modelos autorregressivos, VAEs e fluxos normalizadores. Descreva suas potenciais vantagens e os desafios no treinamento de tal modelo.

3. Discuta os trade-offs entre o cálculo exato de verossimilhança (como em fluxos normalizadores) e métodos aproximados (como em VAEs) no contexto de expressividade do modelo e eficiência computacional. Como esses trade-offs podem influenciar a escolha do modelo para diferentes tipos de dados e aplicações?

### Referências

[1] "Agora passamos para outra família de modelos generativos chamados redes adversariais generativas (GANs). As GANs são únicas de todas as outras famílias de modelos que vimos até agora, como modelos autorregressivos, VAEs e modelos de fluxo normalizador, porque não as treinamos usando máxima verossimilhança." (Trecho de Stanford Notes)

[2] "Uma vez que discutimos modelos de variáveis latentes, afirmamos que eles naturalmente definem um processo generativo primeiro amostrando latentes z ∼ p(z) e então gerando observáveis x ∼ pθ (x|z). Isso é bom! No entanto, o problema aparece quando começamos a pensar sobre o treinamento. Para ser mais preciso, o objetivo do treinamento é um problema." (Trecho de Deep Generative Models)

[3] "Como mencionamos já na seção sobre VAEs (veja Sect. 4.3), a parte problemática é calcular a integral porque não é analiticamente tratável a menos que todas as distribuições sejam Gaussianas e a dependência entre x e z seja linear." (Trecho de Deep Generative Models)