## Finetuning de Large Language Models para Classificação

![image-20240905182314233](C:\Users\diego.rodrigues\AppData\Roaming\Typora\typora-user-images\image-20240905182314233.png)

### Introdução

O finetuning de Large Language Models (LLMs) para tarefas específicas tem se tornado uma prática essencial no campo do Processamento de Linguagem Natural (NLP). Este resumo aborda os aspectos técnicos e matemáticos envolvidos no processo de adaptar um LLM pré-treinado, especificamente um modelo GPT, para uma tarefa de classificação de texto. Focamos em estratégias de preparação de dados, modificação da arquitetura do modelo, técnicas de treinamento e avaliação de desempenho. [1]

### Conceitos Fundamentais

| Conceito                   | Explicação                                                   |
| -------------------------- | ------------------------------------------------------------ |
| **Finetuning**             | Processo de ajuste fino de um modelo pré-treinado para uma tarefa específica, mantendo a maior parte dos pesos originais. [2] |
| **Transfer Learning**      | Técnica que utiliza conhecimento adquirido em uma tarefa para melhorar o desempenho em outra tarefa relacionada. [3] |
| **Classificação de Texto** | Tarefa de atribuir uma ou mais categorias predefinidas a um texto de entrada, baseando-se em seu conteúdo. [4] |

> ⚠️ **Nota Importante**: O finetuning de LLMs para classificação envolve um delicado equilíbrio entre aproveitar o conhecimento pré-treinado e adaptar-se à nova tarefa sem perder generalização.

### 6.3 Criando Data Loaders

#### Lidando com Inputs de Comprimento Variável

Ao trabalhar com textos de comprimento variável em tarefas de classificação, dois principais desafios surgem: a necessidade de processar entradas em lotes (batches) e a limitação do modelo em lidar com sequências de comprimento fixo. Duas estratégias principais são empregadas para abordar esses desafios [5]:

1. **Truncamento**: Limita o comprimento das sequências a um valor máximo predefinido.
2. **Padding**: Adiciona tokens especiais para preencher sequências mais curtas até um comprimento comum.

A escolha entre truncamento e padding depende da natureza dos dados e da tarefa específica:

👍 **Vantagens do Truncamento**:
* Computacionalmente mais eficiente
* Evita o processamento de informações potencialmente irrelevantes

👎 **Desvantagens do Truncamento**:
* Pode resultar em perda de informações importantes em textos longos

👍 **Vantagens do Padding**:
* Preserva todas as informações originais
* Permite processamento de sequências de qualquer comprimento

👎 **Desvantagens do Padding**:
* Aumenta o custo computacional
* Pode introduzir ruído no processamento de tokens de padding

No contexto do finetuning de LLMs para classificação, a abordagem de padding é frequentemente preferida, pois preserva informações potencialmente cruciais para a tarefa de classificação. [6]

#### Implementação de PyTorch Dataset e DataLoader

A implementação eficiente de um pipeline de dados para finetuning envolve o uso das classes `Dataset` e `DataLoader` do PyTorch. Vamos analisar uma implementação típica:

```python
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        self.max_length = max_length or self._longest_encoded_length()
        self.encoded_texts = [
            encoded_text[:self.max_length] + 
            [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
```

Esta implementação da classe `SpamDataset` realiza várias operações cruciais [7]:

1. Carrega os dados de um arquivo CSV.
2. Tokeniza os textos usando um tokenizador específico (geralmente o mesmo usado no pré-treinamento do LLM).
3. Determina o comprimento máximo das sequências.
4. Realiza padding ou truncamento das sequências para um comprimento uniforme.
5. Converte os dados em tensores PyTorch.

> ✔️ **Destaque**: A determinação dinâmica do comprimento máximo (`_longest_encoded_length`) permite adaptar o dataset a diferentes conjuntos de dados sem hardcoding.

Para criar um `DataLoader` eficiente, usa-se:

```python
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    drop_last=True
)
```

O `DataLoader` oferece várias vantagens [8]:
- Carregamento em lotes (batching) automático
- Shuffling dos dados para melhor generalização
- Paralelização do carregamento de dados com `num_workers`
- Opção de descartar o último lote incompleto com `drop_last`

#### Técnicas de Attention Masking para Sequências com Padding

Ao trabalhar com sequências padded, é crucial implementar attention masking para evitar que o modelo atenda a tokens de padding. Isso é tipicamente feito criando uma máscara de atenção que atribui peso zero aos tokens de padding [9]:

```python
def create_attention_mask(input_ids, pad_token_id=50256):
    return (input_ids != pad_token_id).float()
```

Esta máscara é então aplicada durante o cálculo da atenção no modelo:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Onde $M$ é a máscara de atenção, tipicamente com valores muito negativos (e.g., -10000) nos locais correspondentes aos tokens de padding.

#### Perguntas Técnicas/Teóricas

1. Como a escolha entre truncamento e padding pode afetar o desempenho de um modelo de classificação de texto? Considere cenários com textos de comprimentos muito variados.

2. Explique como a implementação de attention masking ajuda a lidar com sequências de comprimento variável em modelos baseados em atenção. Quais seriam as consequências de não usar masking em sequências com padding?

3. Discuta as implicações de performance e memória ao escolher diferentes valores para `batch_size` e `num_workers` no DataLoader do PyTorch. Como você otimizaria esses parâmetros para um dataset muito grande?

Certamente. Vamos prosseguir com o próximo tópico do nosso resumo aprofundado.

### 6.4 Inicializando um modelo com pesos pré-treinados

<image: Um diagrama mostrando a transferência de pesos de um modelo GPT pré-treinado para um novo modelo, com setas indicando quais camadas são transferidas e quais são inicializadas aleatoriamente>

#### Princípios de Transfer Learning em NLP

Transfer Learning é uma técnica fundamental no campo de NLP, especialmente quando se trabalha com Large Language Models. O princípio básico envolve a utilização de conhecimento adquirido em uma tarefa (geralmente mais geral e com grande volume de dados) para melhorar o desempenho em outra tarefa relacionada (geralmente mais específica e com menos dados disponíveis). [10]

No contexto de LLMs, o processo tipicamente segue estas etapas:

1. **Pré-treinamento**: O modelo é treinado em uma tarefa de modelagem de linguagem em um corpus grande e diversificado.
2. **Transferência**: Os pesos aprendidos são transferidos para um novo modelo com arquitetura similar ou idêntica.
3. **Adaptação**: O novo modelo é então ajustado (finetuned) para a tarefa específica desejada.

> ✔️ **Destaque**: O sucesso do transfer learning em NLP se deve à capacidade dos modelos pré-treinados de capturar representações linguísticas ricas e generalizáveis.

#### Carregando e Verificando Pesos Pré-treinados do Modelo GPT

O processo de carregar pesos pré-treinados em um modelo GPT envolve várias etapas técnicas. Vamos examinar um exemplo típico:

```python
from gpt_download import download_and_load_gpt2
from chapter05 import GPTModel, load_weights_into_gpt

# Configuração do modelo
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# Download e carregamento dos pesos
model_size = "124M"
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# Inicialização do modelo
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()
```

Este código realiza as seguintes operações cruciais [11]:

1. Define a configuração base do modelo, incluindo tamanho do vocabulário, comprimento do contexto, dimensões dos embeddings, etc.
2. Baixa os pesos pré-treinados do modelo GPT-2.
3. Inicializa uma nova instância do modelo com a configuração especificada.
4. Carrega os pesos pré-treinados no modelo inicializado.
5. Coloca o modelo em modo de avaliação.

> ⚠️ **Nota Importante**: É crucial garantir que a arquitetura do modelo inicializado seja compatível com os pesos pré-treinados. Discrepâncias podem levar a erros ou comportamentos inesperados.

Para verificar se os pesos foram carregados corretamente, pode-se realizar uma inferência simples:

```python
from chapter04 import generate_text_simple
from chapter05 import text_to_token_ids, token_ids_to_text

text = "Every effort moves"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
```

Se o modelo gerar texto coerente, é um bom indicador de que os pesos foram carregados corretamente. [12]

#### Desafios no Uso de Modelos Pré-treinados para Tarefas Downstream sem Finetuning

Utilizar modelos pré-treinados diretamente para tarefas downstream sem finetuning apresenta vários desafios [13]:

1. **Desalinhamento de Tarefas**: O modelo pré-treinado é otimizado para modelagem de linguagem, não para classificação ou outras tarefas específicas.

2. **Viés do Domínio**: O corpus de pré-treinamento pode não representar adequadamente o domínio da tarefa alvo.

3. **Ausência de Camada de Saída Específica**: Modelos como GPT não possuem naturalmente uma camada de saída adequada para classificação.

4. **Overfitting Potencial**: Sem finetuning, o modelo pode se apegar excessivamente a padrões do pré-treinamento não relevantes para a tarefa alvo.

5. **Ineficiência Computacional**: Usar o modelo completo sem adaptação pode ser computacionalmente ineficiente para tarefas simples.

Para ilustrar esses desafios, podemos tentar usar o modelo pré-treinado diretamente para classificação:

```python
text = ("Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))
```

O resultado provavelmente será incoerente ou não relacionado à tarefa de classificação de spam, demonstrando a necessidade de finetuning. [14]

#### Perguntas Técnicas/Teóricas

1. Explique como o princípio de "catastrophic forgetting" pode afetar o processo de finetuning de um LLM pré-treinado. Que estratégias podem ser empregadas para mitigar esse problema?
2. Considere um cenário onde você tem um LLM pré-treinado em textos em inglês e deseja adaptá-lo para classificação de sentimentos em português. Quais desafios específicos você antecipa e como os abordaria?
3. Discuta as implicações éticas e de viés ao usar modelos pré-treinados em grandes corpora de texto da internet para tarefas downstream específicas. Como você avaliaria e mitigaria potenciais vieses?

Certamente. Vamos prosseguir com o próximo tópico do nosso resumo aprofundado.

### 6.5 Adicionando uma Cabeça de Classificação

<image: Um diagrama detalhado mostrando a arquitetura de um modelo GPT com uma nova camada de classificação adicionada no topo, destacando as camadas congeladas e as que serão finetuned>

#### Modificando a Arquitetura do LLM Pré-treinado para Tarefas de Classificação

A adaptação de um LLM pré-treinado para uma tarefa de classificação envolve modificações significativas na arquitetura do modelo, principalmente na camada de saída. Este processo é crucial para transformar um modelo generativo em um classificador discriminativo. [15]

O processo típico de modificação inclui:

1. **Remoção da Camada de Saída Original**: A camada final do modelo GPT, que mapeia para o tamanho do vocabulário, é removida.
2. **Adição de uma Nova Camada de Classificação**: Uma nova camada linear é adicionada, mapeando as representações ocultas para o número de classes desejado.
3. **Ajuste das Dimensões**: Garantir que a nova camada de classificação seja compatível com as dimensões das representações ocultas do modelo.

Vamos examinar uma implementação típica:

```python
import torch

num_classes = 2  # Para classificação binária (e.g., spam vs. não-spam)
hidden_size = model.out_head.in_features

# Remover a camada de saída original
del model.out_head

# Adicionar nova camada de classificação
model.out_head = torch.nn.Linear(hidden_size, num_classes)

# Inicializar os pesos da nova camada
torch.nn.init.normal_(model.out_head.weight, std=0.02)
torch.nn.init.zeros_(model.out_head.bias)
```

> ✔️ **Destaque**: A inicialização adequada dos pesos da nova camada é crucial para um finetuning eficiente. Valores muito grandes ou muito pequenos podem levar a problemas de convergência.

#### Substituição da Camada de Saída: De Vocabulário para Classes

A substituição da camada de saída é um passo crítico na adaptação do LLM para classificação. Essa mudança altera fundamentalmente a natureza do modelo [16]:

1. **Camada Original**: 
   - Dimensões: `(hidden_size, vocab_size)` (e.g., `(768, 50257)` para GPT-2 pequeno)
   - Função: Gerar probabilidades para cada token no vocabulário

2. **Nova Camada de Classificação**:
   - Dimensões: `(hidden_size, num_classes)` (e.g., `(768, 2)` para classificação binária)
   - Função: Gerar logits para cada classe

Esta mudança pode ser representada matematicamente como:

$$
\text{output} = \text{softmax}(W_{new} \cdot h + b_{new})
$$

Onde:
- $h$ é o vetor de representação oculta (hidden state)
- $W_{new}$ é a matriz de pesos da nova camada (dimensões: `hidden_size × num_classes`)
- $b_{new}$ é o vetor de bias da nova camada

> ⚠️ **Nota Importante**: Ao substituir a camada de saída, é essencial ajustar a função de perda e as métricas de avaliação para refletir a tarefa de classificação, em vez da tarefa de modelagem de linguagem original.

#### Técnicas de Congelamento Seletivo de Camadas para Finetuning Eficiente

O congelamento seletivo de camadas é uma técnica crucial para o finetuning eficiente de LLMs. Esta abordagem permite adaptar o modelo para a nova tarefa enquanto preserva o conhecimento geral adquirido durante o pré-treinamento. [17]

Principais considerações:

1. **Camadas Inferiores**: Tendem a capturar características mais gerais da linguagem.
2. **Camadas Superiores**: Geralmente são mais específicas à tarefa e ao domínio.

Uma estratégia comum é congelar as camadas inferiores e permitir o finetuning apenas das camadas superiores e da nova camada de classificação:

```python
# Congelar todas as camadas
for param in model.parameters():
    param.requires_grad = False

# Descongelar as últimas n camadas
n_layers_to_finetune = 3
for i in range(n_layers_to_finetune):
    for param in model.trf_blocks[-(i+1)].parameters():
        param.requires_grad = True

# Descongelar a camada de classificação
for param in model.out_head.parameters():
    param.requires_grad = True
```

Esta abordagem oferece várias vantagens:
- Reduz o risco de overfitting em datasets pequenos
- Diminui o tempo de treinamento e o uso de memória
- Preserva o conhecimento geral adquirido no pré-treinamento

> 💡 **Dica**: O número ideal de camadas a serem finetuned pode variar dependendo do tamanho do dataset e da similaridade entre a tarefa de pré-treinamento e a tarefa alvo. Experimente com diferentes configurações para otimizar o desempenho.

#### Atenção Causal e suas Implicações para Tarefas de Classificação

A arquitetura GPT utiliza atenção causal, o que significa que cada token só pode atender a tokens anteriores na sequência. Isso apresenta desafios únicos para tarefas de classificação [18]:

1. **Representação Final**: Para classificação, geralmente usamos a representação do último token como entrada para a camada de classificação.

2. **Informação Contextual**: O último token tem acesso a toda a informação da sequência, tornando-o ideal para classificação.

3. **Padding e Masking**: É crucial implementar masking adequado para lidar com sequências de comprimento variável e padding.

A implementação da atenção causal pode ser representada matematicamente como:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Onde $M$ é uma matriz de masking triangular inferior:

$$
M_{ij} = 
\begin{cases} 
0 & \text{se } i \geq j \\
-\infty & \text{caso contrário}
\end{cases}
$$

> ⚠️ **Atenção**: Ao adaptar um modelo GPT para classificação, é importante manter a estrutura de atenção causal para preservar a capacidade do modelo de processar sequências de forma coerente.

#### Perguntas Técnicas/Teóricas

1. Como a escolha do número de camadas a serem finetuned afeta o trade-off entre a preservação do conhecimento pré-treinado e a adaptação à nova tarefa? Discuta cenários em que você optaria por finetunear mais ou menos camadas.

2. Explique como a atenção causal pode ser tanto uma vantagem quanto uma limitação em tarefas de classificação de texto. Como você poderia adaptar o modelo para mitigar possíveis limitações?

3. Considere um cenário onde você está finetuning um LLM para classificação de textos muito longos (por exemplo, artigos científicos inteiros). Como você lidaria com as limitações de comprimento de sequência do modelo GPT neste caso?

Certamente. Vamos prosseguir com o próximo tópico do nosso resumo aprofundado.

### 6.6 Calculando a Loss de Classificação e Acurácia

<image: Um diagrama mostrando o fluxo de dados através do modelo, culminando na camada de classificação, com visualizações da função de perda cross-entropy e do cálculo de acurácia>

#### Cross-Entropy Loss para Classificação Multi-classe

A função de perda cross-entropy é fundamental para treinar modelos de classificação, incluindo LLMs finetuned. Esta função quantifica a diferença entre a distribuição de probabilidade prevista pelo modelo e a distribuição real (one-hot encoded) das classes. [19]

Para classificação multi-classe, a cross-entropy loss é definida como:

$$
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

Onde:
- $C$ é o número de classes
- $y_i$ é o valor real (0 ou 1) para a classe $i$
- $\hat{y}_i$ é a probabilidade prevista para a classe $i$

No contexto de finetuning de LLMs para classificação, implementamos esta loss da seguinte forma:

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits do último token de saída
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

> ✔️ **Destaque**: A função `cross_entropy` do PyTorch combina o cálculo de softmax e log-likelihood, otimizando a performance e estabilidade numérica.

#### Ativação Softmax para Distribuição de Probabilidade sobre Classes

A função softmax é utilizada para converter os logits (saídas brutas da camada de classificação) em uma distribuição de probabilidade sobre as classes. Matematicamente, a softmax é definida como [20]:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^C e^{x_j}}
$$

Onde $x_i$ é o logit para a classe $i$ e $C$ é o número total de classes.

No contexto de nossa implementação:

```python
def get_probabilities(logits):
    return torch.softmax(logits, dim=-1)

# Uso
logits = model(input_batch)[:, -1, :]
probabilities = get_probabilities(logits)
```

A softmax garante que as probabilidades somem 1 e sejam não-negativas, propriedades essenciais para interpretação em classificação.

#### Métricas de Acurácia de Classificação e seu Cálculo

A acurácia é uma métrica fundamental para avaliar modelos de classificação. Ela representa a proporção de previsões corretas em relação ao total de previsões. [21]

Para classificação binária ou multi-classe, a acurácia é calculada como:

$$
\text{Acurácia} = \frac{\text{Número de previsões corretas}}{\text{Número total de previsões}}
$$

Implementação em PyTorch:

```python
def calc_accuracy_batch(logits, target_batch):
    predicted_labels = torch.argmax(logits, dim=-1)
    correct_predictions = (predicted_labels == target_batch).sum().item()
    return correct_predictions / len(target_batch)

# Uso
logits = model(input_batch)[:, -1, :]
accuracy = calc_accuracy_batch(logits, target_batch)
```

> ⚠️ **Nota Importante**: Embora a acurácia seja intuitiva, ela pode ser enganosa em datasets desbalanceados. Considere métricas adicionais como precisão, recall e F1-score para uma avaliação mais completa.

#### Técnicas de Avaliação de Performance por Batch e por Dataset

A avaliação de performance do modelo pode ser feita tanto por batch quanto por dataset completo. Cada abordagem tem suas vantagens [22]:

1. **Avaliação por Batch**:
   - Mais rápida e eficiente em memória
   - Útil para monitoramento durante o treinamento
   - Pode ser volátil devido à variabilidade entre batches

2. **Avaliação por Dataset**:
   - Fornece uma visão mais estável e abrangente do desempenho
   - Mais custosa computacionalmente
   - Essencial para avaliação final e comparação de modelos

Implementação de avaliação por dataset:

```python
def calc_accuracy_loader(data_loader, model, device):
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            correct_predictions += (predicted_labels == target_batch).sum().item()
            num_examples += target_batch.size(0)
    
    return correct_predictions / num_examples
```

> 💡 **Dica**: Durante o treinamento, alterne entre avaliações rápidas por batch e avaliações periódicas mais completas por dataset para balancear eficiência e confiabilidade.

#### Perguntas Técnicas/Teóricas

1. Em um cenário de classificação multi-classe altamente desbalanceado, como a cross-entropy loss pode ser problemática? Proponha e explique uma modificação na função de perda para lidar com este desbalanceamento.

2. Discuta as vantagens e desvantagens de usar a representação do último token versus uma representação agregada (como média ou max pooling) de todos os tokens para classificação em um modelo baseado em GPT. Como essa escolha pode afetar o desempenho em diferentes tipos de tarefas de classificação?

3. Explique como você implementaria uma técnica de early stopping baseada na acurácia de validação durante o finetuning de um LLM. Que considerações adicionais você teria ao aplicar early stopping em um modelo grande com muitos parâmetros?

Certamente. Vamos prosseguir com o último tópico do nosso resumo aprofundado.

### 6.7 Finetuning do Modelo em Dados Supervisionados

<image: Um diagrama de fluxo mostrando o processo iterativo de finetuning, incluindo forward pass, cálculo de loss, backpropagation, e atualização de pesos, com gráficos mostrando a evolução da loss e acurácia ao longo das épocas>

#### Algoritmos de Otimização para Finetuning: AdamW e Weight Decay

O finetuning eficiente de LLMs requer algoritmos de otimização robustos. O AdamW (Adam com Weight Decay desacoplado) é uma escolha popular devido à sua eficácia em lidar com problemas de otimização não-convexos e sua capacidade de adaptar as taxas de aprendizado para cada parâmetro. [23]

A atualização de parâmetros no AdamW é dada por:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{aligned}
$$

Onde:
- $m_t$, $v_t$ são as estimativas do primeiro e segundo momento
- $\beta_1$, $\beta_2$ são os fatores de decaimento para os momentos
- $\alpha$ é a taxa de aprendizado
- $\lambda$ é o fator de weight decay
- $g_t$ é o gradiente no tempo $t$

Implementação em PyTorch:

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
```

> ✔️ **Destaque**: O weight decay desacoplado no AdamW ajuda a prevenir overfitting, especialmente importante no finetuning de modelos grandes com datasets relativamente pequenos.

#### Estratégias de Seleção de Learning Rate para Finetuning

A escolha da learning rate (taxa de aprendizado) é crucial para o sucesso do finetuning. Algumas estratégias comuns incluem [24]:

1. **Learning Rate Constante**: Simples, mas pode ser subótima.
2. **Learning Rate Decay**: Diminui a LR ao longo do tempo, permitindo convergência fina.
3. **Cyclical Learning Rates**: Alterna entre valores altos e baixos, potencialmente escapando de mínimos locais.
4. **Learning Rate Warmup**: Começa com uma LR baixa e aumenta gradualmente, estabilizando o treinamento inicial.

Uma abordagem eficaz é o uso de learning rate warmup seguido de decay:

```python
from transformers import get_linear_schedule_with_warmup

num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

> 💡 **Dica**: Experimente com diferentes schedulers e parâmetros para encontrar a configuração ideal para seu caso específico.

#### Loops de Treinamento Baseados em Épocas e Critérios de Early Stopping

O treinamento baseado em épocas envolve iterar sobre todo o dataset múltiplas vezes. Um loop de treinamento típico para finetuning inclui [25]:

```python
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
```

Early stopping é uma técnica crucial para prevenir overfitting, parando o treinamento quando o desempenho no conjunto de validação para de melhorar.

> ⚠️ **Nota Importante**: Ajuste o critério de early stopping (por exemplo, paciência) com base no tamanho e complexidade do seu modelo e dataset.

#### Monitoramento de Métricas de Treinamento e Validação para Detectar Overfitting

O monitoramento contínuo das métricas de treinamento e validação é essencial para detectar overfitting e ajustar o processo de finetuning [26]. Métricas importantes incluem:

1. **Loss de Treinamento e Validação**: Divergência entre estas curvas indica overfitting.
2. **Acurácia de Treinamento e Validação**: Similar à loss, mas mais interpretável.
3. **Gradientes**: Normas de gradiente muito altas ou baixas podem indicar problemas.

Implementação de um logger básico:

```python
class TrainingLogger:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def log(self, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Acc')
        plt.plot(self.val_accuracies, label='Val Acc')
        plt.legend()
        plt.title('Accuracy over Epochs')
        
        plt.tight_layout()
        plt.show()
```

> 💡 **Dica**: Utilize ferramentas como TensorBoard ou Weights & Biases para visualização e monitoramento mais avançados durante o finetuning.

#### Perguntas Técnicas/Teóricas

1. Compare e contraste as vantagens e desvantagens de usar AdamW versus SGD com momentum para finetuning de LLMs. Em quais cenários você preferiria um sobre o outro?

2. Discuta as implicações de usar uma learning rate muito alta versus muito baixa durante o finetuning de um LLM. Como você determinaria a learning rate "ideal" para um projeto específico?

3. Proponha e explique uma estratégia para lidar com o problema de "catastrophic forgetting" durante o finetuning de um LLM para uma tarefa específica, mantendo sua capacidade de generalização em outras tarefas.

### Perguntas Avançadas

1. Considere um cenário onde você está finetuning um LLM para uma tarefa de classificação multi-rótulo com um grande número de classes possíveis (por exemplo, mais de 1000). Como você adaptaria a arquitetura do modelo e as técnicas de treinamento para lidar eficientemente com este cenário? Discuta possíveis trade-offs e otimizações.

2. Explique como você implementaria um sistema de finetuning contínuo para um LLM em produção, que se adapta a novos dados e tarefas ao longo do tempo sem perder o desempenho em tarefas anteriores. Quais desafios técnicos e práticos você antecipa e como os abordaria?

3. Discuta as implicações éticas e de viés ao finetunear LLMs para tarefas de classificação em domínios sensíveis (por exemplo, análise de sentimentos em contextos políticos ou detecção de discurso de ódio). Como você garantiria a robustez e fairness do modelo finetuned?

4. Proponha uma abordagem para combinar finetuning de modelo e aprendizado federado em um cenário onde os dados de treinamento são sensíveis e distribuídos entre múltiplos clientes. Quais desafios técnicos você enfrentaria e como os resolveria?

5. Compare e contraste as abordagens de finetuning de LLMs com técnicas mais recentes como Prompt Engineering e In-Context Learning. Em quais cenários cada abordagem seria mais apropriada, e como você poderia combinar essas técnicas para otimizar o desempenho em tarefas de classificação complexas?

### Conclusão

O finetuning de Large Language Models para tarefas de classificação é um processo complexo que envolve considerações cuidadosas em várias etapas, desde a preparação dos dados até a otimização e avaliação do modelo. As técnicas discutidas neste resumo, incluindo a modificação da arquitetura do modelo, estratégias de otimização e monitoramento de performance, são cruciais para adaptar efetivamente LLMs pré-treinados para tarefas específicas de classificação.

A compreensão profunda desses conceitos e técnicas permite aos praticantes de NLP e Machine Learning aproveitar o poder dos LLMs de forma eficiente, adaptando-os para uma variedade de aplicações práticas. À medida que o campo continua a evoluir, é essencial manter-se atualizado com as últimas pesquisas e melhores práticas, sempre considerando as implicações éticas e práticas do uso de modelos de linguagem poderosos em aplicações do mundo real.

### Referências

[1] "Finetuning de Large Language Models (LLMs) para tarefas específicas tem se tornado uma prática essencial no campo do Processamento de Linguagem Natural (NLP)." (Trecho do início do capítulo)

[2] "Finetuning: Processo de ajuste fino de um modelo pré-treinado para uma tarefa específica, mantendo a maior parte dos pesos originais." (Trecho da tabela de Conceitos Fundamentais)

[3] "Transfer Learning: Técnica que utiliza conhecimento adquirido em uma tarefa para melhorar o desempenho em outra tarefa relacionada." (Trecho da tabela de Conceitos Fundamentais)

[4] "Classificação de Texto: Tarefa de atribuir uma ou mais categorias predefinidas a um texto de entrada, baseando-se em seu conteúdo." (Trecho da tabela de Conceitos Fundamentais)

[5] "Ao trabalhar com textos de comprimento variável em tarefas de classificação, dois principais desafios surgem: a necessidade de processar entradas em lotes (batches) e a limitação do modelo em lidar com sequências de comprimento fixo." (Trecho da seção 6.3)

[6] "No contexto do finetuning de LLMs para classificação, a abordagem de padding é frequentemente preferida, pois preserva informações potencialmente cruciais para a tarefa de classificação." (Trecho da seção 6.3)

[7] "Esta implementação da classe `SpamDataset` realiza várias operações cruciais" (Trecho da seção 6.3)

[8] "O `DataLoader` oferece várias vantagens" (Trecho da seção 6.3)

[9] "Ao trabalhar com sequências padded, é crucial implementar attention masking para evitar que o modelo atenda a tokens de padding." (Trecho da seção 6.3)

[10] "Transfer Learning é uma técnica fundamental no campo de NLP, especialmente quando se trabalha com Large Language Models." (Trecho da seção 6.4)

[11] "Este código realiza as seguintes operações cruciais" (Trecho da seção 6.4)

[12] "Se o modelo gerar texto coerente, é um bom indicador de que os pesos foram carregados corretamente." (Trecho da seção 6.4)

[13] "Utilizar modelos pré-treinados diretamente para tarefas downstream sem finetuning apresenta vários desafios" (Trecho da seção 6.4)

[14] "O resultado provavelmente será incoerente ou não relacionado à tarefa de classificação de spam, demonstrando a necessidade de finetuning." (Trecho da seção 6.4)

[15] "A adaptação de um LLM pré-treinado para uma tarefa de classificação envolve modificações significativas na arquitetura do modelo, principalmente na camada de saída." (Trecho da seção 6.5)

[16] "A substituição da camada de saída é um passo crítico na adaptação do LLM para classificação. Essa mudança altera fundamentalmente a natureza do modelo" (Trecho da seção 6.5)

[17] "O congelamento seletivo de camadas é uma técnica crucial para o finetuning eficiente de LLMs." (Trecho da seção 6.5)

[18] "A arquitetura GPT utiliza atenção causal, o que significa que cada token só pode atender a tokens anteriores na sequência. Isso apresenta desafios únicos para tarefas de classificação" (Trecho da seção 6.5)

[19] "A função de perda cross-entropy é fundamental para treinar modelos de classificação, incluindo LLMs finetuned." (Trecho da seção 6.6)

[20] "A função softmax é utilizada para converter os logits (saídas brutas da camada de classificação) em uma distribuição de probabilidade sobre as classes." (Trecho da seção 6.6)

[21] "A acurácia é uma métrica fundamental para avaliar modelos de classificação. Ela representa a proporção de previsões corretas em relação ao total de previsões." (Trecho da seção 6.6)

[22] "A avaliação de performance do modelo pode ser feita tanto por batch quanto por dataset completo. Cada abordagem tem suas vantagens" (Trecho da seção 6.6)

[23] "O finetuning eficiente de LLMs requer algoritmos de otimização robustos. O AdamW (Adam com Weight Decay desacoplado) é uma escolha popular devido à sua eficácia em lidar com problemas de otimização não-convexos e sua capacidade de adaptar as taxas de aprendizado para cada parâmetro." (Trecho da seção 6.7)

[24] "A escolha da learning rate (taxa de