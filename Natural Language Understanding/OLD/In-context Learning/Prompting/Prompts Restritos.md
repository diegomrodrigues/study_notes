## Técnicas Avançadas de Prompting: Prompts Restritos

<image: Um diagrama mostrando diferentes tipos de prompts restritos, incluindo múltipla escolha, preenchimento de lacunas, e formatos estruturados como JSON. O diagrama deve enfatizar como essas restrições direcionam a saída do modelo de linguagem.>

### Introdução

Prompts restritos são uma técnica avançada de engenharia de prompts que visa melhorar o controle e a precisão das respostas geradas por modelos de linguagem de grande escala (LLMs). Esta abordagem envolve a especificação de formatos de resposta possíveis ou a restrição do espaço de saída diretamente no prompt [1]. Ao limitar as opções de resposta do modelo, os prompts restritos podem aumentar significativamente a confiabilidade e a utilidade das saídas do LLM em várias aplicações, desde classificação até geração de conteúdo estruturado.

> ⚠️ **Nota Importante**: A eficácia dos prompts restritos depende fortemente da capacidade do modelo de linguagem em compreender e seguir instruções complexas. Modelos mais avançados tendem a responder melhor a esse tipo de orientação [2].

### Conceitos Fundamentais

| Conceito                  | Explicação                                                   |
| ------------------------- | ------------------------------------------------------------ |
| **Prompt Restrito**       | Uma técnica de prompting que limita explicitamente o espaço de respostas possíveis do modelo, fornecendo um formato ou estrutura específica para a saída desejada [1]. |
| **Espaço de Saída**       | ==O conjunto de todas as respostas possíveis que um modelo de linguagem pode gerar. Prompts restritos visam reduzir este espaço para melhorar a precisão [3].== |
| **Fidelidade ao Formato** | A capacidade do modelo de aderir consistentemente ao formato de saída especificado no prompt [4]. |

### Tipos de Prompts Restritos

<image: Uma árvore de decisão mostrando diferentes tipos de prompts restritos e quando usar cada um, incluindo múltipla escolha para classificação, preenchimento de lacunas para extração de informações, e formatos estruturados para dados complexos.>

1. **Múltipla Escolha**
   - Descrição: Limita as respostas do modelo a um conjunto predefinido de opções.
   - Exemplo de prompt:
     ```
     Classifique o sentimento do seguinte texto como positivo, negativo ou neutro:
     "O filme foi surpreendentemente bom, superando minhas expectativas."
     Opções: (A) Positivo, (B) Negativo, (C) Neutro
     Resposta:
     ```
   - Vantagens: Ideal para tarefas de classificação, reduz drasticamente o espaço de saída [5].

2. **Preenchimento de Lacunas**
   - Descrição: Solicita ao modelo que complete informações específicas em um template.
   - Exemplo de prompt:
     ```
     Complete as informações faltantes:
     Nome da empresa: ___________
     Ano de fundação: ___________
     Setor de atuação: ___________
     ```
   - Vantagens: Útil para extração de informações estruturadas a partir de texto não estruturado [6].

3. **Formatos Estruturados (JSON, XML)**
   - Descrição: Requer que o modelo gere saídas em um formato de dados específico.
   - Exemplo de prompt:
     ```
     Gere um objeto JSON com as seguintes informações sobre um carro:
     {
       "marca": "",
       "modelo": "",
       "ano": 0,
       "características": []
     }
     ```
   - Vantagens: Facilita a integração com sistemas que esperam dados estruturados [7].

> ✔️ **Destaque**: Prompts restritos podem melhorar significativamente a precisão e utilidade das respostas do modelo em tarefas específicas, especialmente quando combinados com instruções claras e exemplos [8].

### Implementação de Prompts Restritos

Para implementar prompts restritos de forma eficaz, considere as seguintes etapas:

1. **Definição clara do formato**:
   - Especifique explicitamente a estrutura ou as opções de resposta desejadas.
   - Use delimitadores claros para separar instruções, contexto e o espaço de resposta.

2. **Fornecimento de exemplos**:
   - Inclua demonstrações de respostas corretamente formatadas no prompt.
   - Utilize few-shot prompting para melhorar a compreensão do modelo sobre o formato desejado [9].

3. **Validação da saída**:
   - Implemente verificações para garantir que a resposta do modelo adere ao formato especificado.
   - Considere o uso de parsers ou expressões regulares para validar estruturas mais complexas.

```python
import re
import json

def validate_json_output(response):
    try:
        json_obj = json.loads(response)
        return json_obj
    except json.JSONDecodeError:
        return None

def extract_multiple_choice(response):
    pattern = r'\(([A-C])\)'
    match = re.search(pattern, response)
    return match.group(1) if match else None

# Exemplo de uso
json_prompt = "Gere um objeto JSON com nome e idade de uma pessoa."
response = llm_model.generate(json_prompt)
valid_json = validate_json_output(response)

mc_prompt = "Classifique o sentimento (A) Positivo, (B) Negativo, (C) Neutro"
response = llm_model.generate(mc_prompt)
choice = extract_multiple_choice(response)
```

> ❗ **Ponto de Atenção**: A implementação de prompts restritos pode requerer ajustes iterativos para encontrar o equilíbrio ideal entre restrição e flexibilidade, dependendo da tarefa específica e do modelo utilizado [10].

#### Questões Técnicas/Teóricas

1. Como a especificação de um formato de saída estruturado no prompt pode afetar a perplexidade do modelo durante a geração?
2. Quais são as considerações éticas ao usar prompts restritos para orientar as respostas de um LLM em cenários de tomada de decisão?

### Vantagens e Desvantagens dos Prompts Restritos

| 👍 Vantagens                                            | 👎 Desvantagens                                               |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| Aumenta a precisão e consistência das respostas [11]   | Pode limitar a criatividade e flexibilidade do modelo [12]   |
| Facilita a integração com sistemas automatizados [13]  | Requer cuidado na formulação para evitar vieses indesejados [14] |
| Melhora a interpretabilidade das saídas do modelo [15] | Pode não ser adequado para tarefas que exigem respostas mais abertas ou exploratórias [16] |

### Aplicações Avançadas de Prompts Restritos

1. **Geração de Código Estruturado**:
   Prompts restritos podem ser usados para gerar snippets de código em linguagens específicas, garantindo a aderência a padrões de sintaxe.

   ```python
   def generate_python_function(prompt):
       function_template = """
       def {function_name}({parameters}):
           \"\"\"
           {docstring}
           \"\"\"
           # Implementação aqui
           {implementation}
       """
       # Uso do LLM para preencher o template
       return llm_model.generate(prompt + "\n" + function_template)
   ```

2. **Extração de Entidades Nomeadas**:
   Utilização de prompts restritos para extrair informações específicas de textos não estruturados.

   ```python
   def extract_named_entities(text):
       prompt = f"""
       Extraia as seguintes entidades do texto, se presentes:
       - Pessoa: ___________
       - Organização: ___________
       - Local: ___________
       
       Texto: {text}
       """
       return llm_model.generate(prompt)
   ```

3. **Geração de Resumos Estruturados**:
   Criação de resumos que seguem uma estrutura predefinida, útil para relatórios padronizados.

   ```python
   def generate_structured_summary(text):
       prompt = f"""
       Crie um resumo estruturado do seguinte texto com estas seções:
       1. Principais Pontos:
          - 
          - 
          - 
       2. Análise:
          
       3. Conclusão:
          
       Texto original: {text}
       """
       return llm_model.generate(prompt)
   ```

> 💡 **Insight**: A combinação de prompts restritos com técnicas de few-shot learning pode melhorar significativamente a capacidade do modelo de gerar saídas estruturadas complexas e aderentes a formatos específicos [17].

### Considerações Matemáticas

A eficácia dos prompts restritos pode ser analisada através da teoria da informação. Considere a entropia condicional $H(Y|X)$, que mede a incerteza da variável aleatória $Y$ dado o conhecimento de $X$:

$$
H(Y|X) = -\sum_{x \in X} p(x) \sum_{y \in Y} p(y|x) \log p(y|x)
$$

No contexto de prompts restritos:
- $X$ representa o prompt (incluindo as restrições)
- $Y$ representa o espaço de saídas possíveis

==Prompts restritos eficazes reduzem $H(Y|X)$, diminuindo a incerteza sobre a saída esperada [18].==

#### Questões Técnicas/Teóricas

1. Como você poderia quantificar a redução no espaço de saída proporcionada por um prompt restrito em comparação com um prompt aberto?
2. Discuta as implicações de usar prompts restritos em termos de viés e equidade nos resultados gerados por LLMs.

### Conclusão

Prompts restritos representam uma técnica poderosa na engenharia de prompts, oferecendo um meio de aumentar a precisão, consistência e utilidade das saídas de modelos de linguagem de grande escala. Ao limitar explicitamente o espaço de respostas possíveis, essa abordagem facilita a geração de conteúdo estruturado e a integração com sistemas automatizados. No entanto, é crucial balancear as restrições com a flexibilidade necessária para cada tarefa específica, considerando as potenciais limitações e vieses que podem ser introduzidos.

À medida que os LLMs continuam a evoluir, a habilidade de formular prompts restritos eficazes torna-se uma competência cada vez mais valiosa para engenheiros de IA e cientistas de dados. O domínio dessa técnica permite a criação de aplicações mais robustas e confiáveis, aproveitando todo o potencial dos modelos de linguagem avançados em cenários que exigem alta precisão e estruturação de dados.

### Questões Avançadas

1. Como você projetaria um sistema de prompts restritos adaptativo que ajusta dinamicamente o nível de restrição com base no desempenho do modelo em tempo real?

2. Considerando as limitações dos prompts restritos em tarefas criativas, proponha uma abordagem híbrida que combine restrições com espaço para geração livre. Como você avaliaria a eficácia dessa abordagem?

3. Analise as implicações éticas e práticas de usar prompts restritos em sistemas de suporte à decisão críticos, como diagnósticos médicos ou análises financeiras. Quais salvaguardas deveriam ser implementadas?

4. Desenvolva um framework teórico para quantificar o trade-off entre a restrição do espaço de saída e a qualidade/relevância das respostas geradas por LLMs. Como isso poderia ser aplicado para otimizar prompts em diferentes domínios?

5. Discuta como a técnica de prompts restritos poderia ser estendida para modelos multimodais que trabalham com texto e imagem. Quais novos desafios e oportunidades surgiriam nesse cenário?

### Referências

[1] "Prompting relies on contextual generation. Given the prompt as context, the language model generates the next token based on its token probability, conditioned on the prompt: P(wi|w<i)." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[2] "A prompt can be a question (like "What is a transformer network?"), possibly in a structured format (like "Q: What is a transformer network? A:"), or can be an instruction (like "Translate the following sentence into Hindi: 'Chop the garlic finely'")." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[3] "A prompt can also contain demonstrations, examples to help make the instructions clearer. (like "Give the sentiment of the following sentence. Example Input: "I really loved Taishan Cuisine." Output: positive".)" (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[4] "As we'll see, prompting can be applied to inherently generative tasks (like summarization and translation) as well as to ones more naturally thought of as classification tasks." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[5] "Prompts get language models to generate text, but they also can be viewed as a learning signal, because these demonstrations can help language models learn to perform novel tasks." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[6] "For this reason we also refer to prompting as in-context-learning—learning that improves model performance or reduces some loss but does not involve gradient-based updates to the model's underlying parameters." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[7] "We can get the model to classify the sentiment of this text by taking the text and appending an incomplete statement to the review like In short, our stay was:" (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[8] "The power of this approach is that with suitable additions to the context a single LLM can produce outputs appropriate for many different tasks." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[9] "Consider the following templates for a variety of tasks:" (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[10] "Each template consists of an input text, designated as {input}, followed by a verbatim prompt to be passed to an LLM." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[11] "Notice the design pattern of the prompts above: the input is followed by some text which in turn will be completed by the desired response. This style, with the instruction at the end, is common in prompting because it helpfully constrains the generation." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[12] "This prompt doesn't do a good job of constraining possible continuations. Instead of a French translation, models given this prompt may instead generate another sentence in English that simply extends the English review." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[13] "Prompts need to be designed unambiguously, so that any reasonable continuation would accomplish the desired task (Reynolds and McDonell, 2021)." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[14] "An even more constraining style of prompt can specify the set of possible answers in the prompt." (Excerpt from Model Alignment, Prompting, and In-Context Learning)

[15] "This prompt uses a number of more sophisticated prompting characteristics. It specifies the two allowable choices (P) and (N), and ends the prompt with the open parenthesis that strongly suggests the