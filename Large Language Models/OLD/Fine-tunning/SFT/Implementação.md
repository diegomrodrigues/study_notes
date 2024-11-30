Certamente! Vou adicionar comentários detalhados a cada linha do código, incluindo as dimensões dos tensores quando houver operações entre eles. Isso proporcionará uma compreensão mais profunda do funcionamento interno da classe LlamaSFTTrainer.

```python
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalPrediction

class LlamaSFTTrainer(Trainer):
    """
    LlamaSFTTrainer é uma classe personalizada para treinar modelos LlamaForCausalLM.
    Herda da classe Trainer do Hugging Face e implementa funcionalidades específicas
    para o treinamento de modelos LLaMA.
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[LlamaTokenizer] = None,
        **kwargs
    ):
        # Chama o construtor da classe pai (Trainer) com os argumentos fornecidos
        super().__init__(model, args, train_dataset, eval_dataset, tokenizer, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Calcula a perda para o modelo LLaMA.
        """
        # Verifica se há labels nos inputs e as remove se existirem
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # Passa os inputs pelo modelo
        # inputs: Dict[str, torch.Tensor] onde cada tensor tem shape [batch_size, seq_length]
        # outputs: Objeto contendo loss e logits, ambos com shape [batch_size, seq_length, vocab_size]
        outputs = model(**inputs)
        
        if labels is not None:
            # Calcula a perda usando label smoothing se configurado, ou usa a perda padrão do modelo
            # loss: escalar tensor []
            loss = self.label_smoother(outputs, labels) if self.label_smoother is not None else outputs.loss
        else:
            loss = outputs.loss

        # Retorna a perda e opcionalmente os outputs
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: LlamaForCausalLM,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Realiza um passo de predição/avaliação no modelo.
        """
        # Se não estiver gerando predições ou se estiver calculando apenas a perda, usa o método da classe pai
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        # Verifica se há labels nos inputs
        has_labels = "labels" in inputs
        # Prepara os inputs para o modelo
        inputs = self._prepare_inputs(inputs)

        # Configura os parâmetros para geração de texto
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )

        # Gera tokens usando o modelo
        # input_ids: [batch_size, seq_length]
        # attention_mask: [batch_size, seq_length]
        # generated_tokens: [batch_size, max_length]
        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # Faz o padding dos tokens gerados se necessário
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
            # generated_tokens após padding: [batch_size, max_length]

        # Calcula a perda se houver labels
        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    # outputs: Objeto contendo loss e logits, ambos com shape [batch_size, seq_length, vocab_size]
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    # loss: escalar tensor []
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        # Prepara as labels para retorno
        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            # labels após padding: [batch_size, max_length]

        # Retorna a perda, tokens gerados e labels
        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        """
        Faz o padding de um tensor para um comprimento máximo especificado.
        """
        # Verifica se o tokenizer está disponível
        if self.tokenizer is None:
            raise ValueError("Tokenizer é necessário para fazer o padding dos tensores até o comprimento máximo.")
        
        # Se o tensor já tem o comprimento máximo ou maior, retorna sem modificação
        if tensor.shape[-1] >= max_length:
            return tensor
        
        # Faz o padding do tensor
        # tensor: [batch_size, seq_length]
        padded_tensor = self.tokenizer.pad(
            {"input_ids": tensor},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        # padded_tensor: [batch_size, max_length]
        
        return padded_tensor

```

Agora, vou adicionar comentários linha por linha ao exemplo de uso do LlamaSFTTrainer, incluindo informações sobre as dimensões dos tensores quando relevante:

```python
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments
from datasets import load_dataset

# Carrega o modelo LLaMA pré-treinado
# O modelo carregado terá a arquitetura LlamaForCausalLM com parâmetros pré-treinados
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Carrega o tokenizador LLaMA correspondente
# O tokenizador será usado para converter texto em tokens e vice-versa
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Carrega o dataset
# Nota: Substitua "your_dataset" pelo nome ou caminho do seu dataset real
dataset = load_dataset("your_dataset")

# Separa o dataset em conjuntos de treinamento e validação
# train_dataset: Dataset contendo exemplos de treinamento
# eval_dataset: Dataset contendo exemplos de validação
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Configura os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",              # Diretório para salvar os checkpoints do modelo e logs
    num_train_epochs=3,                  # Número total de épocas de treinamento
    per_device_train_batch_size=4,       # Número de exemplos por batch durante o treinamento
    per_device_eval_batch_size=4,        # Número de exemplos por batch durante a avaliação
    warmup_steps=500,                    # Número de passos de aquecimento para o scheduler de taxa de aprendizagem
    weight_decay=0.01,                   # Fator de decaimento de peso para regularização L2
    logging_dir="./logs",                # Diretório para salvar os logs de treinamento
)

# Inicializa o treinador personalizado LlamaSFTTrainer
trainer = LlamaSFTTrainer(
    model=model,                         # Modelo LLaMA pré-treinado
    args=training_args,                  # Argumentos de treinamento configurados acima
    train_dataset=train_dataset,         # Dataset de treinamento
    eval_dataset=eval_dataset,           # Dataset de avaliação
    tokenizer=tokenizer,                 # Tokenizador LLaMA
)

# Inicia o processo de treinamento
# Isso executará o loop de treinamento, realizando avaliações periódicas e salvando checkpoints
trainer.train()

# Após o treinamento, o modelo estará fine-tuned no dataset fornecido
# Os resultados do treinamento, incluindo checkpoints do modelo, estarão no diretório especificado em output_dir

```

Estes comentários detalhados fornecem uma explicação clara de cada linha de código, tanto na classe LlamaSFTTrainer quanto no exemplo de uso. As informações sobre as dimensões dos tensores são particularmente úteis para entender como os dados fluem através do modelo durante o treinamento e a avaliação.



---



Certainly, I can help you improve this implementation by incorporating more aspects of the Hugging Face SFT (Supervised Fine-Tuning) Trainer. I'll update the `LlamaSFTTrainer` class with additional features and improvements. Here's an enhanced version:



```python
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalPrediction
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

class LlamaSFTTrainer(Trainer):
    """
    LlamaSFTTrainer é uma classe personalizada para treinar modelos LlamaForCausalLM.
    Herda da classe Trainer do Hugging Face e implementa funcionalidades específicas
    para o treinamento de modelos LLaMA com suporte a LoRA e métricas personalizadas.
    """
    def __init__(
        self,
        model: Union[LlamaForCausalLM, PeftModel],
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[LlamaTokenizer] = None,
        data_collator: Optional[DataCollatorForLanguageModeling] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[LoraConfig] = None,
        **kwargs
    ):
        # Inicializa o modelo com LoRA se a configuração for fornecida
        if peft_config:
            model = get_peft_model(model, peft_config)
        
        # Configura o data collator padrão se não for fornecido
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Configura a função de métricas padrão se não for fornecida
        if compute_metrics is None:
            compute_metrics = self.default_compute_metrics
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Calcula a perda para o modelo LLaMA.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = inputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        outputs = model(**inputs)

        if self.label_smoother is not None and "labels" in inputs:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: Union[LlamaForCausalLM, PeftModel],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Realiza um passo de predição/avaliação no modelo.
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.model.config.max_length,
            "num_beams": self.model.config.num_beams,
            "do_sample": self.args.do_sample,
            "top_p": self.args.top_p,
            "temperature": self.args.temperature,
            **self.gen_kwargs,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None        

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        """
        Faz o padding de um tensor para um comprimento máximo especificado.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer é necessário para fazer o padding dos tensores até o comprimento máximo.")
        
        if tensor.shape[-1] >= max_length:
            return tensor
        
        padded_tensor = self.tokenizer.pad(
            {"input_ids": tensor},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        
        return padded_tensor

    def default_compute_metrics(self, eval_preds):
        """
        Função padrão para calcular métricas de avaliação.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = {}
        
        # Calcula a perplexidade
        loss = torch.nn.CrossEntropyLoss()(torch.tensor(preds).view(-1, self.model.config.vocab_size), torch.tensor(labels).view(-1))
        result["perplexity"] = torch.exp(loss).item()

        # Você pode adicionar mais métricas personalizadas aqui, como BLEU, ROUGE, etc.
        
        return result

    def create_optimizer(self):
        """
        Cria um otimizador personalizado com diferentes taxas de aprendizado para diferentes partes do modelo.
        """
        if self.optimizer is None:
            decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Salva o modelo, possivelmente apenas os adaptadores LoRA se estiver usando PEFT.
        """
        if self.use_peft:
            self.model.save_pretrained(output_dir)
        else:
            super().save_model(output_dir, _internal_call)

```

This improved implementation of `LlamaSFTTrainer` incorporates several enhancements and additional features from the Hugging Face SFT Trainer:

1. Support for Parameter-Efficient Fine-Tuning (PEFT) with LoRA: The trainer now accepts a `peft_config` parameter, which allows for easy integration of LoRA (Low-Rank Adaptation) for more efficient fine-tuning.

2. Improved data collator handling: If no data collator is provided, it defaults to `DataCollatorForLanguageModeling` with appropriate settings for causal language modeling.

3. Enhanced `compute_loss` method: It now handles cases where labels are not explicitly provided, deriving them from input_ids when necessary.

4. More flexible `prediction_step`: The method now supports various generation parameters like `do_sample`, `top_p`, and `temperature`.

5. Default metrics computation: A `default_compute_metrics` method is added to calculate perplexity and can be easily extended to include other metrics like BLEU or ROUGE.

6. Custom optimizer creation: The `create_optimizer` method is overridden to apply different learning rates and weight decay to different parts of the model, which can be beneficial for fine-tuning large language models.

7. PEFT-aware model saving: The `save_model` method is modified to handle saving of PEFT models correctly.

8. Additional type hints and docstrings for better code readability and maintenance.

These improvements make the `LlamaSFTTrainer` more versatile and aligned with state-of-the-art practices for fine-tuning large language models. The trainer now supports efficient fine-tuning techniques, provides more control over the training process, and includes basic evaluation metrics out of the box.

To use this improved trainer, you would typically set it up like this:

```python
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments
from peft import LoraConfig

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained("path_to_llama_model")
tokenizer = LlamaTokenizer.from_pretrained("path_to_llama_tokenizer")

# Set up LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the trainer
trainer = LlamaSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

# Start training
trainer.train()
```

This setup allows for efficient fine-tuning of LLaMA models with LoRA, custom metrics, and other advanced features. You can further customize the trainer by overriding its methods or adding your own functionality as needed.



---



from transformers import Trainer, TrainingArguments
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import EvalPrediction
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, AdapterConfig, PeftModel, PeftConfig, TaskType
import numpy as np

class LlamaSFTTrainer(Trainer):
    """
    LlamaSFTTrainer é uma classe personalizada para treinar modelos LlamaForCausalLM.
    Suporta treinamento eficiente com LoRA e Adapters usando a biblioteca PEFT.
    """
    def __init__(
        self,
        model: Union[LlamaForCausalLM, PeftModel],
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[LlamaTokenizer] = None,
        data_collator: Optional[DataCollatorForLanguageModeling] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Union[LoraConfig, AdapterConfig]] = None,
        **kwargs
    ):
        # Inicializa o modelo com PEFT se a configuração for fornecida
        if peft_config:
            if not isinstance(model, PeftModel):
                model = get_peft_model(model, peft_config)
            self.peft_config = peft_config
        else:
            self.peft_config = None
        
```python
    # Configura o data collator padrão se não for fornecido
    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Configura a função de métricas padrão se não for fornecida
    if compute_metrics is None:
        compute_metrics = self.default_compute_metrics
    
    super().__init__(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **kwargs
    )

def compute_loss(self, model, inputs, return_outputs=False):
    """
    Calcula a perda para o modelo LLaMA.
    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
    
    outputs = model(**inputs)

    if self.label_smoother is not None and "labels" in inputs:
        loss = self.label_smoother(outputs, labels)
    else:
        loss = outputs.loss

    return (loss, outputs) if return_outputs else loss

def prediction_step(
    self,
    model: Union[LlamaForCausalLM, PeftModel],
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Realiza um passo de predição/avaliação no modelo.
    """
    if not self.args.predict_with_generate or prediction_loss_only:
        return super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
    
    has_labels = "labels" in inputs
    inputs = self._prepare_inputs(inputs)

    gen_kwargs = {
        "max_length": self.model.config.max_length,
        "num_beams": self.model.config.num_beams,
        "do_sample": self.args.do_sample,
        "top_p": self.args.top_p,
        "temperature": self.args.temperature,
        **self.gen_kwargs,
    }

    generated_tokens = self.model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **gen_kwargs,
    )
    
    if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
        generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

    with torch.no_grad():
        if has_labels:
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
            else:
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        else:
            loss = None        

    if self.args.prediction_loss_only:
        return (loss, None, None)

    labels = inputs["labels"]
    if labels.shape[-1] < gen_kwargs["max_length"]:
        labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

    return (loss, generated_tokens, labels)

def _pad_tensors_to_max_len(self, tensor, max_length):
    """
    Faz o padding de um tensor para um comprimento máximo especificado.
    """
    if self.tokenizer is None:
        raise ValueError("Tokenizer é necessário para fazer o padding dos tensores até o comprimento máximo.")
    
    if tensor.shape[-1] >= max_length:
        return tensor
    
    padded_tensor = self.tokenizer.pad(
        {"input_ids": tensor},
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"]
    
    return padded_tensor

def default_compute_metrics(self, eval_preds):
    """
    Função padrão para calcular métricas de avaliação.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = {}
    
    # Calcula a perplexidade
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(preds).view(-1, self.model.config.vocab_size), torch.tensor(labels).view(-1))
    result["perplexity"] = torch.exp(loss).item()

    # Você pode adicionar mais métricas personalizadas aqui, como BLEU, ROUGE, etc.
    
    return result

def create_optimizer(self):
    """
    Cria um otimizador personalizado com diferentes taxas de aprendizado para diferentes partes do modelo.
    """
    if self.optimizer is None:
        decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer

def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    """
    Salva o modelo, possivelmente apenas os adaptadores ou LoRA se estiver usando PEFT.
    """
    if isinstance(self.model, PeftModel):
        self.model.save_pretrained(output_dir)
    else:
        super().save_model(output_dir, _internal_call)

def _save(self, output_dir: Optional[str] = None, state_dict=None):
    """
    Salva o checkpoint, sobrescrevendo o método padrão para lidar corretamente com modelos PEFT.
    """
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(self.model, PeftModel):
        self.model.save_pretrained(output_dir)
    else:
        self.model.save_pretrained(output_dir)
    
    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)

    # Salva o estado do otimizador
    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    
    # Salva o scheduler
    if self.lr_scheduler is not None:
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    # Salva o estado do treinamento
    train_state = {
        "best_model_checkpoint": self.state.best_model_checkpoint,
        "epoch": self.state.epoch,
        "global_step": self.state.global_step,
    }
    torch.save(train_state, os.path.join(output_dir, "trainer_state.json"))

def _load_from_checkpoint(self, resume_from_checkpoint):
    """
    Carrega o modelo e o estado do treinamento a partir de um checkpoint, lidando com modelos PEFT.
    """
    if isinstance(self.model, PeftModel):
        # Carrega apenas os adaptadores para modelos PEFT
        self.model = PeftModel.from_pretrained(self.model.base_model, resume_from_checkpoint)
    else:
        # Carrega o modelo completo para modelos não-PEFT
        self.model = self.model.from_pretrained(resume_from_checkpoint)

    # Carrega o estado do otimizador
    optimizer_path = os.path.join(resume_from_checkpoint, "optimizer.pt")
    if os.path.isfile(optimizer_path):
        self.optimizer.load_state_dict(torch.load(optimizer_path))

    # Carrega o scheduler
    scheduler_path = os.path.join(resume_from_checkpoint, "scheduler.pt")
    if self.lr_scheduler is not None and os.path.isfile(scheduler_path):
        self.lr_scheduler.load_state_dict(torch.load(scheduler_path))

    # Carrega o estado do treinamento
    trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
    if os.path.isfile(trainer_state_path):
        trainer_state = torch.load(trainer_state_path)
        self.state.best_model_checkpoint = trainer_state["best_model_checkpoint"]
        self.state.epoch = trainer_state["epoch"]
        self.state.global_step = trainer_state["global_step"]

def train(self, resume_from_checkpoint=None, **kwargs):
    """
    Inicia ou retoma o treinamento, lidando adequadamente com modelos PEFT.
    """
    if resume_from_checkpoint is not None:
        self._load_from_checkpoint(resume_from_checkpoint)

    return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)
```