## Chapter 6: Language Models

### 6. Introduction

- **Language Modeling:**  The task of assigning probabilities to sequences of word tokens, capturing the statistical regularities of language.
- **Applications of Language Models:**  Various applications where language models are used to generate fluent text, including machine translation, speech recognition, summarization, and dialogue systems.
- **Noisy Channel Model and Machine Translation:**  Using language models within the noisy channel framework for machine translation, emphasizing the importance of monolingual data.

### 6.1 N-Gram Language Models

- **Relative Frequency Estimation:**  Using relative frequencies to estimate the probability of a sentence, but highlighting the problem of data sparsity and high variance.
- **Chain Rule of Probability:**  Applying the chain rule to decompose the probability of a sequence into conditional probabilities of words given their preceding context.
- **N-Gram Approximation:**  Introducing the n-gram approximation, which conditions on only the past n-1 words for tractability.
- **Padding with Special Symbols:**  Adding start-of-string (□) and end-of-string (■) symbols to handle boundary cases in n-gram models.
- **Estimating N-Gram Probabilities:**  Using relative frequencies to estimate the probabilities of n-grams.
- **Limitations of N-Gram Models:**  Discussing the limitations of n-gram models, including high bias for small n and high variance for large n.

### 6.2 Smoothing and Discounting

- **Bias-Variance Trade-off:**  Recognizing the bias-variance trade-off in n-gram language models, with smoothing as a technique to reduce variance.
- **Lidstone Smoothing:**  Adding pseudocounts to relative frequency estimates to avoid zero probabilities.
- **Laplace Smoothing and Jeffreys-Perks Law:**  Specific cases of Lidstone smoothing with different pseudocount values.
- **Effective Counts and Discounting:**  Introducing the concept of effective counts to account for smoothing and computing discounts for observed n-grams.
- **Absolute Discounting:**  Borrowing probability mass from observed n-grams and redistributing it equally to unseen n-grams.
- **Katz Backoff:**  Using lower-order n-gram models to estimate probabilities for unseen higher-order n-grams.
- **Interpolation:**  Computing the probability of a word in context as a weighted average of its probabilities across different order n-grams.
- **Expectation-Maximization (EM) for Interpolation:**  Using EM to learn the weights of the interpolation model by treating the n-gram order as missing data.
- **Kneser-Ney Smoothing:**  A sophisticated smoothing technique that accounts for word versatility and uses continuation probabilities to redistribute probability mass.

### 6.3 Recurrent Neural Network Language Models

- **Discriminative Approach to Language Modeling:**  Treating language modeling as a discriminative learning problem, where the goal is to maximize conditional probability.
- **Reparametrization with Dense Vectors:**  Representing word and context probabilities using dense vectors and the softmax function.
- **Recurrent Neural Networks (RNNs):**  Introducing recurrent neural networks (RNNs) and their use in language modeling.
- **Elman Unit:**  Defining the Elman unit as a simple recurrent operation for updating context vectors.
- **Backpropagation Through Time:**  Extending backpropagation to RNNs for learning parameters through time.
- **Long Short-Term Memory (LSTM):**  Introducing LSTMs as a variant of RNNs that address the vanishing gradient problem and allow for long-range dependencies.

### 6.4 Evaluating Language Models

- **Intrinsic vs. Extrinsic Evaluation:**  Distinguishing between task-neutral intrinsic evaluation (e.g., held-out likelihood) and task-specific extrinsic evaluation.
- **Held-out Likelihood:**  Using the likelihood of held-out data to evaluate language models.
- **Perplexity:**  Defining perplexity as a deterministic transformation of held-out likelihood and its relationship with model performance.
- **Penn Treebank and 1B Word Benchmark:**  Mentioning two widely used language modeling datasets and typical perplexities achieved by n-gram and neural models.

### 6.5 Out-of-Vocabulary Words

- **Closed-Vocabulary Assumption:**  Recognizing the limitations of closed-vocabulary settings, where unknown words are not accounted for.
- **Handling Out-of-Vocabulary Words (UNK):**  Strategies for dealing with unknown words, including marking them with a special token and using character-level or subword-level models.

### Additional Resources

- **Non-Recurrent Neural Language Models:**  Mentioning earlier neural language models, such as the neural probabilistic language model and the log-bilinear language model.