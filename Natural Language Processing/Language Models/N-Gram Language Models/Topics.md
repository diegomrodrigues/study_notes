## Chapter 3: N-gram Language Models

This chapter provides a comprehensive overview of n-gram language models, a foundational concept in natural language processing. Here's a breakdown of the key points and concepts introduced:

### 3.1 N-Grams

- **Goal:** To estimate the probability of a word given its history (previous words) and to assign probabilities to entire sequences of words.
- **Challenge:** Estimating probabilities for long sequences directly from counts is impractical due to the vast number of possible sequences.
- **Markov Assumption:**  Approximating the history by considering only the previous *n* words, simplifying the calculation of probabilities.
- **N-Gram Models:**  Using bigrams (2 words), trigrams (3 words), or higher-order n-grams to approximate the probability of a word given its context.
- **Maximum Likelihood Estimation (MLE):**  Estimating n-gram probabilities based on relative frequencies of sequences in a training corpus.
- **Log Probabilities:**  Storing and computing probabilities in log space to avoid numerical underflow during multiplication of many probabilities.
- **Long-Range Context and Efficiency:**  Discussing the challenges and techniques for handling long n-grams, large corpora, and computational efficiency in practical implementations.

### 3.2 Evaluating Language Models: Training and Test Sets

- **Intrinsic Evaluation:**  Evaluating the quality of a language model independent of its application, using metrics like perplexity.
- **Extrinsic Evaluation:**  Measuring the performance of a language model within a specific task, comparing its impact on the task's accuracy.
- **Training, Development, and Test Sets:**  Distinguishing between these datasets and their roles in model training, parameter tuning, and unbiased evaluation.
- **Training on the Test Set:**  Avoiding this bias, where a model is overfitted to the test data, leading to misleading evaluation results.

### 3.3 Evaluating Language Models: Perplexity

- **Perplexity:**  A standard intrinsic metric for evaluating language models, representing the weighted average branching factor of a language.
- **Inverse Relationship with Probability:**  Lower perplexity indicates a higher probability of the test set, signifying a better model.
- **Perplexity as a Measure of Surprise:**  A model with lower perplexity is less surprised by the test data, suggesting better prediction capability.
- **Computing Perplexity for Different N-Gram Models:**  Calculating perplexity for unigrams, bigrams, and higher-order n-grams.
- **Perplexity as a Weighted Average Branching Factor:**  Illustrating how perplexity can be interpreted as the weighted average of the number of possible next words in a language.

### 3.4 Sampling Sentences from a Language Model

- **Sampling:**  Generating sentences randomly according to the probabilities defined by a language model, providing a visualization of the model's knowledge.
- **Visualizing Unigram and N-Gram Sampling:**  Explaining the process of sampling sentences from unigram and higher-order n-gram models.

### 3.5 Generalizing vs. Overfitting the Training Set

- **Corpus Dependence:**  Highlighting the dependence of n-gram models on the specific training corpus, leading to varying predictions for different genres or dialects.
- **Overfitting:**  The tendency of n-grams to overfit the training data, resulting in poor generalization to unseen data.
- **Subword Tokenization:**  Using subword tokens to address the issue of unseen words in the test set, as any word can be represented as a sequence of known subwords.

### 3.6 Smoothing, Interpolation, and Backoff

- **Problem with Zero Frequency N-Grams:**  The challenge of handling sequences that do not occur in the training set, leading to incorrect probability estimates and hindering model evaluation.
- **Smoothing Techniques:**  Addressing the zero-frequency issue by distributing probability mass from seen to unseen events.
- **Laplace (Add-One) Smoothing:**  A simple smoothing algorithm that adds a pseudocount of 1 to each n-gram count, resulting in non-zero probabilities for all sequences.
- **Discounting:**  Viewing smoothing as discounting (lowering) non-zero counts to redistribute probability mass to unseen events.
- **Add-K Smoothing:**  A variation of add-one smoothing where a fractional count k is added to each count, offering more flexibility.
- **Language Model Interpolation:**  Combining n-gram probabilities of different orders (unigrams, bigrams, trigrams) using weighted averaging (linear interpolation) to improve generalization.
- **Context-Conditioned Weights:**  Adjusting the weights in interpolation based on the context of previous words, giving more weight to more reliable estimates.
- **Stupid Backoff:**  A simple backoff algorithm that backs off to lower-order n-grams when a higher-order n-gram has zero counts, without discounting probabilities.

### 3.7 Advanced: Perplexity's Relation to Entropy

- 
- **Entropy:**  A measure of information, providing a lower bound on the number of bits required to encode a random variable using the optimal coding scheme.
- **Entropy Rate:**  The entropy of a language measured per word or token, computed by considering sequences of infinite length.
- **Stationary and Ergodic Processes:**  Describing the conditions for applying the Shannon-McMillan-Breiman theorem to estimate entropy using a single long sequence.
- **Cross-Entropy:**  A measure of the similarity between two probability distributions, where one distribution is used to evaluate the accuracy of the other.
- **Cross-Entropy as an Upper Bound on Entropy:**  Showing that the cross-entropy of a model on the true distribution is always greater than or equal to the true entropy.
- **Perplexity and Cross-Entropy:**  Formally defining perplexity as 2 raised to the power of the cross-entropy of a model on a sequence.

### 3.8 Summary

This chapter introduced the core concepts of n-gram language models, including their definition, training, evaluation, and smoothing techniques. The chapter also discussed the relationship between perplexity and entropy in information theory, providing a theoretical foundation for evaluating language models.

### Bibliographical and Historical Notes

- **Early Pioneers:**  Highlighting the contributions of Markov, Shannon, and Chomsky in developing the foundations of n-gram models.
- **Resurgence of N-Grams:**  Discussing the independent contributions of Jelinek and Baker in using n-grams for speech recognition.
- **Evolution of Smoothing Techniques:**  Reviewing various smoothing algorithms, including Add-one, Add-K, Good-Turing, and Witten-Bell discounting.
- **State-of-the-Art N-Gram Modeling:**  Mentioning Modified Interpolated Kneser-Ney as a standard technique and toolkits like SRILM and KenLM.
- **Transition to Neural Language Models:**  Discussing the limitations of n-grams and the emergence of neural language models, which address the challenges of parameter growth and generalization.

This thorough breakdown provides a strong foundation for understanding n-gram language models, their strengths and limitations, and their role in the history of natural language processing.