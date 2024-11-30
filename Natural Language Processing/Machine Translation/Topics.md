## Chapter 18: Machine Translation


**18.1 Machine Translation as a Task**

*   **Formalization as an optimization problem:** Machine translation is framed as an optimization problem, maximizing a scoring function over all possible target sentences given a source sentence.  This establishes the core task and highlights the need for decoding and learning algorithms.
*   **Challenge of decoding:** The massive search space of possible translations is identified as a significant hurdle in machine translation, contrasting with simpler sequence labeling tasks that benefit from locality assumptions.  The inherent lack of such assumptions in human translation is emphasized.
*   **NP-hardness of decoding:** The computational complexity of decoding in machine translation is noted as NP-hard, justifying the need for approximate algorithms.
*   **Challenge of model estimation:** The difficulty of estimating translation models from parallel sentences is highlighted, due to the lack of explicit word-to-word alignments in the data.
*   **Latent variable approach (SMT):** Treating word alignment as a latent variable is presented as one solution, leading to classical statistical machine translation (SMT) systems.
*   **Expressive function approach (NMT):** Using more expressive functions to model the source-target relationship is introduced as an alternative solution, leading to neural machine translation (NMT) systems.
*   **Vauquois Pyramid as a theoretical framework:** The Vauquois Pyramid is introduced as a model for the levels of translation, progressing from word-level to semantic representation, and finally to a hypothetical interlingua. This theoretical framework guides the discussion of different translation approaches.
*   **Evaluation of translations (Adequacy and Fluency):** Two main criteria for evaluating translations—adequacy (correct meaning) and fluency (natural language)—are defined and illustrated with examples.  The tension between these two is highlighted.


**18.1.1 Evaluating Translations**

*   **Automated evaluation metrics:** Automated metrics for evaluating machine translations are introduced, emphasizing their reliance on comparing system translations to reference translations produced by human experts.
*   **BLEU (Bilingual Evaluation Understudy):** The BLEU metric, based on n-gram precision and brevity penalty, is introduced as a popular quantitative metric. The calculation of n-gram precision and its smoothing are explained.  The brevity penalty is explained as a way to avoid bias toward shorter translations.
*   **Limitations of BLEU:** BLEU's limitations are highlighted, such as its insensitivity to meaning and fluency, particularly concerning issues like pronoun resolution.  This is presented to motivate further development in evaluation metrics.
*   **Bias and fairness in translation:** The intersection between translation evaluation and issues of bias and fairness is introduced, emphasizing how biases in training data can lead to biased translations, particularly regarding gender stereotypes in profession assignment.  This points to broader societal concerns with data-driven methods.
*   **Other automated metrics:** Other evaluation metrics, such as METEOR (weighted F-measure), TER (Translation Error Rate), and RIBES (rank correlation-based metric), are briefly introduced, highlighting their strengths and weaknesses in comparison to BLEU.


**18.1.2 Data**

*   **Parallel corpora as primary data source:**  Parallel corpora (sentence-level aligned translations) are identified as the essential data for data-driven machine translation.
*   **Examples of parallel corpora:** Well-known parallel corpora (Hansards, EuroParl) are mentioned, highlighting their limitations in terms of domain and style.
*   **Growth and diversification of translation datasets:** The expansion of available data to include news, subtitles, social media, dialogues, TED talks, and scientific articles is noted, showing the evolution of data resources for machine translation.
*   **Bottleneck in parallel data:** The scarcity of parallel corpora, especially for low-resource language pairs, is pointed out as a major bottleneck in machine translation research.
*   **Pivot language approach:** The use of high-resource languages as pivot languages to bridge low-resource language pairs is described as a strategy to alleviate the data sparsity issue.
*   **Judeo-Christian Bible as a unique resource:** The Bible's extensive translation history across numerous languages is highlighted as a valuable albeit limited data source.
*   **Methods for creating parallel corpora:**  Methods for automatically creating parallel corpora (e.g., alignment from web pages, crowdsourcing) are mentioned.



**18.2 Statistical Machine Translation**

*   **Decomposition of the scoring function:** The scoring function for statistical machine translation is decomposed into adequacy and fluency scores.  This allows separate modeling and data utilization.
*   **Noisy channel model:** The noisy channel model is introduced as a probabilistic framework for statistical machine translation, justifying the decomposition of the scoring function into likelihood and prior terms.
*   **Language modeling for fluency:**  The fluency score is linked to language modeling, emphasizing the use of established language modeling techniques to evaluate the target sentence.
*   **Translation model estimation:** The core problem of estimating the translation model probability is highlighted.



**18.2.1 Statistical Translation Modeling**

*   **Word-to-word alignment:** Word-to-word alignment is introduced as the simplest form of translation modeling, introducing the concept of an alignment set which links source and target words.
*   **Joint probability of alignment and translation:** The joint probability of the alignment and translation is defined as a product of alignment and translation probabilities, showing how the model factors across words.  This factorization implies strong independence assumptions.
*   **Independence assumptions in word-to-word alignment:**  The key assumptions of independence in alignment probability and translation probability are explicitly stated and their limitations discussed.
*   **IBM Models 1-6:**  The IBM Models 1-6 are mentioned as a series of alignment models with increasingly relaxed independence assumptions.
*   **IBM Model 1 and convexity:** IBM Model 1, with its strong independence assumption, is highlighted as providing a convex learning objective, suitable for initialization.


**18.2.2 Estimation**

*   **Parameter estimation with annotated alignments:** Parameter estimation is described for the case where word alignments are provided. Relative frequencies are used to estimate translation probabilities.
*   **EM algorithm for unsupervised estimation:** The Expectation-Maximization (EM) algorithm is introduced for learning translation probabilities from unaligned data. The E-step (updating alignment beliefs) and M-step (updating translation model) are described.
*   **Convergence properties of EM:** The EM algorithm's convergence properties are discussed, highlighting that it converges to a local optimum, except for the case of IBM Model 1 where global optimality is guaranteed.


**18.2.3 Phrase-Based Translation**

*   **Limitations of word-to-word translation:** The limitations of word-to-word translation in handling multi-word expressions are illustrated with an example.
*   **Phrase-based translation as a generalization:** Phrase-based translation is presented as a generalization of word-based models, handling multi-word units and alignments.
*   **Phrase alignment and probability modeling:** The definition of phrase alignment and its incorporation into the translation probability model is detailed.


**18.2.4 Syntax-Based Translation**

*   **Motivation from the Vauquois Pyramid:** Syntax-based translation is motivated by the Vauquois Pyramid, suggesting that operating at a higher syntactic level can simplify translation.
*   **Handling word order differences:** The issue of word order differences between languages (e.g., adjective-noun ordering) is discussed as a problem for word-based models and a motivation for syntax-based approaches.
*   **Synchronous Context-Free Grammars (SCFGs):** SCFGs are introduced as a formalism for modeling syntactic structure in both source and target languages.  Their ability to handle word reordering is explained.
*   **CKY parsing algorithm:** The use of the CKY algorithm for parsing with SCFGs is mentioned, highlighting the relationship between parsing and translation in this framework.
*   **Computational cost of combining SCFGs with language models:** The computational challenges of combining SCFGs with target language models are noted.


**18.3 Neural Machine Translation**

*   **Encoder-decoder architecture:** The encoder-decoder architecture is introduced as the basis for neural machine translation systems, where the encoder maps the source sentence to a representation and the decoder generates the target sentence.
*   **Recurrent neural networks for decoding:**  Recurrent neural networks (RNNs), specifically LSTMs, are introduced as a common choice for the decoder, generating the target sentence word by word.
*   **End-to-end training:**  The end-to-end training of the encoder and decoder from parallel sentences using conditional log-likelihood maximization is explained.
*   **Sequence-to-sequence model:** The sequence-to-sequence model is introduced as the simplest encoder-decoder architecture, using the final hidden state of the encoder as the context for the decoder.
*   **Reversing the source sentence:** The practice of reversing the source sentence to improve performance is noted.
*   **Deep LSTMs for encoder and decoder:** The use of deep LSTMs (multiple layers) in both encoder and decoder is discussed.
*   **Ensemble methods:** Ensemble methods, combining multiple models, are presented as a way to improve performance.
*   **Standard training procedures:** Standard training procedures for sequence-to-sequence models are summarized.


**18.3.1 Neural Attention**

*   **Neural attention mechanism:** The general neural attention mechanism is described, emphasizing its use of queries to select information from a memory of key-value pairs.  The mathematical formulation is detailed.
*   **Attention in encoder-decoder models:** The application of attention to encoder-decoder models is explained, where the encoder's hidden states serve as keys and values, and the decoder's hidden states serve as queries.
*   **Bidirectional LSTMs for encoding:** The use of bidirectional LSTMs for creating the key-value matrix from the source sentence is explained.
*   **Computation of attention weights:** The computation of attention weights (using softmax or sigmoid) is detailed, showing how the compatibility between queries and keys is used to weigh the values.
*   **Incorporating context vectors into decoder:** The method for incorporating the context vector (weighted average of source representations) into the decoder's output probability model is detailed.


**18.3.2 Neural Machine Translation without Recurrence**

*   **Transformer architecture:** The transformer architecture, which eliminates recurrence using self-attention, is introduced.
*   **Self-attention mechanism:** Self-attention is explained as a mechanism that allows attending to different parts of the input sequence within the encoder and decoder.
*   **Multiple attention heads:**  The use of multiple attention heads to capture different aspects of the input is mentioned.
*   **Positional encodings:**  Positional encodings are introduced as a way to incorporate word order information into the transformer model.  Their sinusoidal formulation is described.
*   **Convolutional neural networks for encoding:** The use of convolutional neural networks for encoding in neural machine translation is discussed, highlighting their speed advantage over recurrent models.


**18.3.3 Out-of-Vocabulary Words**

*   **Challenges of out-of-vocabulary (OOV) words:** The problem of handling out-of-vocabulary (OOV) words is introduced, emphasizing the limitations of word-based and phrase-based approaches.
*   **Causes of OOV words:** The reasons for OOV words (new proper nouns and morphological variations) are discussed.


**18.4 Decoding**

*   **Intractability of exact decoding:** The computational intractability of finding the optimal translation is reiterated, highlighting the necessity of approximate decoding methods.
*   **Beam search:** Beam search is introduced as a commonly used approximate decoding algorithm.
*   **Incremental decoding algorithm:** The incremental decoding algorithm is presented as a simpler approximation to the optimization problem.
*   **Limitations of dynamic programming:**  The limitations of dynamic programming for decoding are explained, due to the non-Markovian nature of the hidden states in RNNs.
*   **NP-completeness of RNN decoding:** The NP-completeness of decoding with RNNs is mentioned.

**18.5 Training Towards the Evaluation Metric**

*   **Mismatch between likelihood and evaluation metrics:** The mismatch between likelihood-based training objectives and evaluation metrics like BLEU is emphasized. This motivates training directly towards the evaluation metric.
*   **Minimum Error Rate Training (MERT):** MERT is presented as a method to directly minimize the error rate of translations, using a finite set of candidate translations.
*   **Minimum risk training:** Minimum risk training is introduced as a method to smooth the non-differentiable error function by considering the expected error over a distribution of translations.
*   **Annealing:** Annealing is presented as a way to generalize the minimum risk training by exponentiating translation probabilities.
*   **Importance sampling:** Importance sampling is described as a technique for approximating the expected error by sampling from a proposal distribution and weighting the samples appropriately.

**Exercises:** The chapter concludes with exercises designed to test the reader's understanding of the concepts covered.  These exercises include computational tasks, conceptual questions about model design and limitations, and problem-solving tasks related to the algorithms described in the chapter.