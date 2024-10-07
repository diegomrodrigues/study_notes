## Chapter 6: Vector Semantics and Embeddings

### 6.1 Lexical Semantics

- **Lemma/Citation Form vs. Wordform:** Definition and examples of lemmas (e.g., mouse for mice) and wordforms (e.g., mice, singing).  Context: Discussing dictionary definitions.
- **Word Sense:** Definition and examples of different senses of a polysemous word (e.g., mouse as rodent vs. device). Context: Introducing polysemy and the need for disambiguation.
- **Synonymy:**  Definition (substitutability without changing truth conditions) and discussion of true vs. approximate synonymy (principle of contrast). Context: Exploring relationships between word senses.
- **Word Similarity:** Definition (approximate meaning overlap between words, not necessarily truth-preserving substitutability) and human judgment datasets (SimLex-999). Context: Shifting from sense relations to word relations.
- **Word Relatedness:** Definition (association between words based on co-participation in events or semantic fields, not similarity of features). Context: Expanding beyond similarity to other relationships.
- **Semantic Field:** Definition (set of words covering a specific domain with structured relations) and connection to topic models. Context: Explaining one type of word relatedness.
- **Semantic Frame & Roles:** Definition (words denoting participants in an event) and examples of frames, roles, and their impact on paraphrasing. Context: Introducing event-based semantic relations.
- **Connotation:** Definition (affective meaning related to emotions/evaluations) and examples of positive/negative connotation, sentiment. Context: Adding affective aspects to word meaning.
- **Dimensions of Affective Meaning:** Valence, arousal, dominance (Osgood's model). Context: Introducing the idea of representing meaning as points in space.

### 6.2 Vector Semantics

- **Distributional Hypothesis:** Words with similar distributions have similar meanings. Context: Connecting word meaning to distributional patterns.
- **Word Embeddings:** Representing words as points in multidimensional semantic space (dense and sparse).  Context: Introduction to the core concept of vector semantics.
- **Representation Learning:** Automatically learning useful representations from text, versus feature engineering. Context: Highlighting the significance of self-supervised learning.

### 6.3 Words and Vectors

- **Co-occurrence Matrix:** Representation of word co-occurrence frequency (term-document and term-term matrices). Context: Foundation for distributional models.
- **Term-Document Matrix:** Rows as words, columns as documents, cells as counts. Context: Representing documents and words as vectors.
- **Count Vector:** Representing a document as a vector of word counts. Context: Term-document matrix and vector space model of IR.
- **Vector Space Model (VSM):** Representing documents as vectors in a high-dimensional space. Context: Finding similar documents in IR.
- **Term-Term Matrix (Word-Context Matrix):** Rows as target words, columns as context words, cells as co-occurrence counts within a window. Context: Representing word meaning based on neighboring words.

### 6.4 Cosine for Measuring Similarity

- **Dot Product (Inner Product):** Measuring the similarity between two vectors by summing the product of their corresponding elements. Context: Foundation for cosine similarity.
- **Vector Length:** The magnitude of a vector, calculated as the square root of the sum of the squared elements. Context: Normalizing the dot product.
- **Cosine Similarity:** Normalized dot product, measuring the angle between two vectors. Context: Addressing the issue of vector length in similarity calculations.
- **Unit Vector:** Vector with length 1, obtained by dividing a vector by its length. Context: Simplifying cosine calculation.

### 6.5 TF-IDF: Weighing Terms in the Vector

- **Term Frequency (TF):**  Raw count or log-scaled count of a word in a document. Context: Measuring the importance of a word within a document.
- **Document Frequency (DF):** Number of documents containing a specific word. Context: Measuring the importance of a word across documents.
- **Inverse Document Frequency (IDF):** Log-scaled inverse of document frequency, giving higher weight to rare words. Context: Emphasizing discriminative words.
- **TF-IDF Weighting:** Product of TF and IDF, balancing within-document and across-document importance. Context: A weighting scheme for term-document matrices.

### 6.6 Pointwise Mutual Information (PMI)

- **Pointwise Mutual Information (PMI):** Measures how much more two words co-occur than expected by chance. Context: A weighting scheme for term-term matrices.
- **Positive PMI (PPMI):** Replaces negative PMI values with zero for increased reliability. Context: Addressing issues with negative PMI values.

### 6.7 Applications of TF-IDF or PPMI Vector Models

- **Document Similarity:** Computing the centroid of word vectors in a document and using cosine similarity to compare documents. Context: Applications in IR, plagiarism detection, etc.
- **Word Similarity:** Using cosine similarity between word vectors to find synonyms, track meaning changes, and discover word meanings. Context: Applications in paraphrase detection, semantic analysis, etc.

### 6.8 Word2vec

- **Static Embeddings:** Fixed word embeddings learned once. Context: Introduction to dense word embeddings like Word2Vec.
- **Self-Supervision:** Using running text as implicit training data. Context: How Word2Vec learns without explicit labels.
- **Skip-gram with Negative Sampling (SGNS):** Training a classifier to distinguish true context words from noise words, using the learned weights as embeddings. Context: Core algorithm for Word2Vec.
- **Target and Context Embeddings:** Two embedding matrices, $W$ and $C$, for target and context words. Context: Understanding the parameters of the Skip-gram model.
- **Noise Words:** Random words from the lexicon used as negative examples. Context: Training the skip-gram classifier.

### 6.8.1 The Classifier

- **Probability of Context Word:** $P(+|w,c)$: Probability that $c$ is a real context word for $w$. Context: The binary classification task in skip-gram.
- **Sigmoid Function:** Converting dot product similarity to probability. Context: Implementing logistic regression.

### 6.8.2 Learning Skip-Gram Embeddings

- **Loss Function:** Minimizing the negative log-likelihood of the observed positive and negative examples. Context: Objective function for Skip-gram training.
- **Stochastic Gradient Descent:** Iteratively updating embeddings to minimize the loss function. Context: Optimization algorithm for Skip-gram.

### 6.8.3 Other Kinds of Static Embeddings

- **FastText:** Addresses unknown words and word sparsity by using subword n-grams. Context: Extending Word2Vec for morphological richness and unknown words.
- **GloVe (Global Vectors):** Captures global corpus statistics by optimizing a function of word co-occurrence probabilities. Context: An alternative embedding algorithm.

### 6.9 Visualizing Embeddings

- **Listing Similar Words:** Sorting cosine similarity scores to find the closest words to a target word. Context: Simple method for visualizing word embeddings.
- **Clustering:** Hierarchical clustering to group similar words in the embedding space. Context: Visualizing relationships between clusters of words.
- **Dimensionality Reduction:** Projecting high-dimensional embeddings into 2D space for visualization (e.g., t-SNE). Context: Visualizing word relationships in a low-dimensional space.

### 6.10 Semantic Properties of Embeddings

- **Context Window Size:** Impact on syntactic vs. topical similarity. Context: Exploring how window size affects learned representations.
- **First-Order vs. Second-Order Co-occurrence:** Syntagmatic (nearby words) vs. paradigmatic (similar neighbors) associations. Context: Differentiating types of word relationships.
- **Parallelogram Model:** Solving analogy problems by vector addition and subtraction. Context: Capturing relational meanings with embeddings.

### 6.10.1 Embeddings and Historical Semantics

- **Diachronic Word Embeddings:** Tracking semantic change over time by comparing embeddings trained on different time periods. Context: Applications in historical linguistics.

### 6.11 Bias and Embeddings

- **Bias Amplification:** Embeddings can amplify biases present in the training data. Context: Ethical considerations and potential societal impact.
- **Mitigating Bias:** Techniques for reducing bias in embeddings. Context: Addressing fairness and representation issues.

### 6.12 Evaluating Vector Models

- **Extrinsic Evaluation:** Evaluating embeddings based on their performance on downstream tasks. Context: Measuring the practical utility of embeddings.
- **Intrinsic Evaluation:** Evaluating embeddings based on their ability to capture word similarity and relatedness. Context: Using datasets like WordSim-353, SimLex-999, and TOEFL.

### 6.13 Summary

- **Core Concepts of Vector Semantics:** Recap of word representations as vectors and the distinction between sparse and dense models.
- **Key Algorithms and Techniques:** Summary of TF-IDF, PPMI, Word2Vec, GloVe, and cosine similarity.

This detailed outline provides a comprehensive structure for studying the chapter, covering key concepts and their context within the larger discussion of vector semantics and embeddings. It's designed to be useful for someone with advanced knowledge of AI, statistics, and deep learning, enabling efficient review and in-depth understanding.