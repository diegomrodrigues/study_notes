This paper introduces DeepSeekMath 7B, a 7 billion parameter language model specializing in mathematical reasoning.  Let's break down the key contributions and methods:

**I. Core Contributions:**

1. **Scalable Math Pre-training:**  DeepSeekMath significantly improves upon existing open-source models by leveraging a massive, high-quality dataset (DeepSeekMath Corpus) containing 120 billion math-related tokens extracted from Common Crawl.  This dataset is far larger than previously used datasets like MathPile and OpenWebMath.  The iterative data collection pipeline, using fastText classification and human annotation, is a crucial aspect of this contribution.  The success highlights the untapped potential of Common Crawl for specialized LLM training.
2. **Improved Pre-training Strategy:** The paper demonstrates that initializing with a code-trained model (DeepSeek-Coder-Base-v1.5 7B) before math pre-training leads to better performance compared to starting with a general-purpose LLM. This finding provides evidence supporting the hypothesis that code training enhances reasoning abilities, at least in the context of mathematics.  Surprisingly, training solely on arXiv papers yielded no significant improvement, suggesting the quality of data is more crucial than sheer quantity.
3. **Group Relative Policy Optimization (GRPO):** This novel reinforcement learning algorithm improves upon Proximal Policy Optimization (PPO) by eliminating the need for a separate critic network.  Instead, GRPO estimates the baseline reward from group scores (multiple model outputs for the same question), significantly reducing computational costs and memory usage.  GRPO shows substantial performance gains on various benchmarks, both in-domain and out-of-domain.
4. **Unified Paradigm for RL Methods:** The paper proposes a unifying framework to understand different reinforcement learning techniques (RFT, DPO, PPO, GRPO) used for LLM fine-tuning, showing them as variations on direct or simplified RL approaches. This framework facilitates comparative analysis and identifies key components influencing performance (data source, reward function, algorithm).

**II. Key Methods and Techniques:**

1. 
2. **Iterative Data Collection Pipeline:** This multi-stage process begins with a seed dataset (OpenWebMath) and iteratively refines a fastText classifier to identify and collect relevant mathematical content from Common Crawl.  Human annotation is employed to ensure data quality.  The iterative approach addresses the initial bias in the seed dataset, resulting in a more comprehensive and diverse corpus.
3. **Data Decontamination:**  To avoid benchmark contamination, the authors carefully filter out web pages containing content from the evaluation benchmarks themselves.  This ensures fair and unbiased evaluation.
4. **Group Relative Policy Optimization (GRPO):** This is a core innovation. By using group average rewards as a baseline, GRPO reduces the complexity and computational burden of standard PPO, which requires a separate critic network.  This makes the RL process more efficient.
5. **Outcome and Process Supervision:** GRPO is applied with both outcome supervision (reward at the end of the output) and process supervision (reward at each reasoning step).  The paper compares the effectiveness of both approaches.
6. **Iterative Reinforcement Learning:** The reward model is continuously retrained during the RL process, using a replay mechanism incorporating past data.  This iterative approach further improves performance.

**III. Evaluation and Results:**

DeepSeekMath-Base 7B and DeepSeekMath-RL 7B significantly outperform other open-source models on various mathematical reasoning benchmarks (GSM8K, MATH, CMATH, etc.), in both zero-shot and few-shot settings, often exceeding models with significantly more parameters.  The results convincingly demonstrate the effectiveness of the proposed pre-training and RL methods.  DeepSeekMath approaches the performance of closed-source models like GPT-4 and Gemini-Ultra.

**IV. Limitations and Future Work:**

The authors acknowledge that DeepSeekMath's performance on geometry problems and formal theorem proving is still behind closed-source models.  Future work will focus on improving data selection, exploring more robust RL algorithms (handling noisy rewards), and investigating advanced sampling techniques to enhance the effectiveness of reinforcement learning.

**In summary:** This paper makes significant contributions to the field of LLMs for mathematical reasoning. The large-scale, high-quality dataset, the improved pre-training strategy, and the efficient GRPO algorithm are key factors in DeepSeekMath's impressive performance. The proposed unified paradigm for RL methods provides a valuable framework for future research in this area.