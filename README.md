# ReShARC: Improving ARC-AGI-1 Performance

**Paper:** Improving ARC-AGI-1 performance through inference-time strategies and prompt engineering
**Authors:** Jason Chan, Aaron Fletcher (School of Computer Science, Sheffield University)
**Contact:** jlychan1@sheffield.ac.uk, ahafletcher1@sheffield.ac.uk
**Repository:** [https://github.com/afletcher53/resharc](https://github.com/afletcher53/resharc) 

## Abstract

This project investigates strategies to enhance the performance of Large Language Models (LLMs) on the *Abstraction and Reasoning Corpus* (ARC-AGI-1). The ARC challenges AI systems with tasks requiring genuine reasoning from very few examples, a domain where current LLMs often falter. This work introduces more granular evaluation metrics—Grid Dimension Match (GDM), Pixel Accuracy (PA), and Foreground Pixel Accuracy (FPA)—alongside the standard Exact Match (EM). Baselines for various LLMs are established using Chain-of-Thought (CoT) prompting. The study further explores inference-time prompt engineering techniques, specifically 'Reflection' and 'Repeating', demonstrating their ability to improve EM scores and partial credit metrics on OpenAI's o4 Mini (Medium) model without model retraining. These findings highlight the importance of nuanced evaluation and showcase how simple, cognitively-inspired prompting can significantly boost LLM reasoning on complex ARC tasks.

## Key Features & Contributions

* **Granular Evaluation Metrics:** Introduces and utilizes GDM, PA, and FPA to provide a more detailed understanding of model performance beyond binary EM.
* **LLM Baselines for ARC-AGI-1:** Establishes performance benchmarks for several contemporary LLMs using CoT prompting on the `eval-pub` dataset.
* **Inference-Time Prompt Engineering:**
    * Investigates 'Reflection' and 'Repeating' techniques to improve LLM reasoning.
    * 'Repeating' achieved a 10.4% relative EM increase (from 0.163 to 0.180) on the o4 Mini (Medium) model.
    * 'Reflection' consistently enhanced all partial credit metrics for the same model.
* **Analysis of Model Behavior:** Observes that newer models may exhibit "all-or-nothing" performance or show signs of potential dataset contamination.

## Methodology

### Models Evaluated

* Llama-3.2-1B-Instruct
* Llama-3.1-8B-Instruct
* OpenAI gpt-3.5-turbo
* OpenAI gpt-4o
* OpenAI o4 mini-(Medium)
* Google Gemini 2.5 Flash (Thinking)

### Dataset

All experiments were conducted on the 400 evaluation samples in the ARC corpus, specifically the **`eval-pub`** split. Raw responses from each model can be found [here](https://github.com/afletcher53/ReSharc/tree/main/data/generated_sft/prompt_engineering).
* Original ARC dataset files (including training, evaluation, and test challenges/solutions) can be found in `data/arc/`.
* Augmented dataset generation was explored (see `data/augmented_ds/` and Appendix A of the paper).


### Prompting Strategies

* **Chain-of-Thought (CoT):** Baseline prompting strategy.
* **CoT + Reflection:** LLM reviews and refines its own generated thinking.
* **CoT + Repeating:** LLM re-reads the question before answering.

*(Refer to Appendix A in the paper for detailed prompt structures.)*
### Evaluation Metrics

A prediction grid is defined as $P$, and the ground truth solution grid as $S$. A prediction $P$ is considered valid if it is a non-jagged grid.

1.  **Grid Dimension Match (GDM):** Assesses if the predicted output grid has the correct dimensions.

    $$
    \textnormal{GDM} =
    \begin{cases}
    1 & \textnormal{if } P \textnormal{ is valid and } \textnormal{shape}(P) = \textnormal{shape}(S) \\
    0 & \textnormal{otherwise}
    \end{cases}
    $$

2.  **Pixel Accuracy (PA):** Measures the fraction of correctly predicted cells, given correct dimensions ($N$ is the total number of cells).

    $$
    \textnormal{PA} =
    \begin{cases}
    \frac{1}{N} \sum_{i,j} I(P_{i,j} = S_{i,j}) & \textnormal{if GDM} = 1 \\
    0 & \textnormal{otherwise}
    \end{cases}
    $$
    Where $I(\cdot)$ is the indicator function.

3.  **Foreground Pixel Accuracy (FPA):** Focuses on the accuracy of non-background (non-zero) cells in the solution $S$, given correct dimensions.

    $$
    \textnormal{FPA} =
    \begin{cases}
    \frac{\sum_{i,j} I(P_{i,j} = S_{i,j} \textnormal{ and } S_{i,j} \neq 0)}{\sum_{i,j} I(S_{i,j} \neq 0)} & \textnormal{if GDM}=1 \textnormal{ and } \sum_{i,j} I(S_{i,j} \neq 0) > 0 \\
    0 & \textnormal{otherwise}
    \end{cases}
    $$

4.  **Exact Match (EM):** The prediction must be identical to the solution in dimensions and content.

    $$
    \textnormal{EM} =
    \begin{cases}
    1 & \textnormal{if } P \textnormal{ is valid and } P = S \\
    0 & \textnormal{otherwise}
    \end{cases}
    $$

## Results Highlights

* **OpenAI o4 Mini (Medium) with CoT + Repeating:** Achieved an EM score of 0.180 (a 10.4% relative increase over baseline CoT's 0.163). GDM: 0.180, PA: 0.172, FPA: 0.169.
* **OpenAI o4 Mini (Medium) with CoT + Reflection:** Achieved an EM score of 0.170. GDM: 0.205, PA: 0.196, FPA: 0.191. This strategy consistently improved all partial credit metrics.

*(Refer to Table 1 and Table 2 in the paper for full baseline and prompting technique comparison results.)*

## Explored & Future Directions

* **Data Augmentation:** Geometric and color augmentations were explored to expand the training dataset (see `src/arc_utils.py` for potential implementation and Appendix A of the paper). The code for the original shARC augmentation approach is available at [https://github.com/afletcher53/shARC](https://github.com/afletcher53/shARC).
* **Supervised Fine-Tuning (SFT):** Investigated but faced hardware and context length constraints. Relevant data for SFT can be found in `data/filtered_sft/` and `data/generated_sft/`.
* **Future Work:**
    * Redefining color representations for broader augmentation.
    * Developing compute-aware metrics (e.g., EM per 1k tokens).
    * Combining prompting heuristics with self-consistency ensembles or test-time fine-tuning.
