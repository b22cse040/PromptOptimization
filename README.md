# PromptOptimization

## Instruction
We design a multi-objective optimization setup for evaluating and improving LLM-generated instructions that rate machine-generated summaries. The pipeline uses two distinct LLM roles:

- ```Optimizer LLM``` : This LLM is tasked to judge (LLM-as-a-Judge) the rating of the quality of machine generated summaries along 4 labels : {fluency, coherence, consistency and relevance} - each as a multi-task classification (1 | 2 | 3 | 4 | 5).
- ```Evaluator LLM``` : This LLM plays the role of an optimizer. It compares the Optimizer LLM's predicted labels with ground truth annotations and generates detailed recommendation for improving the instruction used to produce human aligned summary evaluations.

We maintain a Heap to track the top-k best performing instructions (based  on the accuracy and F1 scores across the 4 metrics). The top-k prompts are injected into the Optimizer LLM's meta-prompt in the next iteration, helping it learn from better instructions.

Over time, this loops drives the generation of improved instructions that yield more accurate and aligned predictions.

<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/7e3c2d35-0f59-41b0-aade-11d5ab46fde3" />


Evaluator LLM : openai/gpt-4.1-nano

Optimizer LLM : meta-llama/llama-3.2-3b-instruct
