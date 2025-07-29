# PromptOptimization

## Instruction
We design a multi-objective optimization setup for evaluating and improving LLM-generated instructions that rate machine-generated summaries. The pipeline uses two distinct LLM roles:

- ```Rater LLM``` : This LLM is tasked to judge (LLM-as-a-Judge) the rating of the quality of machine generated summaries along 4 labels : {fluency, coherence, consistency and relevance} - each as a multi-task classification (1 | 2 | 3 | 4 | 5).
- ```Recommender LLM``` : This LLM plays the role of an optimizer. It compares the rater LLM's previous instruction along with it's performances on (f1, accuracy) and generates detailed recommendation for improving the instruction used to produce human aligned summary evaluations.

We maintain a Heap to track the top-k best performing instructions (based  on the accuracy and F1 scores across the 4 metrics). The top-k prompts are injected into the Optimizer LLM's meta-prompt in the next iteration, helping it learn from better instructions. For current expermiment we have used k = 10.

Over time, this loops drives the generation of improved instructions that yield more accurate and aligned predictions.

### F1 plot
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/53f3f9a9-c603-4199-81ac-5a27366c6b03" />
### Accuracy Plot
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/22111b3f-ff2a-4dfc-ba0f-33f502000eaa" />


Evaluator LLM : meta-llama/llama-3-8b-instruct

Optimizer LLM : meta-llama/llama-3-8b-instruct
