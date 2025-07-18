# PromptOptimization

Using LLM to act as a judge (LLM-as-a-Judge) to improve summarization of texts over iterations, while optimizing fluency, coherence, consistency and relevance in a multi-task setting.

The setup uses two 2 LLMs: 
 - An Optimizer LLM that summarizes tasks and incorporates previous interations to improve over the previous iterations of the task (like an optimizer in classical ML setting).
 - An Evaluator LLM that acts a judge and rates theses scores between (1 - 5), and recommends improvements based on the current instruction.

We also maintain a heap (top_k_prompts) that, keeps the top-k best performing instructions in an ascending order. These are present in the meta prompt of the optimizer-LLM to improve the instruction that optimizes the performaances.
