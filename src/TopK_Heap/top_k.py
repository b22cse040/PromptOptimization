## File that maintains the Heap, will keep the top-K most relevant prompts in
## ascending order
import heapq

def score_prompt(averaged_scores: dict) -> float:
  return sum(averaged_scores.values()) / len(averaged_scores)

class TopKHeap:
  def __init__(self, k) -> None:
    self.k = k
    self.heap = []

  def __len__(self) -> int:
    return len(self.heap)

  def push(self, prompt_data: dict):
    score = score_prompt(prompt_data["scores"])
    entry = (score, prompt_data)

    if len(self.heap) < self.k:
      heapq.heappush(self.heap, entry)

    else: heapq.heappushpop(self.heap, entry)

  def get_topK(self):
    return sorted(self.heap)