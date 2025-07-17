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
    score_dict = prompt_data.get("scores", {})
    avg_score = sum(score_dict.values()) / len(score_dict) if score_dict else 0

    heapq.heappush(self.heap, (-avg_score, prompt_data))

    if len(self.heap) > self.k:
      heapq.heappop(self.heap)

  def get_topK(self):
    return [entry[1] for entry in sorted(self.heap)]