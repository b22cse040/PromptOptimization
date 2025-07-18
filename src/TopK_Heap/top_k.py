import heapq
import itertools

class TopKHeap:
  def __init__(self, k) -> None:
    self.k = k
    self.heap = []
    self.counter = itertools.count()  # unique counter to break ties

  def __len__(self) -> int:
    return len(self.heap)

  def push(self, prompt_data: dict):
    score_dict = prompt_data.get("scores", {})
    avg_score = sum(score_dict.values()) / len(score_dict) if score_dict else 0
    count = next(self.counter)

    # Now each heap item is: (priority, tie-breaker, actual data)
    heapq.heappush(self.heap, (-avg_score, count, prompt_data))

    if len(self.heap) > self.k:
      heapq.heappop(self.heap)

  def get_topK(self):
    return [entry[2] for entry in sorted(self.heap)]