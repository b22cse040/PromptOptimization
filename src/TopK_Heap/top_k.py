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
    metrics = prompt_data.get("metrics", {})
    rank = self._rank(metrics)
    count = next(self.counter)

    # Push with rank directly (min-heap will keep smallest at root)
    heapq.heappush(self.heap, (-rank, count, prompt_data))

    if len(self.heap) > self.k:
        # Pop the worst loss (Highest Loss CE)
        heapq.heappop(self.heap)

  def get_topK(self):
    return [entry[2] for entry in sorted(self.heap, key=lambda x : -x[0])]

  def _rank(self, metrics: dict):
    """
    Ranking is based on:

    - CE_Total if present and numeric.
    - else (CE_fluency + CE_coherence + CE_consistency + CE_relevance) / 4
    """
    if "CE_Total" in metrics:
      return float(metrics["CE_Total"])

    ce_vals = [v for k, v in metrics.items() if k.startswith("CE_")]
    if ce_vals:
      return float(sum(ce_vals)) / len(ce_vals)

    return 0.0

  def __getitem__(self, index):
    topk = self.get_topK()
    return topk[index]