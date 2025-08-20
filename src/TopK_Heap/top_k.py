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
    heapq.heappush(self.heap, (rank, count, prompt_data))

    if len(self.heap) > self.k:
        # Pop the smallest → ensures only top-k largest ranks remain
        heapq.heappop(self.heap)

  def get_topK(self):
    return [entry[2] for entry in sorted(self.heap, reverse=True)]

  def _rank(self, metrics : dict):
    """Compute the average of accuracy and f1-scores across all metric reports."""
    scores = []
    for metric_report in metrics.values():
      accuracy = metric_report.get("accuracy", 0.0)
      f1 = metric_report.get("f1", 0.0)
      avg = (accuracy + f1) / 2
      scores.append(avg)
    return sum(scores) / len(scores) if len(scores) else 0

  def __getitem__(self, index):
    topk = self.get_topK()
    return topk[index]