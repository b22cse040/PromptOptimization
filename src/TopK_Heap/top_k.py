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
    avg_f1 = self._compute_macro_f1_avg(metrics)
    count = next(self.counter)

    # Now each heap item is: (priority, tie-breaker, actual data)
    heapq.heappush(self.heap, (-avg_f1, count, prompt_data))

    if len(self.heap) > self.k:
      heapq.heappop(self.heap)

  def get_topK(self):
    return [entry[2] for entry in sorted(self.heap)]

  def _compute_macro_f1_avg(self, metrics : dict):
    """Compute the average of macro F1-scores across all metric reports."""
    macro_f1s = []
    for metric_report in metrics.values():
      f1 = metric_report.get("macro_avg", {}).get("f1-score", 0.0)
      macro_f1s.append(f1)
    return sum(macro_f1s) / len(macro_f1s) if len(macro_f1s) else 0.0
