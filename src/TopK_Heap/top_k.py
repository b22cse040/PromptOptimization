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

  def _rank(self, metrics: dict):
    """ Ranking is based solely on CE_Total. """
    ce_total = metrics.get("CE_Total", 0.0)
    return float(ce_total)

  def __getitem__(self, index):
    topk = self.get_topK()
    return topk[index]