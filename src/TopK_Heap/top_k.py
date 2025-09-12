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
    rank = self._rank(metrics)  # lower CE = better
    count = next(self.counter)

    # Store as (-rank, count, data) to make it a max-heap on CE
    heapq.heappush(self.heap, (-rank, count, prompt_data))

    if len(self.heap) > self.k:
        # Pop the worst (highest CE loss, i.e. most negative -rank)
        heapq.heappop(self.heap)


  def get_topK(self):
    # Sort ascending by CE loss (lowest first)
    return [entry[2] for entry in sorted(self.heap, key=lambda x: -x[0])]

  def _rank(self, metrics: dict):
    """
    Ranking is based on (lower is better):

    - CE_Total if present and numeric.
    - else average of (CE_fluency + CE_coherence + CE_consistency + CE_relevance).
    - defaults to 0.0 if no CE metrics are found.
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

if __name__ == "__main__":
  topk = TopKHeap(3)

  topk.push({"id": "A", "metrics": {"CE_fluency": 0.3}})
  topk.push({"id": "B", "metrics": {"CE_fluency": 0.7}})
  topk.push({"id": "C", "metrics": {"CE_fluency": 0.2}})
  topk.push({"id": "D", "metrics": {"CE_fluency": 0.5}})

  print([x["id"] for x in topk.get_topK()])
