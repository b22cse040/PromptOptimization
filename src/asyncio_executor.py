"""A collection of concurrency utilities to augment the Python language:"""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Semaphore
from typing import Optional


class RestrictedConcurrencyThreadPoolExecutor(ThreadPoolExecutor):
  """
  This executor restricts concurrency (max active threads) and, optionally, rate (max calls per second).
  It is similar in functionality to the @concurrent decorator, but implemented at the executor level.
  """

  def __init__(
    self,
    max_workers: Optional[int] = None,
    *args,
    max_calls_per_second: float = float("inf"),
    **kwargs,
  ):
    if max_workers is None:
      max_workers: int = 24
    if not isinstance(max_workers, int) or (max_workers < 1):
      raise ValueError("Expected `max_workers`to be a non-negative integer.")
    kwargs["max_workers"] = max_workers
    super().__init__(*args, **kwargs)
    self._semaphore = Semaphore(max_workers)
    self._max_calls_per_second = max_calls_per_second

    # If we have an infinite rate, don't enforce a delay
    self._min_time_interval_between_calls = 1 / self._max_calls_per_second

    # Tracks the last time a call was started (not finished, just started)
    self._time_last_called = 0.0
    self._lock = Lock()  # Protects access to _time_last_called

  def submit(self, fn, *args, **kwargs):

    # Rate limiting logic: Before starting a new call, ensure we wait long enough if needed
    if self._min_time_interval_between_calls > 0.0:
      with self._lock:
        time_elapsed_since_last_called = time.time() - self._time_last_called
        time_to_wait = max(
          0.0,
          self._min_time_interval_between_calls
          - time_elapsed_since_last_called,
          )

        # Wait the required time
        if time_to_wait > 0:
          time.sleep(time_to_wait)

        # Update the last-called time after the wait
        # with self._lock:
        self._time_last_called = time.time()

    # Enforce concurrency limit
    self._semaphore.acquire()

    try:
      future = super().submit(fn, *args, **kwargs)
      future.add_done_callback(lambda _: self._semaphore.release())
      return future
    except Exception as e:
      self._semaphore.release()
      raise e