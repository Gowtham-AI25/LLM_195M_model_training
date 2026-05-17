import threading
import queue


class AsyncWandBLogger:
    """
    Asynchronous wrapper over a base WandB logger.

    Core Idea:
    ----------
    - Training thread produces logging tasks
    - Queue stores tasks (buffer)
    - Background thread consumes tasks and executes logging

    This ensures:
    - Training never waits for logging
    - Logging runs in parallel
    """

    def __init__(self, base_logger, max_queue_size: int = 1000):
        """
        Initialize async logger.

        Parameters:
        ----------
        base_logger : WandBLogger
            The actual logger that performs wandb.log()

        max_queue_size : int
            Maximum number of pending tasks allowed in queue
        """

        # Underlying logger (actual wandb calls happen here)
        self.base_logger = base_logger

        # Queue acts as buffer between training and logging
        self.queue = queue.Queue(maxsize=max_queue_size)

        # Background worker thread
        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True  # stops automatically when main program exits
        )

        # Start worker immediately
        self.worker_thread.start()

    # =========================================================
    # 🔹 WORKER (CONSUMER)
    # =========================================================
    def _worker(self):
        """
        Infinite loop that:
        - waits for tasks
        - executes them
        - stops only when receiving a termination signal (None)
        """

        while True:

            # Wait for next task (blocking call)
            task = self.queue.get()

            # 🔴 Termination signal
            if task is None:
                break

            # Unpack task
            fn, kwargs = task

            # Execute logging function
            try:
                fn(**kwargs)
            except Exception as e:
                print(f"[AsyncLogger Error] {e}")

    # =========================================================
    # 🔹 TASK SUBMISSION (PRODUCER SIDE)
    # =========================================================
    def _enqueue(self, fn, **kwargs):
        """
        Push a logging task into queue.

        Each task = (function, arguments)

        Example:
        --------
        (log_step, {"step": 100, "metrics": {...}})
        """

        # Avoid blocking if queue is full
        if not self.queue.full():
            self.queue.put((fn, kwargs))
        else:
            # Optional: drop task or handle overflow
            pass

    # =========================================================
    # 🟢 STEP LOGGING
    # =========================================================
    def log_step(self, **kwargs):
        """
        Logs step-level metrics (called every step).
        """
        self._enqueue(self.base_logger.log_step, **kwargs)

    # =========================================================
    # 🟡 INTERVAL LOGGING
    # =========================================================
    def log_interval(self, **kwargs):
        """
        Logs optimizer + gradient metrics.
        """
        self._enqueue(self.base_logger.log_interval, **kwargs)

    # =========================================================
    # 🔷 LOGIT LOGGING
    # =========================================================
    def log_logits(self, **kwargs):
        """
        Logs output distribution metrics.
        """
        self._enqueue(self.base_logger.log_logits, **kwargs)

    # =========================================================
    # 🔷 EMBEDDING LOGGING
    # =========================================================
    def log_embedding(self, **kwargs):
        """
        Logs embedding diagnostics.
        """
        self._enqueue(self.base_logger.log_embedding, **kwargs)

    # =========================================================
    # 🔵 HISTOGRAM LOGGING
    # =========================================================
    def log_histograms(self, **kwargs):
        """
        Logs weight/gradient distributions.
        """
        self._enqueue(self.base_logger.log_histograms, **kwargs)

    # =========================================================
    # 🔴 VALIDATION LOGGING
    # =========================================================
    def log_validation(self, **kwargs):
        """
        Logs validation metrics.
        """
        self._enqueue(self.base_logger.log_validation, **kwargs)

    # =========================================================
    # 🔚 CLEAN SHUTDOWN
    # =========================================================
    def finish(self):
        """
        Gracefully shutdown logger.

        Steps:
        ------
        1. Send termination signal to worker
        2. Wait for worker to finish remaining tasks
        3. Close base logger (wandb)
        """

        # Send stop signal
        self.queue.put(None)

        # Wait until worker completes all tasks
        self.worker_thread.join()

        # Finalize base logger
        self.base_logger.finish()