class ClassifiedLLMError(RuntimeError):
    def __init__(
        self,
        *,
        client_error_msg: str,
        error_code: str,
        is_retryable: bool,
    ) -> None:
        super().__init__(client_error_msg)
        self.client_error_msg = client_error_msg
        self.error_code = error_code
        self.is_retryable = is_retryable


class LLMStreamCancelled(RuntimeError):
    """An in-flight LLM stream was aborted because the user stopped the chat.

    Not a provider error: callers (chat loop, deep research loop, research
    agents) must let it unwind to the turn runner, which treats it as the
    stop-button path instead of persisting a model error.
    """


class LLMCycleTimeoutError(ClassifiedLLMError):
    """A single chat LLM cycle exceeded its wall-clock budget.

    Raised by the step stream loop (``cycle_timeout_s``), which force-closes
    the provider stream so the model server stops generating. The chat loop
    converts it into a forced final-answer cycle — the turn keeps everything
    streamed so far and the model must answer with it. Only when the timeout
    hits an already-forced final cycle does it escape to the turn runner, so
    the client message stays honest about what happened.
    """

    def __init__(self, *, timeout_s: float) -> None:
        super().__init__(
            client_error_msg=(
                "This response was taking too long to generate and was "
                "stopped. Try regenerating or continuing the chat."
            ),
            error_code="LLM_CYCLE_TIMEOUT",
            is_retryable=True,
        )
        self.timeout_s = timeout_s
