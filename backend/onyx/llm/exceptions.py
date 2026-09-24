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
