"""Base Custom LLM class for evaluation framework."""

import os
import logging
import threading
from typing import Any, Optional, Union

import litellm
from litellm.exceptions import InternalServerError

from lightspeed_evaluation.core.system.exceptions import LLMError

logger = logging.getLogger(__name__)


class TokenTracker:
    """Tracks token usage from LiteLLM calls via callbacks.

    Thread-safe implementation using per-instance callbacks. Only counts tokens
    when actively expecting a callback (after reset() is called), ensuring
    compatibility with third-party libraries (Ragas, DeepEval) that may make
    LLM calls from their own internal threads.

    Usage:
        tracker = TokenTracker()
        tracker.start()  # Register callback
        # ... make LLM calls ...
        tracker.stop()   # Unregister callback
        input_tokens, output_tokens = tracker.get_counts()
    """

    _callback_lock = (
        threading.Lock()
    )  # Class-level lock for callback list modifications

    def __init__(self) -> None:
        """Initialize token tracker."""
        self.input_tokens = 0
        self.output_tokens = 0
        self._callback_registered = False
        self._lock = threading.Lock()  # Instance lock for token counter updates
        self._pending_callbacks = 0  # Number of callbacks we're waiting for
        self._callback_condition = threading.Condition(self._lock)

    def _token_callback(
        self,
        _kwargs: dict[str, Any],
        completion_response: Any,
        _start_time: float,
        _end_time: float,
    ) -> None:
        """Capture token usage from LiteLLM completion response.

        Only counts tokens if this tracker is actively expecting a callback
        (indicated by _pending_callbacks > 0). This allows compatibility with
        third-party libraries (Ragas, DeepEval) that may make LLM calls from
        their own internal threads.
        """
        # Only count tokens if we're expecting a callback (after reset() was called)
        with self._callback_condition:
            if self._pending_callbacks <= 0:
                return

            if hasattr(completion_response, "usage") and completion_response.usage:
                usage = completion_response.usage
                self.input_tokens += getattr(usage, "prompt_tokens", 0)
                self.output_tokens += getattr(usage, "completion_tokens", 0)

            # Always decrement and notify, even if usage was missing
            self._pending_callbacks = max(0, self._pending_callbacks - 1)
            self._callback_condition.notify_all()

    def start(self) -> None:
        """Register the token tracking callback."""
        if self._callback_registered:
            return
        with TokenTracker._callback_lock:
            if (
                not hasattr(litellm, "success_callback")
                or litellm.success_callback is None
            ):
                litellm.success_callback = []
            litellm.success_callback.append(self._token_callback)
        self._callback_registered = True

    def stop(self) -> None:
        """Unregister the token tracking callback."""
        if not self._callback_registered:
            return
        with TokenTracker._callback_lock:
            if self._token_callback in litellm.success_callback:
                litellm.success_callback.remove(self._token_callback)
        self._callback_registered = False

    def get_counts(self) -> tuple[int, int]:
        """Get accumulated token counts, waiting for pending callbacks if needed.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        with self._callback_condition:
            # Wait for pending callbacks with a timeout to handle race conditions
            if self._pending_callbacks > 0:
                self._callback_condition.wait(timeout=0.1)
                # Clear stale pending state after timeout to avoid repeated waits
                self._pending_callbacks = 0
            return self.input_tokens, self.output_tokens

    def reset(self) -> None:
        """Reset token counts to zero and mark that we expect a callback."""
        with self._callback_condition:
            self.input_tokens = 0
            self.output_tokens = 0
            # Indicate we're expecting a callback after the next LLM call
            self._pending_callbacks = 1


class BaseCustomLLM:  # pylint: disable=too-few-public-methods
    """Base LLM class with core calling functionality."""

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with model configuration."""
        self.model_name = model_name
        self.llm_params = llm_params

        self.setup_ssl_verify()

        # Always drop unsupported parameters for cross-provider compatibility
        litellm.drop_params = True

    def setup_ssl_verify(self) -> None:
        """Setup SSL verification based on LLM parameters."""
        ssl_verify = self.llm_params.get("ssl_verify", True)

        if ssl_verify:
            # Use our combined certifi bundle (includes system + custom certs)
            litellm.ssl_verify = os.environ.get("SSL_CERTIFI_BUNDLE", True)
        else:
            # Explicitly disable SSL verification
            litellm.ssl_verify = False

    def call(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        return_single: bool = True,
        **kwargs: Any,
    ) -> Union[str, list[str]]:
        """Make LLM call and return response(s).

        Args:
            prompt: Text prompt to send
            n: Number of responses to generate (default 1)
            temperature: Override temperature (uses config default if None)
            return_single: If True and n=1, return single string. If False, always return list.
            **kwargs: Additional LLM parameters

        Returns:
            Single string if return_single=True and n=1, otherwise list of strings
        """
        temp = (
            temperature
            if temperature is not None
            else self.llm_params.get("temperature", 0.0)
        )

        call_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "n": n,
            "max_completion_tokens": self.llm_params.get("max_completion_tokens"),
            "timeout": self.llm_params.get("timeout"),
            "num_retries": self.llm_params.get("num_retries", 3),
            **kwargs,
        }

        try:
            response = litellm.completion(**call_params)

            # Extract content from all choices
            results = []
            for choice in response.choices:  # type: ignore
                content = choice.message.content  # type: ignore
                if content is None:
                    content = ""
                results.append(content.strip())

            # Return format based on parameters
            if return_single and n == 1:
                if not results:
                    raise LLMError("LLM returned empty response")
                return results[0]

            return results

        except InternalServerError as e:
            # Check if it's an SSL/certificate error
            error_msg = str(e)
            if "[X509]" in error_msg or "PEM lib" in error_msg:
                raise LLMError(
                    f"Judge LLM SSL certificate verification failed: {error_msg}"
                ) from e

            # Otherwise, it's a different internal server error
            raise LLMError(f"LLM internal server error: {error_msg}") from e

        except Exception as e:
            raise LLMError(f"LLM call failed: {str(e)}") from e
