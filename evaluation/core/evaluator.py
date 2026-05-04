"""
Evaluator Module

This module implements LLM-as-a-judge evaluation for retrieval-augmented generation
(RAG) and knowledge graph-based systems. It provides three specialized evaluator classes
for different evaluation scenarios:

1. BaseEvaluator: Compares baseline, RAG, and KGRAG methods head-to-head
2. ContextEvaluator: Ranks retrieved context quality across methods
3. ResponseEvaluator: Scores generated response quality across methods

The evaluation process uses LLMs (DeepSeek, OpenAI, OpenRouter, or Gemini) to assess quality
through structured prompts and randomized method ordering to minimize bias.

Key Features:
    - Multi-provider LLM support (DeepSeek, OpenAI, OpenRouter, Gemini)
    - Automatic retry mechanism for robust inference
    - Bias mitigation through random method ordering
    - Structured JSON response parsing
    - Comprehensive logging and error handling

Architecture:
    1. Initialize evaluator with LLM provider and evaluation prompt
    2. Format input data with query and method outputs
    3. Randomize method order (for context/response evaluation)
    4. Send to LLM with structured prompt
    5. Parse JSON response
    6. Map results back to original method names

Usage Example:
    >>> from evaluation import BaseEvaluator, ContextEvaluator, ResponseEvaluator
    >>>
    >>> # Compare three methods
    >>> evaluator = BaseEvaluator(chat_client="deepseek")
    >>> result = {
    ...     'question': "What is my average heart rate?",
    ...     'baseline': {'full_context': "...", 'response': "..."},
    ...     'rag': {'full_context': "...", 'response': "..."},
    ...     'kgrag': {'full_context': "...", 'response': "..."}
    ... }
    >>> evaluated_result, scores = evaluator.evaluate(result)
    >>>
    >>> # Rank contexts
    >>> context_eval = ContextEvaluator(chat_client="openai")
    >>> contexts = {'method_a': "...", 'method_b': "...", 'method_c': "..."}
    >>> ranking, details, _ = context_eval.evaluate(query, contexts)
    >>> print(f"Best method: {ranking[0]}")
    >>>
    >>> # Score responses
    >>> response_eval = ResponseEvaluator(chat_client="gemini")
    >>> responses = {'method_a': "...", 'method_b': "...", 'method_c': "..."}
    >>> scores = response_eval.evaluate(query, responses)

Output Structure:
    BaseEvaluator:
        result = {
            'question': str,
            'baseline': {'full_context': str, 'response': str, 'eval': score},
            'rag': {'full_context': str, 'response': str, 'eval': score},
            'kgrag': {'full_context': str, 'response': str, 'eval': score}
        }

    ContextEvaluator:
        ranking = ['method_c', 'method_a', 'method_b']  # Best to worst
        details = {'ranking': [...], 'justification': {...}}

    ResponseEvaluator:
        scores = {
            'method_a': {'relevance': 4, 'completeness': 5, 'overall_quality': 1},
            'method_b': {'relevance': 3, 'completeness': 3, 'overall_quality': 2},
            ...
        }

Dependencies:
    - openai: For OpenAI-compatible API access (OpenAI, DeepSeek, OpenRouter)
    - google.generativeai: For Gemini API access
    - utils.parse: For JSON response parsing
    - shared.prompts.Eval: For evaluation prompt templates

"""

from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import random
import time
from types import SimpleNamespace

from openai import OpenAI
import requests

from ..config.settings import config
from shared.prompts.Eval import EVAL_PROMPT_CONTEXT, EVAL_PROMPT_RANK
from shared.utils import parse_llm_response

# Configure logging
logger = logging.getLogger(__name__)


class _LLMJudgeMixin:
    """Shared provider initialization and inference helpers for evaluators."""

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "429" in message
            or "rate limit" in message
            or "rate-limit" in message
            or "rate-limited" in message
        )

    @staticmethod
    def _sleep(seconds: float, reason: str) -> None:
        if seconds <= 0:
            return
        logger.info("Sleeping %.1fs before %s", seconds, reason)
        time.sleep(seconds)

    @staticmethod
    def _prepare_gemini_contents(
        messages: Union[List[Dict[str, str]], str]
    ) -> Any:
        """Convert OpenAI-style chat messages into Gemini-compatible contents."""
        if isinstance(messages, str):
            return messages

        if not isinstance(messages, list):
            return messages

        normalized_contents = []
        for message in messages:
            if not isinstance(message, dict):
                normalized_contents.append(message)
                continue

            role = message.get("role", "user")
            if "parts" in message:
                parts = message["parts"]
            else:
                content = message.get("content", "")
                parts = [content] if content else []

            normalized_contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            })

        # All current evaluation calls send a single user prompt. Returning the
        # bare prompt string keeps Gemini SDK handling simple and robust.
        if (
            len(normalized_contents) == 1
            and normalized_contents[0].get("role") == "user"
            and len(normalized_contents[0].get("parts", [])) == 1
        ):
            return normalized_contents[0]["parts"][0]

        return normalized_contents

    @staticmethod
    def _build_gemini_payload(contents: Any) -> Dict[str, Any]:
        """Convert normalized Gemini contents into the REST payload schema."""
        if isinstance(contents, str):
            return {
                "contents": [
                    {
                        "parts": [{"text": contents}],
                    }
                ]
            }

        payload_contents = []
        for item in contents:
            if isinstance(item, str):
                payload_contents.append({"parts": [{"text": item}]})
                continue

            if not isinstance(item, dict):
                payload_contents.append({"parts": [{"text": str(item)}]})
                continue

            parts = []
            for part in item.get("parts", []):
                if isinstance(part, str):
                    parts.append({"text": part})
                else:
                    parts.append(part)

            payload_item = {"parts": parts}
            role = item.get("role")
            if role:
                payload_item["role"] = role
            payload_contents.append(payload_item)

        return {"contents": payload_contents}

    @staticmethod
    def _extract_gemini_text(data: Dict[str, Any]) -> str:
        """Extract plain-text content from Gemini REST responses."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text_chunks = []
        for part in parts:
            if isinstance(part, dict) and "text" in part:
                text_chunks.append(part["text"])
            elif isinstance(part, str):
                text_chunks.append(part)
        return "".join(text_chunks)

    def _initialize_llm_client(
        self,
        chat_client: str,
        chat_model: Optional[str] = None,
        evaluator_label: str = "evaluator",
    ) -> None:
        """Initialize the requested LLM provider."""
        self.chat_client = chat_client
        self.chat_api_base_url = config.get_api_url(chat_client)
        self.chat_api_key = config.get_api_key(chat_client)

        if chat_client == config.CLIENT_GEMINI:
            if not self.chat_api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable not set")
            self.client = None
        else:
            if not self.chat_api_key:
                env_var_name = {
                    config.CLIENT_DEEPSEEK: "DEEPSEEK_API_KEY",
                    config.CLIENT_OPENAI: "OPENAI_API_KEY",
                    config.CLIENT_OPENROUTER: "OPENROUTER_API_KEY",
                }[chat_client]
                raise RuntimeError(f"{env_var_name} environment variable not set")

            client_kwargs = {'api_key': self.chat_api_key}
            if self.chat_api_base_url and chat_client != config.CLIENT_OPENAI:
                client_kwargs['base_url'] = self.chat_api_base_url
            if chat_client == config.CLIENT_OPENROUTER:
                headers = config.get_openrouter_headers()
                if headers:
                    client_kwargs['default_headers'] = headers

            self.client = OpenAI(**client_kwargs)

        self.chat_model = chat_model or config.get_chat_model(chat_client)
        logger.info(
            "Initialized %s %s with model: %s",
            chat_client,
            evaluator_label,
            self.chat_model,
        )

    def llm_inference(
        self,
        messages: Union[List[Dict[str, str]], str],
        attempts: int = config.MAX_INFERENCE_ATTEMPTS
    ) -> Any:
        """Perform LLM inference with automatic retry."""
        for attempt in range(attempts):
            try:
                self._sleep(config.REQUEST_DELAY_SECONDS, "LLM request")
                if self.chat_client == config.CLIENT_GEMINI:
                    payload = self._build_gemini_payload(
                        self._prepare_gemini_contents(messages)
                    )
                    response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.chat_model}:generateContent",
                        params={"key": self.chat_api_key},
                        json=payload,
                        timeout=config.DEFAULT_TIMEOUT_SECONDS,
                    )
                    response.raise_for_status()
                    text = self._extract_gemini_text(response.json())
                    if text:
                        logger.debug(
                            "%s inference successful on attempt %s",
                            self.chat_client,
                            attempt + 1,
                        )
                        return SimpleNamespace(text=text)
                    logger.warning(
                        "Attempt %s failed: Empty %s response",
                        attempt + 1,
                        self.chat_client,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.chat_model,
                        messages=messages,
                        temperature=config.TEMPERATURE,
                        timeout=config.DEFAULT_TIMEOUT_SECONDS,
                    )
                    if response and response.choices:
                        logger.debug(
                            "%s inference successful on attempt %s",
                            self.chat_client,
                            attempt + 1,
                        )
                        return response
                    logger.warning("Attempt %s failed: Empty or invalid response", attempt + 1)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < attempts - 1:
                    delay = (
                        config.RATE_LIMIT_BACKOFF_SECONDS
                        if self._is_rate_limit_error(e)
                        else config.RETRY_DELAY_SECONDS
                    )
                    self._sleep(delay, "retry")

        raise RuntimeError(
            f"LLM inference failed after {attempts} attempts with {self.chat_client}"
        )

    def _response_to_text(self, response: Any) -> str:
        """Extract plain-text content from provider-specific response objects."""
        if self.chat_client == config.CLIENT_GEMINI:
            return response.text
        return response.choices[0].message.content


class BaseEvaluator(_LLMJudgeMixin):
    """
    Base evaluator for comparing baseline, RAG, and KGRAG methods.

    This evaluator assesses the quality of three methods (baseline, RAG, KGRAG)
    by having an LLM judge both the retrieved context and generated response
    for a given query.

    Attributes:
        chat_client: LLM provider name ("deepseek", "openai", "gemini")
        chat_model: Model identifier for the chosen provider
        prompt: Evaluation prompt template
        client: Initialized OpenAI or Gemini client

    Example:
        >>> evaluator = BaseEvaluator(chat_client="deepseek")
        >>> result = evaluator.evaluate(result_dict)
        >>> print(f"Best method: {min(result, key=lambda m: result[m]['eval'])}")
    """

    def __init__(
        self,
        chat_client: str = "deepseek",
        chat_model: Optional[str] = None,
        eval_prompt: str = EVAL_PROMPT_RANK,
    ):
        """
        Initialize the BaseEvaluator.

        Args:
            chat_client: LLM provider to use. Options: "deepseek", "openai", "openrouter", "gemini"
            chat_model: Optional explicit model override
            eval_prompt: Evaluation prompt template

        Raises:
            ValueError: If chat_client is not supported
            RuntimeError: If required API keys are missing
        """
        self.prompt = eval_prompt
        self._initialize_llm_client(
            chat_client=chat_client,
            chat_model=chat_model,
            evaluator_label="client",
        )

    def get_input_data(self, result: Dict[str, Any]) -> Tuple[str, str]:
        """
        Extract context and answer from result dictionary.

        Args:
            result: Result dictionary containing 'full_context' and 'response'

        Returns:
            Tuple of (context, answer)
        """
        context = result.get('full_context', '')
        answer = result.get('response', '')
        return context, answer

    def evaluate(self, result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate baseline, RAG, and KGRAG methods for a single query.

        Args:
            result: Dictionary containing:
                - question: The query string
                - baseline: Dict with 'full_context' and 'response'
                - rag: Dict with 'full_context' and 'response'
                - kgrag: Dict with 'full_context' and 'response'

        Returns:
            Tuple of (updated_result, raw_scores):
                - updated_result: Input dict with 'eval' scores added to each method
                - raw_scores: Dict with evaluation details from LLM

        Example:
            >>> result = {
            ...     'question': "What is my average heart rate?",
            ...     'baseline': {'full_context': "...", 'response': "..."},
            ...     'rag': {'full_context': "...", 'response': "..."},
            ...     'kgrag': {'full_context': "...", 'response': "..."}
            ... }
            >>> updated, scores = evaluator.evaluate(result)
            >>> print(scores['result_1'])  # Baseline score
        """
        logger.info(f"Evaluating query: {result.get('question', 'N/A')[:50]}...")

        # Build input data for LLM
        input_data = {"Query": result['question']}

        # Add baseline method (result_1)
        context, answer = self.get_input_data(result['baseline'])
        input_data['result_1'] = {
            "retrieved_content": context,
            "response": answer
        }

        # Add RAG method (result_2)
        context, answer = self.get_input_data(result['rag'])
        input_data['result_2'] = {
            "retrieved_content": context,
            "response": answer
        }

        # Add KGRAG method (result_3)
        context, answer = self.get_input_data(result['kgrag'])
        input_data['result_3'] = {
            "retrieved_content": context,
            "response": answer
        }

        # Construct prompt
        prompt = self.prompt + "\nInput:\n" + json.dumps(input_data, indent=2) + "\nOutput:"

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Perform LLM inference
        try:
            response = self.llm_inference(messages=messages)

            # Extract content based on client type
            content = self._response_to_text(response)

            # Parse LLM response
            output_llm = parse_llm_response(content)

            # Add evaluation scores to result
            result['baseline']['eval'] = output_llm.get('result_1', {})
            result['rag']['eval'] = output_llm.get('result_2', {})
            result['kgrag']['eval'] = output_llm.get('result_3', {})

            logger.info("Evaluation completed successfully")
            return result, output_llm

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


class ContextEvaluator(_LLMJudgeMixin):
    """
    Evaluator for ranking retrieved context quality across methods.

    This evaluator compares the quality of retrieved context from different methods
    by having an LLM judge which context is most relevant, comprehensive, and useful
    for answering the query. Methods are randomized to avoid position bias.

    Attributes:
        chat_client: LLM provider name
        chat_model: Model identifier
        prompt: Evaluation prompt template
        client: Initialized client
        method_mapping: Mapping from anonymous names to original method names

    Example:
        >>> evaluator = ContextEvaluator(chat_client="openai")
        >>> contexts = {
        ...     'baseline': "Heart rate: 72 bpm",
        ...     'rag': "Heart rate: 72 bpm, Sleep: 7.5 hrs",
        ...     'kgrag': "Heart rate: 72 bpm (resting), Sleep: 7.5 hrs, Steps: 8500"
        ... }
        >>> ranking, details, input_data = evaluator.evaluate(
        ...     "Why am I feeling tired?", contexts
        ... )
        >>> print(ranking)  # ['kgrag', 'rag', 'baseline']
    """

    def __init__(
        self,
        chat_client: str = "deepseek",
        chat_model: Optional[str] = None,
        eval_prompt: str = EVAL_PROMPT_CONTEXT,
    ):
        """
        Initialize the ContextEvaluator.

        Args:
            chat_client: LLM provider to use. Options: "deepseek", "openai", "openrouter", "gemini"
            chat_model: Optional explicit model override
            eval_prompt: Evaluation prompt template for context ranking

        Raises:
            ValueError: If chat_client is not supported
            RuntimeError: If required API keys are missing
        """
        self.prompt = eval_prompt
        self._initialize_llm_client(
            chat_client=chat_client,
            chat_model=chat_model,
            evaluator_label="context evaluator",
        )

    def evaluate(
        self,
        query: str,
        context_dict: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate and rank retrieved contexts from different methods.

        Args:
            query: The user query string
            context_dict: Dictionary mapping method names to retrieved context strings

        Returns:
            Tuple of (ranking, details, input_data):
                - ranking: List of method names ordered from best to worst
                - details: Dict with LLM evaluation details including justifications
                - input_data: The formatted input sent to the LLM

        Example:
            >>> contexts = {
            ...     'baseline': "Heart rate: 72 bpm",
            ...     'rag': "Heart rate: 72 bpm, Sleep quality: Good",
            ...     'kgrag': "Heart rate: 72 bpm (normal), Sleep: 7.5 hrs (85% efficiency)"
            ... }
            >>> ranking, details, _ = evaluator.evaluate("Am I sleeping well?", contexts)
            >>> print(ranking[0])  # Best method
            'kgrag'
        """
        logger.info(f"Evaluating contexts for query: {query[:50]}...")

        # Build input data with randomized method order
        input_data = {"query": query}
        methods = list(context_dict.keys())

        # Randomize to avoid position bias
        random.shuffle(methods)
        logger.debug(f"Randomized method order: {methods}")

        input_data["methods"] = {}
        method_mapping = {}

        for i, method in enumerate(methods):
            method_name = f"method_{i+1}"
            method_mapping[method_name] = method
            context = context_dict[method]
            input_data["methods"][method_name] = {"retrieved_content": context}

        # Construct prompt
        prompt = self.prompt + "\nInput:\n" + json.dumps(input_data, indent=2) + "\nOutput:"

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Perform LLM inference
        try:
            response = self.llm_inference(messages=messages)

            # Extract content
            content = self._response_to_text(response)

            # Parse LLM response
            output_llm = parse_llm_response(content)

            # Convert ranking back to original method names
            ranking = [method_mapping[method] for method in output_llm.get('ranking', [])]

            logger.info(f"Context ranking: {ranking}")
            return ranking, output_llm, input_data

        except Exception as e:
            logger.error(f"Context evaluation failed: {e}")
            raise


class ResponseEvaluator(_LLMJudgeMixin):
    """
    Evaluator for scoring generated response quality across methods.

    This evaluator assesses the quality of generated responses from different methods
    by having an LLM judge multiple dimensions (relevance, completeness, accuracy, etc.).
    Methods are randomized to avoid position bias, then mapped back to original names.

    Attributes:
        chat_client: LLM provider name
        chat_model: Model identifier
        prompt: Evaluation prompt template
        client: Initialized client

    Example:
        >>> evaluator = ResponseEvaluator(chat_client="deepseek")
        >>> responses = {
        ...     'baseline': "Your heart rate is 72 bpm.",
        ...     'rag': "Your heart rate is 72 bpm, which is normal.",
        ...     'kgrag': "Your resting heart rate is 72 bpm, which is in the healthy range (60-100 bpm)."
        ... }
        >>> scores = evaluator.evaluate("What is my heart rate?", responses)
        >>> for method, score in scores.items():
        ...     print(f"{method}: {score['overall_quality']}")
    """

    def __init__(
        self,
        chat_client: str = "deepseek",
        chat_model: Optional[str] = None,
        eval_prompt: str = EVAL_PROMPT_RANK,
    ):
        """
        Initialize the ResponseEvaluator.

        Args:
            chat_client: LLM provider to use. Options: "deepseek", "openai", "openrouter", "gemini"
            chat_model: Optional explicit model override
            eval_prompt: Evaluation prompt template for response scoring

        Raises:
            ValueError: If chat_client is not supported
            RuntimeError: If required API keys are missing
        """
        self.prompt = eval_prompt
        self._initialize_llm_client(
            chat_client=chat_client,
            chat_model=chat_model,
            evaluator_label="response evaluator",
        )

    def evaluate(
        self,
        query: str,
        response_dict: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate and score generated responses from different methods.

        Args:
            query: The user query string
            response_dict: Dictionary mapping method names to generated response strings

        Returns:
            Dictionary mapping method names to their evaluation scores.
            Each score dict contains dimensions like:
                - relevance: Score for query relevance
                - completeness: Score for answer completeness
                - accuracy: Score for factual accuracy
                - overall_quality: Overall quality ranking (1=best)

        Example:
            >>> responses = {
            ...     'baseline': "72 bpm",
            ...     'rag': "Your heart rate is 72 bpm, which is normal",
            ...     'kgrag': "Your resting heart rate is 72 bpm (normal range: 60-100 bpm)"
            ... }
            >>> scores = evaluator.evaluate("What is my heart rate?", responses)
            >>> print(scores['kgrag']['overall_quality'])  # 1 (best)
        """
        logger.info(f"Evaluating responses for query: {query[:50]}...")

        # Build input data with randomized method order
        input_data = {"query": query}
        methods = list(response_dict.keys())

        # Randomize to avoid position bias
        random.shuffle(methods)
        logger.debug(f"Randomized method order: {methods}")

        input_data["methods"] = {}
        method_mapping = {}

        for i, method in enumerate(methods):
            method_name = f"method_{i+1}"
            method_mapping[method_name] = method
            response = response_dict[method]
            input_data["methods"][method_name] = {"response": response}

        # Construct prompt
        prompt = self.prompt + "\nInput:\n" + json.dumps(input_data, indent=2) + "\nOutput:"

        # Prepare messages
        messages = [{"role": "user", "content": prompt}]

        # Perform LLM inference
        try:
            response = self.llm_inference(messages=messages)
            content = self._response_to_text(response)

            # Parse LLM response
            output_llm = parse_llm_response(content)
            if not isinstance(output_llm, dict):
                snippet = content[:500].replace("\n", " ")
                raise ValueError(
                    "Could not parse judge response as a JSON object. "
                    f"Response snippet: {snippet}"
                )

            # Map scores back to original method names
            renamed_output = {}
            for method_name, scores in output_llm.items():
                if method_name in method_mapping:
                    original_name = method_mapping[method_name]
                    renamed_output[original_name] = scores
                else:
                    logger.warning(f"Unknown method in LLM output: {method_name}")

            missing_methods = set(response_dict) - set(renamed_output)
            if missing_methods:
                raise ValueError(
                    "Judge response missing scores for methods: "
                    f"{sorted(missing_methods)}"
                )

            logger.info(f"Response evaluation completed for {len(renamed_output)} methods")
            return renamed_output

        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            raise


# Legacy aliases for backward compatibility
Evaluater = BaseEvaluator
Evaluater_Context = ContextEvaluator
Evaluater_Response = ResponseEvaluator
