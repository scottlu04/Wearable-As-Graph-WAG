import unittest
from unittest.mock import Mock, patch

import pandas as pd

from response_generation.core.base_processor import BaseQueryProcessor
from response_generation.config import CLIENT_GEMINI, CLIENT_OPENROUTER


class BaseQueryProcessorTests(unittest.TestCase):
    def _build_processor(self) -> BaseQueryProcessor:
        processor = BaseQueryProcessor.__new__(BaseQueryProcessor)
        processor.prompt = 'You are a helpful assistant.'
        processor.chat_model = 'gpt-4o'
        return processor

    def test_query_quick_sets_generation_called_when_response_is_requested(self) -> None:
        processor = self._build_processor()
        processor.search_knowledge_graph = Mock(return_value=({}, None))
        processor.build_context = Mock(return_value=('mock context', {}))
        processor._extract_node_names = Mock(return_value=(['Heart rate'], ['Sleep duration']))
        processor._generate_response_with_usage = Mock(return_value=(
            'mock response',
            {
                'prompt_tokens': 20,
                'completion_tokens': 2,
                'total_tokens': 22,
            },
        ))

        output, _ = processor.query_quick(
            user_query='Why is my heart rate high?',
            par_df=pd.DataFrame(),
            gt_nodes=['Heart rate'],
            query_info={'query_date': '2024-01-03', 'time_granularity': 7, 'openness': 1.0},
            dataset_name='pmdata',
            response_gen=True,
        )

        self.assertEqual(output['response'], 'mock response')
        self.assertTrue(output['generation_called'])
        self.assertEqual(output['input_tokens'], 20)
        self.assertEqual(output['output_tokens'], 2)
        self.assertEqual(output['total_tokens'], 22)
        self.assertGreaterEqual(output['retrieval_time_seconds'], 0)
        self.assertGreaterEqual(output['generation_time_seconds'], 0)
        self.assertGreaterEqual(output['e2e_time_seconds'], 0)
        processor._generate_response_with_usage.assert_called_once()

    def test_query_quick_leaves_generation_called_false_when_response_is_skipped(self) -> None:
        processor = self._build_processor()
        processor.search_knowledge_graph = Mock(return_value=({}, None))
        processor.build_context = Mock(return_value=('mock context', {}))
        processor._extract_node_names = Mock(return_value=(['Heart rate'], []))
        processor._generate_response_with_usage = Mock()

        output, _ = processor.query_quick(
            user_query='Why is my heart rate high?',
            par_df=pd.DataFrame(),
            gt_nodes=['Heart rate'],
            query_info={'query_date': '2024-01-03', 'time_granularity': 7, 'openness': 1.0},
            dataset_name='pmdata',
            response_gen=False,
        )

        self.assertIsNone(output['response'])
        self.assertFalse(output['generation_called'])
        self.assertEqual(output['output_tokens'], 0)
        self.assertGreaterEqual(output['input_tokens'], 0)
        self.assertGreaterEqual(output['retrieval_time_seconds'], 0)
        self.assertGreaterEqual(output['e2e_time_seconds'], 0)
        processor._generate_response_with_usage.assert_not_called()

    @patch('response_generation.core.base_processor.os.getenv')
    @patch('response_generation.core.base_processor.OpenAI')
    def test_initialize_chat_client_supports_openrouter_with_default_headers(
        self,
        mock_openai,
        mock_getenv,
    ) -> None:
        getenv_values = {
            'OPENROUTER_API_KEY': 'or-key',
            'OPENROUTER_HTTP_REFERER': 'https://example.com',
            'OPENROUTER_APP_TITLE': 'WAG Tests',
        }
        mock_getenv.side_effect = lambda key, default=None: getenv_values.get(key, default)

        processor = BaseQueryProcessor.__new__(BaseQueryProcessor)
        processor._initialize_chat_client(CLIENT_OPENROUTER, chat_model='anthropic/claude-3.5-sonnet')

        mock_openai.assert_called_once_with(
            api_key='or-key',
            base_url='https://openrouter.ai/api/v1',
            default_headers={
                'HTTP-Referer': 'https://example.com',
                'X-Title': 'WAG Tests',
            },
        )
        self.assertEqual(processor.chat_provider, CLIENT_OPENROUTER)
        self.assertEqual(processor.chat_api_base_url, 'https://openrouter.ai/api/v1')
        self.assertEqual(processor.chat_model, 'anthropic/claude-3.5-sonnet')

    @patch('response_generation.core.base_processor.os.getenv')
    @patch('response_generation.core.base_processor.OpenAI')
    def test_initialize_embed_client_skips_when_openai_key_is_missing(
        self,
        mock_openai,
        mock_getenv,
    ) -> None:
        mock_getenv.return_value = None

        processor = BaseQueryProcessor.__new__(BaseQueryProcessor)
        processor._initialize_embed_client('openai')

        mock_openai.assert_not_called()
        self.assertIsNone(processor.embed_client)
        self.assertEqual(processor.embed_model, 'gpt-4o-mini')

    @patch('response_generation.core.base_processor.os.getenv')
    @patch('response_generation.core.base_processor.OpenAI')
    def test_initialize_chat_client_supports_gemini(
        self,
        mock_openai,
        mock_getenv,
    ) -> None:
        mock_getenv.side_effect = lambda key, default=None: {
            'GEMINI_API_KEY': 'gemini-key',
        }.get(key, default)

        processor = BaseQueryProcessor.__new__(BaseQueryProcessor)
        processor._initialize_chat_client(CLIENT_GEMINI, chat_model='gemma-4-31b-it')

        mock_openai.assert_not_called()
        self.assertEqual(processor.chat_provider, CLIENT_GEMINI)
        self.assertIsNone(processor.chat_client)
        self.assertEqual(processor.chat_api_key, 'gemini-key')
        self.assertEqual(processor.chat_model, 'gemma-4-31b-it')

    @patch('response_generation.core.base_processor.requests.post')
    def test_gemini_inference_returns_openai_like_response(self, mock_post) -> None:
        processor = self._build_processor()
        processor.chat_provider = CLIENT_GEMINI
        processor.chat_model = 'gemma-4-31b-it'
        processor.chat_api_key = 'gemini-key'
        mock_response = Mock()
        mock_response.json.return_value = {
            'candidates': [
                {'content': {'parts': [{'text': 'mock gemini response'}]}}
            ],
            'usageMetadata': {
                'promptTokenCount': 11,
                'candidatesTokenCount': 3,
                'totalTokenCount': 14,
            },
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = processor.llm_inference(
            messages=[{'role': 'user', 'content': 'hello'}],
            tools=None,
            attempts=1,
        )

        self.assertEqual(response.choices[0].message.content, 'mock gemini response')
        self.assertEqual(response.usage['prompt_tokens'], 11)
        self.assertEqual(response.usage['completion_tokens'], 3)
        self.assertEqual(response.usage['total_tokens'], 14)


if __name__ == '__main__':
    unittest.main()
