import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from evaluation.config.settings import config
from evaluation.core.evaluator import ContextEvaluator, ResponseEvaluator
from evaluation.pipelines.evaluation_pipeline import EvaluationPipeline
from evaluation.utils.evaluation_utils import calculate_wr
from shared.utils.parse import parse_llm_response


class DummyEvaluator:
    def __init__(self, chat_client: str, chat_model: str) -> None:
        self.chat_client = chat_client
        self.chat_model = chat_model

    def evaluate(self, query, response_dict):
        return {
            method: {'overall_quality': index + 1}
            for index, method in enumerate(response_dict.keys())
        }


class EvaluationOpenRouterTests(unittest.TestCase):
    @patch.object(config, 'get_chat_model', return_value='anthropic/claude-3.5-sonnet')
    @patch.object(config, 'get_openrouter_headers', return_value={
        'HTTP-Referer': 'https://example.com',
        'X-Title': 'WAG Evaluation',
    })
    @patch.object(config, 'get_api_url', return_value='https://openrouter.ai/api/v1')
    @patch.object(config, 'get_api_key', return_value='or-key')
    @patch('evaluation.core.evaluator.OpenAI')
    def test_response_evaluator_initializes_openrouter_client(
        self,
        mock_openai,
        _mock_get_api_key,
        _mock_get_api_url,
        _mock_get_headers,
        _mock_get_chat_model,
    ) -> None:
        evaluator = ResponseEvaluator(chat_client='openrouter')

        mock_openai.assert_called_once_with(
            api_key='or-key',
            base_url='https://openrouter.ai/api/v1',
            default_headers={
                'HTTP-Referer': 'https://example.com',
                'X-Title': 'WAG Evaluation',
            },
        )
        self.assertEqual(evaluator.chat_client, 'openrouter')
        self.assertEqual(evaluator.chat_model, 'anthropic/claude-3.5-sonnet')

    def test_evaluation_pipeline_writes_evaluator_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EvaluationPipeline(
                results_path='unused.json',
                output_dir=tmpdir,
                chat_client='openrouter',
                chat_model='openai/gpt-4o-mini',
                run_evaluation=True,
            )
            pipeline.evaluator = DummyEvaluator(
                chat_client='openrouter',
                chat_model='openai/gpt-4o-mini',
            )

            items = [
                {'q_id': 'q1', 'query': 'How am I doing?', 'method': 'Base', 'response': 'okay'},
                {'q_id': 'q1', 'query': 'How am I doing?', 'method': 'Wag', 'response': 'better'},
            ]

            evaluated_items = pipeline._evaluate_group('q1', items)

        self.assertEqual(len(evaluated_items), 2)
        for item in evaluated_items:
            self.assertEqual(item['eval_chat_client'], 'openrouter')
            self.assertEqual(item['eval_chat_model'], 'openai/gpt-4o-mini')
            self.assertIn('eval_scores', item)
            self.assertIn('overall_quality', item)
            self.assertEqual(item['eval_status'], 'success')
            self.assertIsNone(item['eval_error'])

    def test_evaluation_pipeline_preserves_failed_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EvaluationPipeline(
                results_path='unused.json',
                output_dir=tmpdir,
                chat_client='openrouter',
                chat_model='openai/gpt-4o-mini',
                run_evaluation=True,
                max_workers=2,
            )
            pipeline.results = [
                {'q_id': 'q1', 'dataset': 'pmdata', 'method': 'Base', 'query': 'Q1', 'response': 'a'},
                {'q_id': 'q1', 'dataset': 'pmdata', 'method': 'Wag', 'query': 'Q1', 'response': 'b'},
                {'q_id': 'q2', 'dataset': 'pmdata', 'method': 'Base', 'query': 'Q2', 'response': 'c'},
                {'q_id': 'q2', 'dataset': 'pmdata', 'method': 'Wag', 'query': 'Q2', 'response': 'd'},
            ]
            with patch(
                'evaluation.pipelines.evaluation_pipeline.ResponseEvaluator',
                return_value=DummyEvaluator(
                    chat_client='openrouter',
                    chat_model='openai/gpt-4o-mini',
                ),
            ):
                original_evaluate_group = pipeline._evaluate_group

                def evaluate_group_side_effect(q_id, items):
                    if q_id == 'q1':
                        raise RuntimeError('judge parse failure')
                    return original_evaluate_group(q_id, items)

                with patch.object(pipeline, '_evaluate_group', side_effect=evaluate_group_side_effect):
                    pipeline.stats = {
                        'total_evaluations': 0,
                        'evaluations_run': 0,
                        'failed_evaluations': 0,
                        'methods_analyzed': 0,
                    }
                    pipeline._run_evaluations()

        self.assertEqual(len(pipeline.results), 4)
        q1_items = [item for item in pipeline.results if item['q_id'] == 'q1']
        q2_items = [item for item in pipeline.results if item['q_id'] == 'q2']
        self.assertEqual(len(q1_items), 2)
        self.assertEqual(len(q2_items), 2)
        self.assertTrue(all(item['eval_status'] == 'error' for item in q1_items))
        self.assertTrue(all(item['overall_quality'] is None for item in q1_items))
        self.assertTrue(all(item['eval_error'] == 'judge parse failure' for item in q1_items))
        self.assertTrue(all(item['eval_status'] == 'success' for item in q2_items))
        self.assertEqual(pipeline.stats['failed_evaluations'], 1)
        self.assertEqual(pipeline.stats['evaluations_run'], 2)

    def test_evaluation_pipeline_retries_transient_group_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EvaluationPipeline(
                results_path='unused.json',
                output_dir=tmpdir,
                chat_client='openrouter',
                chat_model='openai/gpt-4o-mini',
                run_evaluation=True,
            )
            pipeline.evaluator = DummyEvaluator(
                chat_client='openrouter',
                chat_model='openai/gpt-4o-mini',
            )
            pipeline.max_group_attempts = 2
            items = [
                {'q_id': 'q1', 'query': 'How am I doing?', 'method': 'Base', 'response': 'okay'},
                {'q_id': 'q1', 'query': 'How am I doing?', 'method': 'Wag', 'response': 'better'},
            ]

            original_evaluate_group = pipeline._evaluate_group
            calls = {'count': 0}

            def flaky_evaluate_group(q_id, group_items):
                calls['count'] += 1
                if calls['count'] == 1:
                    raise RuntimeError('temporary parse failure')
                return original_evaluate_group(q_id, group_items)

            with patch.object(pipeline, '_evaluate_group', side_effect=flaky_evaluate_group):
                evaluated_items = pipeline._evaluate_group_with_retries('q1', items)

        self.assertEqual(calls['count'], 2)
        self.assertEqual(len(evaluated_items), 2)
        self.assertTrue(all(item['eval_status'] == 'success' for item in evaluated_items))
        self.assertTrue(all(item['eval_attempts'] == 2 for item in evaluated_items))
        self.assertTrue(all(item['eval_retry_count'] == 1 for item in evaluated_items))

    def test_response_evaluator_rejects_unparseable_judge_output(self) -> None:
        evaluator = ResponseEvaluator.__new__(ResponseEvaluator)
        evaluator.prompt = 'prompt'
        evaluator.chat_client = 'deepseek'
        evaluator.llm_inference = lambda messages: object()
        evaluator._response_to_text = lambda response: 'not json'

        with patch('evaluation.core.evaluator.parse_llm_response', return_value=None):
            with self.assertRaisesRegex(ValueError, 'Could not parse judge response'):
                evaluator.evaluate('How am I doing?', {'Base': 'a', 'Wag': 'b'})

    def test_parse_llm_response_extracts_json_from_wrapped_text(self) -> None:
        parsed = parse_llm_response(
            'Here is the ranking:\n```JSON\n'
            '{"method_1": {"overall_quality": 1}, "method_2": {"overall_quality": 2}}\n'
            '```\nThanks.'
        )

        self.assertEqual(parsed['method_1']['overall_quality'], 1)
        self.assertEqual(parsed['method_2']['overall_quality'], 2)

    def test_parse_llm_response_ignores_trailing_text_after_json(self) -> None:
        parsed = parse_llm_response(
            '{"method_1": {"overall_quality": 1}}\n\nExplanation: method_1 is concise.'
        )

        self.assertEqual(parsed['method_1']['overall_quality'], 1)

    def test_calculate_wr_ignores_queries_with_missing_scores(self) -> None:
        win_rate_df = calculate_wr(
            pd.DataFrame([
                {'q_id': 'q1', 'method': 'Base', 'overall_quality': 1.0},
                {'q_id': 'q1', 'method': 'Wag', 'overall_quality': 2.0},
                {'q_id': 'q2', 'method': 'Base', 'overall_quality': None},
                {'q_id': 'q2', 'method': 'Wag', 'overall_quality': None},
            ])
        )

        self.assertEqual(win_rate_df.loc['Base', 'wins'], 1)
        self.assertEqual(win_rate_df.loc['Base', 'win_rate'], 1.0)
        self.assertEqual(win_rate_df.loc['Wag', 'win_rate'], 0.0)

    def test_context_evaluator_uses_local_method_mapping(self) -> None:
        evaluator = ContextEvaluator.__new__(ContextEvaluator)
        evaluator.prompt = 'prompt'
        evaluator.llm_inference = lambda messages: '{"ranking": ["method_2", "method_1"]}'
        evaluator._response_to_text = lambda response: response

        with patch('evaluation.core.evaluator.parse_llm_response', return_value={'ranking': ['method_2', 'method_1']}), \
             patch('evaluation.core.evaluator.random.shuffle', lambda methods: None):
            ranking, _, _ = evaluator.evaluate(
                'How am I doing?',
                {'Base': 'baseline context', 'Wag': 'wag context'},
            )

        self.assertEqual(ranking, ['Wag', 'Base'])
        self.assertNotIn('method_mapping', evaluator.__dict__)

    @patch.object(config, 'get_chat_model', return_value='gemma-4-31b-it')
    @patch.object(config, 'get_api_key', return_value='gemini-key')
    def test_response_evaluator_initializes_gemini_client_and_normalizes_messages(
        self,
        _mock_get_api_key,
        _mock_get_chat_model,
    ) -> None:
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    'candidates': [
                        {
                            'content': {
                                'parts': [
                                    {'text': '{"method_1": {"overall_quality": 1}, "method_2": {"overall_quality": 2}}'}
                                ]
                            }
                        }
                    ]
                }

        with patch('evaluation.core.evaluator.requests.post', return_value=FakeResponse()) as mock_post, \
             patch('evaluation.core.evaluator.parse_llm_response', return_value={
                 'method_1': {'overall_quality': 1},
                 'method_2': {'overall_quality': 2},
             }), \
             patch('evaluation.core.evaluator.random.shuffle', lambda methods: None):
            evaluator = ResponseEvaluator(chat_client='gemini')
            scores = evaluator.evaluate(
                'How am I doing?',
                {'Base': 'baseline answer', 'Wag': 'wag answer'},
            )

        self.assertEqual(evaluator.chat_model, 'gemma-4-31b-it')
        self.assertIn('gemma-4-31b-it:generateContent', mock_post.call_args.args[0])
        self.assertEqual(mock_post.call_args.kwargs['params'], {'key': 'gemini-key'})
        self.assertIn(
            '"query": "How am I doing?"',
            mock_post.call_args.kwargs['json']['contents'][0]['parts'][0]['text'],
        )
        self.assertEqual(scores['Base']['overall_quality'], 1)
        self.assertEqual(scores['Wag']['overall_quality'], 2)


if __name__ == '__main__':
    unittest.main()
