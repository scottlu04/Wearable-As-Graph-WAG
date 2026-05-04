import unittest
from unittest.mock import patch

import pandas as pd

from response_generation.pipelines.base_pipeline import BasePipeline
from response_generation.pipelines.general_comparison_pipeline import GeneralComparisonPipeline


class DummyProcessor:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.chat_provider = kwargs.get('chat_client')
        self.chat_model = kwargs.get('chat_model')


class DummyClient:
    def __init__(self, output, chat_provider=None, chat_model=None) -> None:
        self.output = output
        self.chat_provider = chat_provider
        self.chat_model = chat_model
        self.last_kwargs = None

    def query_quick(self, **kwargs):
        self.last_kwargs = kwargs
        return self.output, None


class ErrorClient:
    def query_quick(self, **kwargs):
        raise RuntimeError('boom')


class FlakyClient:
    def __init__(self, output) -> None:
        self.output = output
        self.calls = 0
        self.chat_provider = 'deepseek'
        self.chat_model = 'deepseek-chat'

    def query_quick(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError('temporary failure')
        return self.output, None


class RelInfoMutatingClient:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.initial_marker = None

    def query_quick(self, **kwargs):
        rel_info = kwargs['rel_info']
        self.initial_marker = rel_info.get('marker')
        rel_info['marker'] = self.marker
        return {
            'result_dict': {},
            'context': 'context',
            'data': {},
            'response': 'response',
            'status': 'success',
            'infeasible_reason': None,
            'context_tokens': 1,
            'estimated_prompt_tokens': 2,
            'input_tokens': 2,
            'output_tokens': 1,
            'retrieval_time_seconds': 0.01,
            'e2e_time_seconds': 0.02,
            'primary_nodes': [],
            'related_nodes': [],
        }, None


class MinimalPipeline(BasePipeline):
    def get_clients(self):
        return {}


class GeneralPipelineTests(unittest.TestCase):
    def test_general_pipeline_registers_all_methods_in_expected_order(self) -> None:
        with patch('response_generation.pipelines.general_comparison_pipeline.BaseQueryProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.RAGProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.StaticGraphProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.FullContextProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.WAGProcessor', DummyProcessor):
            pipeline = GeneralComparisonPipeline(
                root_dir='resources',
                chat_client='openrouter',
                chat_model='openai/gpt-4o-mini',
            )
            clients = pipeline.get_clients()

        self.assertEqual(
            list(clients.keys()),
            ['Base', 'Rag', 'StaticGraph', 'FullContext', 'Wag']
        )
        for client in clients.values():
            self.assertEqual(client.kwargs.get('chat_client'), 'openrouter')
            self.assertEqual(client.kwargs.get('chat_model'), 'openai/gpt-4o-mini')

    def test_general_pipeline_can_select_methods_without_constructing_others(self) -> None:
        with patch('response_generation.pipelines.general_comparison_pipeline.BaseQueryProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.RAGProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.StaticGraphProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.FullContextProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.WAGProcessor', DummyProcessor):
            pipeline = GeneralComparisonPipeline(
                root_dir='resources',
                methods=['Base', 'StaticGraph', 'Wag'],
            )
            clients = pipeline.get_clients()

        self.assertEqual(list(clients.keys()), ['Base', 'StaticGraph', 'Wag'])

    def test_general_pipeline_preserves_requested_method_order(self) -> None:
        with patch('response_generation.pipelines.general_comparison_pipeline.BaseQueryProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.RAGProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.StaticGraphProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.FullContextProcessor', DummyProcessor), \
             patch('response_generation.pipelines.general_comparison_pipeline.WAGProcessor', DummyProcessor):
            pipeline = GeneralComparisonPipeline(
                root_dir='resources',
                methods=['Wag', 'Base', 'Rag'],
            )
            clients = pipeline.get_clients()

        self.assertEqual(list(clients.keys()), ['Wag', 'Base', 'Rag'])

    def test_general_pipeline_rejects_unknown_methods(self) -> None:
        pipeline = GeneralComparisonPipeline(
            root_dir='resources',
            methods=['Base', 'NotAMethod'],
        )

        with self.assertRaisesRegex(ValueError, 'Unknown method'):
            pipeline.get_clients()

    def test_process_single_row_preserves_schema_for_full_context(self) -> None:
        pipeline = MinimalPipeline(root_dir='resources')
        row = pd.Series({
            'q_id': 'q-1',
            'query': 'Why is my heart rate high?',
            'associated_entity(gt)': ['Heart rate'],
            'query_date': '2024-01-03',
            'time_granularity': 7,
            'openness(assigned)': 1.0,
        })
        par_df = pd.DataFrame({'date': ['2024-01-01'], 'Heart rate': [70]})

        clients = {
            'FullContext': DummyClient({
                'result_dict': {},
                'context': 'full context',
                'data': {'Heart rate': {'x': ['2024-01-01'], 'y': [70]}},
                'response': 'full context infeasible',
                'status': 'infeasible',
                'infeasible_reason': 'context_limit_exceeded',
                'context_tokens': 100,
                'estimated_prompt_tokens': 5000,
                'input_tokens': 5000,
                'output_tokens': 0,
                'retrieval_time_seconds': 0.02,
                'e2e_time_seconds': 0.02,
                'primary_nodes': ['Heart rate'],
                'related_nodes': ['Sleep duration', 'Stress'],
                'full_context_metric_count': 3,
                'full_context_history_span_days': 30,
                'full_context_time_scope': 'query_period',
                'full_context_used_time_range': 7,
                'generation_called': False,
            }, chat_provider='deepseek', chat_model='deepseek-chat'),
        }

        _, results_rows = pipeline.process_single_row(
            row=row,
            clients=clients,
            par_df=par_df,
            dataset_name='pmdata',
            rel_info={},
            user_id='user-1',
        )

        by_method = {row['method']: row for row in results_rows}
        self.assertFalse(clients['FullContext'].last_kwargs['response_gen'])
        self.assertEqual(set(by_method.keys()), {'FullContext'})
        self.assertEqual(by_method['FullContext']['status'], 'infeasible')
        self.assertEqual(
            by_method['FullContext']['infeasible_reason'],
            'context_limit_exceeded'
        )
        self.assertEqual(by_method['FullContext']['full_context_time_scope'], 'query_period')
        self.assertEqual(by_method['FullContext']['full_context_used_time_range'], 7)
        self.assertEqual(by_method['FullContext']['input_tokens'], 5000)
        self.assertEqual(by_method['FullContext']['output_tokens'], 0)
        self.assertEqual(by_method['FullContext']['chat_client'], 'deepseek')
        self.assertEqual(by_method['FullContext']['chat_model'], 'deepseek-chat')
        self.assertFalse(by_method['FullContext']['generation_called'])
    def test_process_single_row_preserves_query_schema_on_errors(self) -> None:
        pipeline = MinimalPipeline(root_dir='resources')
        row = pd.Series({
            'q_id': 'q-2',
            'query': 'Summarize my history.',
            'associated_entity(gt)': ['Heart rate'],
            'query_date': '2024-01-05',
            'time_granularity': 14,
            'openness(assigned)': 0.8,
        })

        _, results_rows = pipeline.process_single_row(
            row=row,
            clients={'Wag': ErrorClient()},
            par_df=pd.DataFrame({'date': ['2024-01-01'], 'Heart rate': [70]}),
            dataset_name='pmdata',
            rel_info={},
            user_id='user-2',
        )

        self.assertEqual(len(results_rows), 1)
        error_row = results_rows[0]
        self.assertEqual(error_row['status'], 'error')
        self.assertEqual(error_row['query_date'], '2024-01-05')
        self.assertEqual(error_row['time_granularity'], 14)
        self.assertEqual(error_row['openness(assigned)'], 0.8)
        self.assertEqual(error_row['associated_entity(gt)'], ['Heart rate'])
        self.assertIsNone(error_row['context'])
        self.assertIsNone(error_row['response'])
        self.assertIsNone(error_row['data'])
        self.assertIsNone(error_row['input_tokens'])
        self.assertIsNone(error_row['output_tokens'])
        self.assertIsNone(error_row['retrieval_time_seconds'])
        self.assertIsNone(error_row['e2e_time_seconds'])
        self.assertEqual(error_row['generation_attempts'], 3)
        self.assertEqual(error_row['generation_retry_count'], 2)

    def test_process_single_row_retries_transient_method_failures(self) -> None:
        pipeline = MinimalPipeline(root_dir='resources')
        row = pd.Series({
            'q_id': 'q-retry',
            'query': 'Explain my activity.',
            'associated_entity(gt)': ['Steps taken'],
            'query_date': '2024-01-05',
            'time_granularity': 7,
            'openness(assigned)': 0.9,
        })
        client = FlakyClient({
            'result_dict': {},
            'context': 'context',
            'data': {},
            'response': 'answer',
            'status': 'success',
            'infeasible_reason': None,
            'context_tokens': 8,
            'estimated_prompt_tokens': 16,
            'input_tokens': 16,
            'output_tokens': 4,
            'retrieval_time_seconds': 0.02,
            'generation_time_seconds': 0.1,
            'e2e_time_seconds': 0.12,
            'primary_nodes': ['Steps taken'],
            'related_nodes': [],
        })

        _, results_rows = pipeline.process_single_row(
            row=row,
            clients={'Wag': client},
            par_df=pd.DataFrame({'date': ['2024-01-01'], 'Steps taken': [1000]}),
            dataset_name='pmdata',
            rel_info={},
            user_id='user-retry',
        )

        self.assertEqual(client.calls, 2)
        self.assertEqual(results_rows[0]['status'], 'success')
        self.assertEqual(results_rows[0]['generation_attempts'], 2)
        self.assertEqual(results_rows[0]['generation_retry_count'], 1)

    def test_process_single_row_can_enable_response_generation(self) -> None:
        pipeline = MinimalPipeline(root_dir='resources', generate_responses=True)
        row = pd.Series({
            'q_id': 'q-3',
            'query': 'Explain my sleep trend.',
            'associated_entity(gt)': ['Sleep duration'],
            'query_date': '2024-01-07',
            'time_granularity': 30,
            'openness(assigned)': 0.6,
        })
        client = DummyClient({
            'result_dict': {},
            'context': 'context here',
            'data': {'Sleep duration': {'x': ['2024-01-01'], 'y': [7.5]}},
            'response': 'generated answer',
            'status': 'success',
            'infeasible_reason': None,
            'context_tokens': 12,
            'estimated_prompt_tokens': 34,
            'input_tokens': 34,
            'output_tokens': 3,
            'retrieval_time_seconds': 0.04,
            'e2e_time_seconds': 0.5,
            'primary_nodes': ['Sleep duration'],
            'related_nodes': ['Activity'],
            'generation_called': True,
        })

        _, results_rows = pipeline.process_single_row(
            row=row,
            clients={'Wag': client},
            par_df=pd.DataFrame({'date': ['2024-01-01'], 'Sleep duration': [7.5]}),
            dataset_name='pmdata',
            rel_info={},
            user_id='user-3',
        )

        self.assertTrue(client.last_kwargs['response_gen'])
        self.assertEqual(results_rows[0]['response'], 'generated answer')
        self.assertTrue(results_rows[0]['generation_called'])

    def test_process_single_row_isolates_rel_info_between_methods(self) -> None:
        pipeline = MinimalPipeline(root_dir='resources')
        row = pd.Series({
            'q_id': 'q-4',
            'query': 'Explain my recovery.',
            'associated_entity(gt)': ['Recovery'],
            'query_date': '2024-01-07',
            'time_granularity': 7,
            'openness(assigned)': 0.5,
        })
        shared_rel_info = {'numeric_metrics': ['Heart rate']}
        client_a = RelInfoMutatingClient(marker='A')
        client_b = RelInfoMutatingClient(marker='B')

        _, results_rows = pipeline.process_single_row(
            row=row,
            clients={'Base': client_a, 'Wag': client_b},
            par_df=pd.DataFrame({'date': ['2024-01-01'], 'Recovery': [1]}),
            dataset_name='pmdata',
            rel_info=shared_rel_info,
            user_id='user-4',
        )

        self.assertEqual(len(results_rows), 2)
        self.assertIsNone(client_a.initial_marker)
        self.assertIsNone(client_b.initial_marker)
        self.assertNotIn('marker', shared_rel_info)

    def test_build_method_metrics_summary(self) -> None:
        pipeline = MinimalPipeline(root_dir='resources')
        results_df = pd.DataFrame([
            {
                'method': 'Base',
                'status': 'success',
                'input_tokens': 10,
                'output_tokens': 2,
                'retrieval_time_seconds': 0.01,
                'generation_time_seconds': 0.20,
                'e2e_time_seconds': 0.21,
            },
            {
                'method': 'Base',
                'status': 'success',
                'input_tokens': 30,
                'output_tokens': 4,
                'retrieval_time_seconds': 0.03,
                'generation_time_seconds': 0.30,
                'e2e_time_seconds': 0.33,
            },
            {
                'method': 'FullContext',
                'status': 'infeasible',
                'input_tokens': 5000,
                'output_tokens': 0,
                'retrieval_time_seconds': 0.05,
                'generation_time_seconds': 0.0,
                'e2e_time_seconds': 0.05,
            },
        ])

        summary = pipeline.build_method_metrics_summary(results_df)

        self.assertEqual(summary['Base']['count'], 2)
        self.assertEqual(summary['Base']['success_count'], 2)
        self.assertEqual(summary['Base']['input_tokens']['mean'], 20.0)
        self.assertEqual(summary['Base']['output_tokens']['max'], 4.0)
        self.assertEqual(summary['FullContext']['infeasible_count'], 1)
        self.assertEqual(summary['FullContext']['e2e_time_seconds']['mean'], 0.05)


if __name__ == '__main__':
    unittest.main()
