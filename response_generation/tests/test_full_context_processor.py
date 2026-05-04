import unittest
from unittest.mock import Mock, patch

import pandas as pd

from response_generation.core.full_context_processor import (
    FullContextProcessor,
    FullHistoryContextProcessor,
    INFEASIBLE_RESPONSE_TEXT,
)
from response_generation.config.settings import MAX_CONTEXT_LENGTH
from shared.models.entity import Entity
from shared.models.relationship import Relationship


class FullContextProcessorTests(unittest.TestCase):
    def test_full_context_limit_is_160k_tokens(self) -> None:
        self.assertEqual(MAX_CONTEXT_LENGTH, 160000)

    def _build_processor(self) -> FullContextProcessor:
        processor = FullContextProcessor.__new__(FullContextProcessor)
        processor.prompt = 'You are a helpful assistant.'
        processor.chat_model = 'gpt-4o'
        processor.nodes_with_embeddings = {
            'heart_rate': Entity(
                id='heart_rate',
                name='Heart rate',
                description='Heart rate description',
                range='0-200',
                recommendation='Rest if elevated.',
                dataSource={'pmdata': {'description': 'Heart rate', 'range': '0-200', 'unit': 'bpm'}},
            ),
            'sleep_duration': Entity(
                id='sleep_duration',
                name='Sleep duration',
                description='Sleep duration description',
                range='0-12',
                recommendation='Sleep more.',
                dataSource={'pmdata': {'description': 'Sleep', 'range': '0-12', 'unit': 'hours'}},
            ),
            'dataset_mismatch': Entity(
                id='dataset_mismatch',
                name='Dataset mismatch',
                description='Should not be included',
                range='0-1',
                recommendation='N/A',
                dataSource={'globem': {'description': 'Other', 'range': '0-1', 'unit': 'x'}},
            ),
        }
        processor.edges = {
            'heart_sleep': Relationship(
                id='heart_sleep',
                entity_1_name='Heart rate',
                entity_2_name='Sleep duration',
                weight=0.8,
                description='Sleep duration is related to Heart rate.',
            ),
        }
        return processor

    def _build_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'Heart rate': [70, 71, 72, 73, 74],
            'Sleep duration': [8.0, 7.5, 7.0, 6.5, 6.0],
            'Dataset mismatch': [1, 1, 1, 1, 1],
        })

    def test_full_context_collects_all_available_metrics_with_query_window(self) -> None:
        processor = self._build_processor()

        output, _ = processor.query_quick(
            user_query='How am I doing?',
            par_df=self._build_dataframe(),
            gt_nodes=['Heart rate'],
            query_info={'query_date': '2024-01-05', 'time_granularity': 2, 'openness': 1.0},
            dataset_name='pmdata',
            response_gen=False,
        )

        self.assertEqual(output['status'], 'success')
        self.assertEqual(output['primary_nodes'], ['Heart rate'])
        self.assertEqual(output['related_nodes'], ['Sleep duration'])
        self.assertEqual(output['full_context_metric_count'], 2)
        self.assertEqual(output['full_context_history_span_days'], 4)
        self.assertEqual(output['full_context_time_scope'], 'query_period')
        self.assertEqual(output['full_context_used_time_range'], 2)
        self.assertEqual(set(output['data'].keys()), {'Heart rate', 'Sleep duration'})
        self.assertEqual(output['data']['Heart rate']['x'], [pd.to_datetime('2024-01-04').date(), pd.to_datetime('2024-01-05').date()])
        self.assertEqual(output['data']['Sleep duration']['x'], [pd.to_datetime('2024-01-04').date(), pd.to_datetime('2024-01-05').date()])
        self.assertIn('description: Heart rate description', output['context'])
        self.assertIn('recommendation: Sleep more.', output['context'])
        self.assertIn('Sleep duration is related to Heart rate:', output['context'])
        self.assertIn('Sleep duration is related to Heart rate.', output['context'])
        self.assertIn('Nodes related to matched nodes which might be helpful', output['context'])
        self.assertNotIn('2024-01-01', output['context'])

    def test_full_context_marks_infeasible_without_calling_llm(self) -> None:
        processor = self._build_processor()
        processor._generate_response = Mock(side_effect=AssertionError('LLM should not be called'))
        processor._generate_response_with_usage = Mock(side_effect=AssertionError('LLM should not be called'))

        with patch('response_generation.core.full_context_processor.MAX_CONTEXT_LENGTH', 1):
            output, _ = processor.query_quick(
                user_query='Summarize everything.',
                par_df=self._build_dataframe(),
                gt_nodes=['Heart rate'],
                query_info={'query_date': '2024-01-05', 'time_granularity': 2, 'openness': 1.0},
                dataset_name='pmdata',
                response_gen=True,
            )

        self.assertEqual(output['status'], 'infeasible')
        self.assertEqual(output['infeasible_reason'], 'context_limit_exceeded')
        self.assertFalse(output['generation_called'])
        self.assertEqual(output['response'], INFEASIBLE_RESPONSE_TEXT)
        self.assertEqual(output['output_tokens'], 0)
        self.assertGreaterEqual(output['input_tokens'], 0)
        self.assertGreaterEqual(output['retrieval_time_seconds'], 0)
        self.assertGreaterEqual(output['e2e_time_seconds'], 0)
        processor._generate_response.assert_not_called()
        processor._generate_response_with_usage.assert_not_called()

    def test_full_history_preserves_previous_all_history_behavior(self) -> None:
        processor = FullHistoryContextProcessor.__new__(FullHistoryContextProcessor)
        processor.prompt = 'You are a helpful assistant.'
        processor.chat_model = 'gpt-4o'
        processor.nodes_with_embeddings = self._build_processor().nodes_with_embeddings
        processor.edges = self._build_processor().edges

        output, _ = processor.query_quick(
            user_query='How am I doing overall?',
            par_df=self._build_dataframe(),
            gt_nodes=['Heart rate'],
            query_info={'query_date': '2024-01-05', 'time_granularity': 2, 'openness': 1.0},
            dataset_name='pmdata',
            response_gen=False,
        )

        self.assertEqual(output['full_context_time_scope'], 'all_history')
        self.assertEqual(output['full_context_used_time_range'], 'all')
        self.assertEqual(
            output['data']['Heart rate']['x'],
            [
                pd.to_datetime('2024-01-01').date(),
                pd.to_datetime('2024-01-02').date(),
                pd.to_datetime('2024-01-03').date(),
                pd.to_datetime('2024-01-04').date(),
                pd.to_datetime('2024-01-05').date(),
            ]
        )
        self.assertIn('Nodes related to matched nodes which might be helpful', output['context'])
        self.assertIn('2024-01-01', output['context'])


if __name__ == '__main__':
    unittest.main()
