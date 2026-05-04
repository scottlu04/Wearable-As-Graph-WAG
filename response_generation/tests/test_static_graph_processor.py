import unittest

import pandas as pd

from response_generation.core.static_graph_processor import StaticGraphProcessor
from shared.models.entity import Entity
from shared.models.relationship import Relationship


class StaticGraphProcessorTests(unittest.TestCase):
    def _build_processor(self) -> StaticGraphProcessor:
        processor = StaticGraphProcessor.__new__(StaticGraphProcessor)
        processor.param = {
            'max_num_related_nodes': 5,
            'demog_metrics': [],
        }
        processor.prior_matrix = None
        processor.nodes_with_embeddings = {
            'heart_rate': Entity(
                id='heart_rate',
                name='Heart rate',
                dataSource={'pmdata': {'description': '', 'range': '', 'unit': ''}},
            ),
            'sleep_duration': Entity(
                id='sleep_duration',
                name='Sleep duration',
                dataSource={'pmdata': {'description': '', 'range': '', 'unit': ''}},
            ),
            'stress': Entity(
                id='stress',
                name='Stress',
                dataSource={'pmdata': {'description': '', 'range': '', 'unit': ''}},
            ),
            'steps': Entity(
                id='steps',
                name='Steps taken',
                dataSource={'pmdata': {'description': '', 'range': '', 'unit': ''}},
            ),
        }
        processor.edges = {
            'edge_sleep': Relationship(
                id='edge_sleep',
                entity_1_name='Heart rate',
                entity_2_name='Sleep duration',
                description='Sleep duration is related to heart rate.',
                weight=0.8,
            ),
            'edge_stress': Relationship(
                id='edge_stress',
                entity_1_name='Heart rate',
                entity_2_name='Stress',
                description='Stress is related to heart rate.',
                weight=0.9,
            ),
            'edge_steps': Relationship(
                id='edge_steps',
                entity_1_name='Heart rate',
                entity_2_name='Steps taken',
                description='Steps are related to heart rate.',
                weight=0.7,
            ),
        }
        return processor

    def test_static_graph_ranks_by_edge_weight(self) -> None:
        processor = self._build_processor()
        par_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'Heart rate': [70, 72, 74],
            'Sleep duration': [7.0, 7.5, 8.0],
            'Stress': [2, 4, 3],
            'Steps taken': [8000, 9000, 10000],
        })

        result_dict, out_df = processor.search_knowledge_graph(
            entities=['Heart rate'],
            query_info={'query_date': '2024-01-03', 'time_granularity': 7, 'openness': 1.0},
            par_df=par_df,
            dataset_name='pmdata',
            rel_info={'data_associated_metrics': ['Heart rate', 'Sleep duration', 'Stress', 'Steps taken']},
        )

        related_nodes = [
            related_info[0].name
            for related_info in result_dict['Heart rate']['related_nodes']
        ]
        self.assertEqual(related_nodes, ['Stress', 'Sleep duration', 'Steps taken'])
        self.assertEqual(out_df.attrs['requested_related_node_count'], 5)
        self.assertEqual(out_df.attrs['actual_related_node_count'], 3)
        self.assertEqual(out_df.attrs['candidate_pool_size'], 3)

    def test_static_graph_excludes_other_primary_nodes_and_missing_history(self) -> None:
        processor = self._build_processor()
        processor.edges['edge_primary'] = Relationship(
            id='edge_primary',
            entity_1_name='Heart rate',
            entity_2_name='Steps taken',
            description='Steps are related to heart rate.',
            weight=0.95,
        )
        par_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'Heart rate': [70, 72],
            'Sleep duration': [7.0, 7.5],
            'Stress': [None, None],
            'Steps taken': [8000, 9000],
        })

        result_dict, out_df = processor.search_knowledge_graph(
            entities=['Heart rate', 'Steps taken'],
            query_info={'query_date': '2024-01-02', 'time_granularity': 7, 'openness': 1.0},
            par_df=par_df,
            dataset_name='pmdata',
            rel_info={'data_associated_metrics': ['Heart rate', 'Sleep duration', 'Stress', 'Steps taken']},
        )

        related_nodes = [
            related_info[0].name
            for related_info in result_dict['Heart rate']['related_nodes']
        ]
        self.assertEqual(related_nodes, ['Sleep duration'])
        self.assertNotIn('Steps taken', related_nodes)
        self.assertEqual(out_df.attrs['candidate_pool_size'], 1)
        self.assertEqual(out_df.attrs['actual_related_node_count'], 1)


if __name__ == '__main__':
    unittest.main()
