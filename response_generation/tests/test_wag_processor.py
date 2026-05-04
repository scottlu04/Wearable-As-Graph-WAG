import unittest
from unittest.mock import patch

import pandas as pd

from response_generation.core.wag_processor import WAGProcessor


class DummyWeightRetriever:
    captured_rel_info = None

    def __init__(self, node_name, rel_info, prior_matrix, hyperparameter, par_df, query_info):
        DummyWeightRetriever.captured_rel_info = rel_info

    def run(self):
        return pd.DataFrame(), pd.DataFrame()


class WAGProcessorTests(unittest.TestCase):
    def test_weight_recomputation_does_not_mutate_shared_rel_info(self) -> None:
        processor = object.__new__(WAGProcessor)
        rel_info = {
            'numeric_metrics': ['Heart rate'],
            'rel_pop_all': 'population-matrix',
            'rel_var_pop_all': 'population-var',
            'rel_pop_sample_size_all': 'population-n',
            'rel_ind_all': 'individual-matrix',
            'rel_var_ind_all': 'individual-var',
            'rel_ind_sample_size_all': 'individual-n',
        }

        with patch('shared.retrieval_package.WeightRetriever', DummyWeightRetriever):
            processor.weight_recomputation(
                node_name='Mental fatigue',
                query_info={'query_date': '2024-01-01', 'time_granularity': 7, 'openness': 0.5},
                rel_info=rel_info,
                prior_matrix=None,
                par_df=pd.DataFrame(),
                hyperparameter={},
            )

        self.assertEqual(rel_info['rel_pop_all'], 'population-matrix')
        self.assertEqual(rel_info['rel_ind_all'], 'individual-matrix')
        self.assertIsNot(DummyWeightRetriever.captured_rel_info, rel_info)
        self.assertIsNone(DummyWeightRetriever.captured_rel_info['rel_pop_all'])
        self.assertIsNone(DummyWeightRetriever.captured_rel_info['rel_ind_all'])


if __name__ == '__main__':
    unittest.main()
