import unittest

import numpy as np

from sparc.rvq_transition_search import search_rvq
from sparc.local_ppg_search import transition_nll


class RVQTransitionSearchTest(unittest.TestCase):
    def test_prior_only_affects_first_stage_and_other_stages_remain_editable(self):
        initial = np.zeros((3,2),dtype=np.int64)
        codebooks = np.array([[[0.],[1.]],[[0.],[1.]]])
        probabilities = np.array([[.999,.001],[.5,.5]])
        target = np.tile([.1,.9]+[0.]*33,(6,1))
        def evaluate(sequences):
            # A token 1 matches the target; all stages independently affect PPG.
            labels = sequences.reshape(len(sequences),-1)
            return np.pad(np.stack((.9-.8*labels,.1+.8*labels),axis=-1),((0,0),(0,0),(0,33)))
        plain = search_rvq(initial,target,evaluate,codebooks,probabilities,0,
                           passes=1,positions=3,neighbors=1,global_candidates=0,batch_size=2)
        soft = search_rvq(initial,target,evaluate,codebooks,probabilities,100,
                          passes=1,positions=3,neighbors=1,global_candidates=0,batch_size=2)
        np.testing.assert_array_equal(plain['tokens'],np.ones((3,2)))
        np.testing.assert_array_equal(soft['tokens'][:,0],initial[:,0])
        np.testing.assert_array_equal(soft['tokens'][:,1],np.ones(3))
        self.assertLess(soft['final_js'],soft['baseline_js'])
        self.assertTrue(all(b['current_objective']<=a['current_objective']+1e-10 for a,b in zip(soft['trace'],soft['trace'][1:])))
        expected_nll = transition_nll(initial[:,0],probabilities)
        for row in soft['trace']:
            self.assertAlmostEqual(row['current_objective'],row['current_js']+100*expected_nll)
        np.testing.assert_array_equal(initial,np.zeros((3,2)))

    def test_invalid_shape_and_weight(self):
        books = np.zeros((2,2,1));p = np.full((2,2),.5)
        for tokens,weight in [(np.zeros(3),0),(np.zeros((3,2)),float('nan'))]:
            with self.assertRaises(ValueError):
                search_rvq(tokens,None,None,books,p,weight)


if __name__ == '__main__': unittest.main()
