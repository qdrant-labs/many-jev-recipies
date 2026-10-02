from recipes.wands import ndcg

def test_ndcg_graded():
    g = {1: 2, 2: 1, 3: 0}
    assert ndcg([1, 2, 3], g) == 1.0
    assert ndcg([], g) == 0.0 and ndcg([1], {}) == 0.0
    assert ndcg([2, 1], g) < ndcg([1, 2], g) < 1.0 + 1e-9
    assert ndcg([9, 1, 2], g) < ndcg([1, 2, 9], g)  # unjudged item counts as gain 0
