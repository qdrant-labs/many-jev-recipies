from jevqu.metrics import ndcg_at_k, paired_bootstrap, recall_at_k

def test_recall_and_ndcg():
    assert recall_at_k([1, 2, 3], {2, 9}, 3) == 0.5
    assert ndcg_at_k([1, 2], {1, 2}, 2) == 1.0
    assert round(ndcg_at_k([9, 1], {1}, 2), 4) == round(1 / 1.585, 4)  # 1/log2(3)
    assert ndcg_at_k([9], set(), 10) == 0.0

def test_paired_bootstrap_direction():
    a = [0.1] * 50; b = [0.3] * 50
    d, lo, hi = paired_bootstrap(a, b)
    assert abs(d - 0.2) < 1e-9 and lo > 0
