from jevqu.evalstark import recall_loss_curve, report_markdown

def test_curve_reports_loss_against_unfiltered():
    rel = [{1}, {2}]
    base = [[1, 3], [3, 2]]
    def run_at(t):
        return [[1, 3], [3]] if t < 0.9 else base   # low threshold filters query 2's answer away
    rows = recall_loss_curve({"off": base}, rel, [0.5, 0.9], run_at)
    assert rows[0]["threshold"] == 0.5 and rows[0]["recall_loss"] > 0
    assert rows[1]["recall_loss"] == 0
    md = report_markdown(rows, base_ndcg=0.5)
    assert "| threshold |" in md and "0.5" in md
