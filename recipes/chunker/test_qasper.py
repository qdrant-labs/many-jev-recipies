from recipes.chunker.qasper import paper_text, questions, scores

def answers(*evidence_lists):
    return {"answer": [{"evidence": ev} for ev in evidence_lists]}

PAPER = {"id": "p1", "title": "A Title", "abstract": "We study things.",
         "full_text": {"section_name": ["Intro", None], "paragraphs": [["First para. ", "Second para."], ["Third para."]]},
         "qas": {"question_id": ["q1", "q2", "q3", "q4"], "question": ["a?", "b?", "c?", "d?"],
                 "answers": [answers(["Second para."], [" Third para.", "FLOAT SELECTED: Table 1"]),
                             answers(["FLOAT SELECTED: Figure 2"]),  # table or figure only
                             answers([]),                              # unanswerable
                             answers(["Not in the paper."])]}}

def test_qasper():
    text = paper_text(PAPER)
    assert text == "A Title\n\nWe study things.\n\nIntro\n\nFirst para.\n\nSecond para.\n\nThird para."
    qs = questions(PAPER, text)
    assert [q["id"] for q in qs] == ["q1"]  # evidence pooled over annotators; FLOAT, empty and missing dropped
    assert [text[a:b] for a, b in qs[0]["refs"]] == ["Second para.", "Third para."]
    assert scores([(0, 10)], [(0, 10)]) == {"recall": 1.0, "precision": 1.0, "iou": 1.0}
    s = scores([(0, 10), (5, 15)], [(10, 20)])  # overlapping retrieved spans count once
    assert s == {"recall": 0.5, "precision": 5 / 15, "iou": 5 / 20}
    assert scores([], [(0, 4)]) == {"recall": 0.0, "precision": 0.0, "iou": 0.0}
