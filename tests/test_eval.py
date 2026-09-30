from eval.judge import parse_verdict


def test_parses_clean_json():
    assert parse_verdict('{"faithful": true, "reason": "all supported"}') == (True, "all supported")


def test_parses_json_wrapped_in_text():
    ok, _ = parse_verdict('Sure! ```json\n{"faithful": false, "reason": "adds a date"}\n```')
    assert ok is False


def test_unparseable_returns_none():
    assert parse_verdict("I think it is fine")[0] is None


from langchain_core.documents import Document

from eval.run_eval import is_hit


def test_hit_requires_source_and_text():
    q = {"expected_source": "kyc.pdf", "expected_text": "once in every two years"}
    good = Document(page_content="at least once in\nevery  two years for high-risk", metadata={"source": "kyc.pdf"})
    wrong_doc = Document(page_content="once in every two years", metadata={"source": "cards.pdf"})
    wrong_chunk = Document(page_content="Definitions of terms", metadata={"source": "kyc.pdf"})
    assert is_hit(good, q)            # line breaks and double spaces are normalised
    assert not is_hit(wrong_doc, q)
    assert not is_hit(wrong_chunk, q)


def test_source_only_question_still_works():
    q = {"expected_source": "leave_policy.md"}
    assert is_hit(Document(page_content="anything", metadata={"source": "leave_policy.md"}), q)
