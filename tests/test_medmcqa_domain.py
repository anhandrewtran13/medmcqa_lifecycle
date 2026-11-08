"""Unit tests for medical-domain adapters (no GPU / pipeline bytecode required)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from flow_rag.medmcqa_domain import (  # noqa: E402
    extract_medical_bridge_entity,
    is_invalid_medmcqa_bridge_lock,
    is_meta_option_value,
    is_option_title_lock,
    is_plausible_medical_bridge_value,
    medmcqa_question_stem,
    mine_medical_hop1_bridge_spans,
)


def test_option_title_lock_detection():
    assert is_option_title_lock("Option A")
    assert is_option_title_lock("option c")
    assert is_option_title_lock("B")
    assert is_option_title_lock("D")
    assert is_option_title_lock("A. Sphenopalatine artery")
    assert is_option_title_lock("C. PCA")
    assert not is_option_title_lock("Zygomatic bone")
    assert not is_option_title_lock("Meckel's cartilage")
    assert not is_option_title_lock("Amoxicillin")


def test_meta_options():
    assert is_meta_option_value("All of the above")
    assert is_meta_option_value("None of the above")
    assert is_meta_option_value("All")
    assert not is_meta_option_value("Stapes")


def test_invalid_smoke_locks():
    for v in (
        "Explanation",
        "Option D",
        "A. Sphenopalatine aery",
        "D. Tetralogy of fallot",
        "0.02",
        "All",
        "Which of the following is not true for myelinated nerve fibers",
    ):
        assert is_invalid_medmcqa_bridge_lock(v), v
    assert not is_invalid_medmcqa_bridge_lock("Pre-Botzinger complex")
    assert not is_invalid_medmcqa_bridge_lock("Meckel's cartilage")


def test_plausible_rejects_option_locks():
    def base(value, entity=None, question=None):
        return True

    assert not is_plausible_medical_bridge_value(
        "Option C", "q", "q", base_fn=base
    )
    assert not is_plausible_medical_bridge_value("A. Stapes", base_fn=base)
    assert not is_plausible_medical_bridge_value("Explanation", base_fn=base)
    assert is_plausible_medical_bridge_value(
        "Meckel's cartilage", base_fn=base
    )


def test_stem_and_entity_from_medmcqa_question():
    q = (
        "All of the following are developed from Meckel's cartilage except:\n\n"
        "A. Zygomatic bone\n"
        "B. Malleus\n"
        "C. Incus\n"
        "D. Stapes\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    stem = medmcqa_question_stem(q)
    assert "Zygomatic" not in stem
    assert "Meckel" in stem
    ent = extract_medical_bridge_entity(q)
    assert ent is not None
    assert "Meckel" in ent or "cartilage" in ent.lower()
    assert not is_option_title_lock(ent)

    q2 = (
        "Which of the following is not true for myelinated nerve fibers?\n\n"
        "A. x\nB. y\n\nAnswer with a single letter (A, B, C, or D)."
    )
    ent2 = extract_medical_bridge_entity(q2)
    assert ent2 is not None
    assert "myelinated" in ent2.lower() or "nerve" in ent2.lower()
    assert not str(ent2).lower().startswith("which")

    q3 = (
        "Drug of choice for scrub typhus is:\n\n"
        "A. Amoxicillin\nB. Doxycycline\n\n"
        "Answer with a single letter (A, B, C, or D)."
    )
    ent3 = extract_medical_bridge_entity(q3)
    assert ent3 is not None
    assert "scrub" in ent3.lower() or "typhus" in ent3.lower()


def test_mine_drops_option_titles():
    Para = types.SimpleNamespace

    def base_fn(question, entity, paragraphs, max_n=36):
        return [
            ("Option C", "title(Option C)", "opt_C", "Incus", 50.0),
            ("Incus", "bridge_span(Incus)", "opt_C", "Incus", 10.0),
        ]

    spans = mine_medical_hop1_bridge_spans(
        "Regarding Meckel's cartilage, which is true?",
        "Meckel's cartilage",
        [
            Para(para_id="opt_C", title="Option C", text="Incus"),
            Para(para_id="exp::0", title="Explanation", text="Stapes is from Reichert."),
        ],
        max_n=20,
        base_fn=base_fn,
    )
    values = [s[0] for s in spans]
    assert "Option C" not in values
    assert any("Meckel" in v for v in values)
    assert "Incus" in values


if __name__ == "__main__":
    test_option_title_lock_detection()
    test_meta_options()
    test_invalid_smoke_locks()
    test_plausible_rejects_option_locks()
    test_stem_and_entity_from_medmcqa_question()
    test_mine_drops_option_titles()
    print("ok")
