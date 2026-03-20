"""Tests for EN→CZ sentence-level translation pipeline.

Run on pod (needs MarianMT models + GPU):
    PYTHONPATH=src python tests/test_translate_en_to_cz.py

Run locally (no GPU, slow but works):
    PYTHONPATH=src python tests/test_translate_en_to_cz.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_split_sentences():
    """split_sentences correctly splits text and rebases entity offsets."""
    from data import split_sentences

    text = "Patient has diabetes. He also has hypertension. No other issues."
    entities = [(12, 20, "DISEASE"), (34, 46, "DISEASE")]
    sents = split_sentences(text, entities)

    assert len(sents) >= 2, f"Expected >=2 sentences, got {len(sents)}"

    # First sentence should have "diabetes" with rebased offset
    s0 = sents[0]
    assert len(s0["entities"]) == 1
    e = s0["entities"][0]
    assert s0["text"][e[0]:e[1]] == "diabetes", f"Got {s0['text'][e[0]:e[1]]!r}"

    # Second sentence should have "hypertension"
    s1 = sents[1]
    assert len(s1["entities"]) == 1
    e = s1["entities"][0]
    assert s1["text"][e[0]:e[1]] == "hypertension", f"Got {s1['text'][e[0]:e[1]]!r}"

    print("PASS: test_split_sentences")


def test_marker_roundtrip():
    """insert_markers + extract_markers preserves entity text and offsets."""
    from translate import insert_markers, extract_markers

    text = "Patient has diabetes and hypertension."
    entities = [(12, 20, "DISEASE"), (25, 37, "DISEASE")]

    marked = insert_markers(text, entities)
    assert "<E0>" in marked and "<E1>" in marked, f"Markers missing: {marked}"

    clean, marker_ents = extract_markers(marked)
    assert clean == text, f"Clean text mismatch: {clean!r} vs {text!r}"
    assert len(marker_ents) == 2, f"Expected 2 entities, got {len(marker_ents)}"

    for s, e, mid in marker_ents:
        assert clean[s:e] in ("diabetes", "hypertension"), f"Bad entity: {clean[s:e]!r}"

    print("PASS: test_marker_roundtrip")


def test_ncbi_brat_format():
    """Downloaded NCBI Disease BRAT files have correct format."""
    brat_dir = Path(__file__).resolve().parent.parent / "data" / "ncbi-disease-brat"
    if not brat_dir.exists():
        print("SKIP: test_ncbi_brat_format (no data/ncbi-disease-brat)")
        return

    txt_files = list(brat_dir.glob("*.txt"))
    ann_files = list(brat_dir.glob("*.ann"))
    assert len(txt_files) > 0, "No .txt files"
    assert len(ann_files) > 0, "No .ann files"
    assert len(txt_files) == len(ann_files), f"Mismatch: {len(txt_files)} txt vs {len(ann_files)} ann"

    # Verify a sample file
    sample_txt = txt_files[0]
    sample_ann = sample_txt.with_suffix(".ann")
    text = sample_txt.read_text(encoding="utf-8")
    ann = sample_ann.read_text(encoding="utf-8")

    for line in ann.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        assert len(parts) == 3, f"Bad ann line: {line!r}"
        info = parts[1].split()
        etype, start, end = info[0], int(info[1]), int(info[2])
        span_text = parts[2]
        actual = text[start:end]
        assert actual == span_text, f"Span mismatch: {actual!r} vs {span_text!r} at {start}:{end}"

    print(f"PASS: test_ncbi_brat_format ({len(txt_files)} docs)")


def test_translate_en_to_cz():
    """EN→CZ translation with markers preserves at least some entities."""
    import torch
    from translate import translate_text, insert_markers, extract_markers, EN_CS_MODEL
    from transformers import MarianMTModel, MarianTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EN_CS_MODEL} on {device}...")
    tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
    model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)

    # Test sentences with medical entities
    test_cases = [
        ("Patient has <E0>diabetes</E0> and <E1>hypertension</E1>.", 2),
        ("The <E0>tumor</E0> was removed surgically.", 1),
        ("She was diagnosed with <E0>breast cancer</E0> last year.", 1),
        ("Most <E0>colon cancers</E0> arise from mutations in <E1>adenomatous polyposis coli</E1>.", 2),
    ]

    total_markers = 0
    survived_markers = 0

    for marked_en, expected_count in test_cases:
        cz_marked = translate_text(marked_en, tok, model, device)
        clean_cz, ents = extract_markers(cz_marked)

        total_markers += expected_count
        survived_markers += len(ents)

        status = "OK" if len(ents) > 0 else "LOST"
        print(f"  [{status}] {expected_count} markers → {len(ents)} survived")
        print(f"    EN: {marked_en}")
        print(f"    CZ: {cz_marked}")
        if ents:
            for s, e, mid in ents:
                print(f"    Entity: {clean_cz[s:e]!r} [{s}:{e}]")

    survival_rate = survived_markers / total_markers if total_markers else 0
    print(f"\nMarker survival: {survived_markers}/{total_markers} = {survival_rate:.0%}")
    assert survival_rate > 0.3, f"Survival rate too low: {survival_rate:.0%}"
    print("PASS: test_translate_en_to_cz")


def test_full_pipeline_en_to_cz():
    """End-to-end: load NCBI doc → split → translate → BRAT output."""
    import torch
    from pathlib import Path
    from translate_sentences import translate_doc_en_to_cz
    from translate_external import load_external_documents
    from translate import EN_CS_MODEL
    from transformers import MarianMTModel, MarianTokenizer

    brat_dir = Path(__file__).resolve().parent.parent / "data" / "ncbi-disease-brat"
    if not brat_dir.exists():
        print("SKIP: test_full_pipeline_en_to_cz (no data)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading EN→CZ model on {device}...")
    en_cs_tok = MarianTokenizer.from_pretrained(EN_CS_MODEL)
    en_cs_model = MarianMTModel.from_pretrained(EN_CS_MODEL).to(device)
    en_cs = (en_cs_tok, en_cs_model)

    # Load first 3 docs
    docs = load_external_documents(brat_dir, ["DISEASE"])[:3]
    print(f"Testing on {len(docs)} docs...")

    success = 0
    for doc in docs:
        entity_type_map = {"DISEASE": "DISEASE"}
        result = translate_doc_en_to_cz(doc, entity_type_map, en_cs, device)
        if result is None:
            print(f"  {doc['doc_id']}: FAILED (no entities survived)")
            continue

        cz_text, cz_entities = result
        # Verify entity spans are valid
        for s, e, etype in cz_entities:
            assert 0 <= s < e <= len(cz_text), f"Bad span [{s}:{e}] in {len(cz_text)} chars"
            span = cz_text[s:e]
            assert len(span) > 0, f"Empty entity span"

        print(f"  {doc['doc_id']}: OK — {len(doc['entities'])} EN entities → {len(cz_entities)} CZ entities")
        for s, e, etype in cz_entities[:3]:
            print(f"    {etype}: {cz_text[s:e]!r}")
        success += 1

    print(f"\nPipeline: {success}/{len(docs)} docs translated successfully")
    assert success > 0, "No docs translated"
    print("PASS: test_full_pipeline_en_to_cz")


if __name__ == "__main__":
    print("=" * 60)
    print("Running translation pipeline tests")
    print("=" * 60)

    # Fast tests (no model loading)
    test_split_sentences()
    test_marker_roundtrip()
    test_ncbi_brat_format()

    # Slow tests (need MarianMT model)
    print("\n--- Translation tests (need MarianMT) ---")
    test_translate_en_to_cz()
    test_full_pipeline_en_to_cz()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
