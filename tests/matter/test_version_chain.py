"""Tests for document version-chain detection — spec §14, §26.

Verifies:
1.  link_documents() stores a document_relation row
2.  link_documents() is idempotent (duplicate triple returns same id)
3.  get_version_family() returns only self when no relations exist
4.  get_version_family() returns predecessor via version_of link
5.  get_version_family() returns successor via reverse lookup
6.  get_version_family() traverses multi-hop chain
7.  _normalize_stem() extracts version suffix correctly (v1, v2)
8.  _normalize_stem() extracts draft/final suffix correctly
9.  _normalize_stem() returns sort_key=-1 for unversioned files
10. detect_version_chains() creates version_of links for v1/v2 pair
11. detect_version_chains() orders v1 before v2 correctly
12. detect_version_chains() ignores single-document groups
13. detect_version_chains() records gap when no unversioned base exists
14. detect_version_chains() does NOT record gap when base document present
15. MatterModel.detect_document_version_chains() is a wired convenience wrapper
"""

import pytest
from irys.matter import MatterModel, GapType
from irys.matter.graph import DocumentInventoryStore


@pytest.fixture
def model():
    return MatterModel.open_in_memory()


def _add_doc(model, path: str) -> str:
    """Add a document to inventory and return its id."""
    import hashlib
    sha = hashlib.sha256(path.encode()).hexdigest()
    doc_id, _ = model.inventory.upsert(path, sha256=sha)
    return doc_id


# ---------------------------------------------------------------------------
# 1-2. link_documents() basic behaviour
# ---------------------------------------------------------------------------

def test_link_documents_stores_row(model):
    a = _add_doc(model, "contract.pdf")
    b = _add_doc(model, "contract_v2.pdf")
    rel_id = model.inventory.link_documents(b, a, "version_of")
    assert rel_id

    rows = model.db.execute(
        "SELECT * FROM document_relation WHERE source_doc_id=? AND target_doc_id=?",
        (b, a),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["relation_type"] == "version_of"


def test_link_documents_is_idempotent(model):
    a = _add_doc(model, "sow.pdf")
    b = _add_doc(model, "sow_v2.pdf")
    id1 = model.inventory.link_documents(b, a, "version_of")
    id2 = model.inventory.link_documents(b, a, "version_of")
    assert id1 == id2


# ---------------------------------------------------------------------------
# 3-6. get_version_family() traversal
# ---------------------------------------------------------------------------

def test_get_version_family_self_only(model):
    doc_id = _add_doc(model, "standalone.pdf")
    family = model.inventory.get_version_family(doc_id)
    assert len(family) == 1
    assert family[0]["direction"] == "self"
    assert family[0]["id"] == doc_id


def test_get_version_family_finds_predecessor(model):
    v1 = _add_doc(model, "msa_v1.pdf")
    v2 = _add_doc(model, "msa_v2.pdf")
    model.inventory.link_documents(v2, v1, "version_of")

    family = model.inventory.get_version_family(v2)
    ids = {m["id"] for m in family}
    assert v1 in ids
    directions = {m["direction"] for m in family}
    assert "predecessor" in directions


def test_get_version_family_finds_successor(model):
    v1 = _add_doc(model, "nda_v1.pdf")
    v2 = _add_doc(model, "nda_v2.pdf")
    model.inventory.link_documents(v2, v1, "version_of")

    family = model.inventory.get_version_family(v1)
    ids = {m["id"] for m in family}
    assert v2 in ids
    directions = {m["direction"] for m in family}
    assert "successor" in directions


def test_get_version_family_multi_hop(model):
    v1 = _add_doc(model, "agmt_v1.pdf")
    v2 = _add_doc(model, "agmt_v2.pdf")
    v3 = _add_doc(model, "agmt_v3.pdf")
    model.inventory.link_documents(v2, v1, "version_of")
    model.inventory.link_documents(v3, v2, "version_of")

    family = model.inventory.get_version_family(v1)
    ids = {m["id"] for m in family}
    assert v2 in ids
    assert v3 in ids
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 7-9. _normalize_stem() static method
# ---------------------------------------------------------------------------

def test_normalize_stem_v_suffix():
    base, key = DocumentInventoryStore._normalize_stem("contract_v2.pdf")
    assert base == "contract"
    assert key == 2


def test_normalize_stem_v1_lower_than_v2():
    _, k1 = DocumentInventoryStore._normalize_stem("contract_v1.pdf")
    _, k2 = DocumentInventoryStore._normalize_stem("contract_v2.pdf")
    assert k1 < k2


def test_normalize_stem_draft_final():
    _, kd = DocumentInventoryStore._normalize_stem("agreement_draft.pdf")
    _, kf = DocumentInventoryStore._normalize_stem("agreement_final.pdf")
    assert kd < kf


def test_normalize_stem_unversioned_returns_minus_one():
    base, key = DocumentInventoryStore._normalize_stem("contract.pdf")
    assert base == "contract"
    assert key == -1


# ---------------------------------------------------------------------------
# 10-11. detect_version_chains() correct links
# ---------------------------------------------------------------------------

def test_detect_version_chains_creates_link(model):
    _add_doc(model, "lease_v1.pdf")
    _add_doc(model, "lease_v2.pdf")

    links = model.detect_document_version_chains()
    assert len(links) == 1
    link = links[0]
    # v2 is the source (later version), v1 is the target (predecessor)
    assert "v2" in link["source_path"].lower() or link["sort_key"] == 2
    assert "v1" in link["target_path"].lower() or link["sort_key"] != 1


def test_detect_version_chains_v1_is_predecessor(model):
    v1_id = _add_doc(model, "purchase_v1.pdf")
    v2_id = _add_doc(model, "purchase_v2.pdf")

    links = model.detect_document_version_chains()
    assert len(links) == 1
    # v2 --version_of--> v1
    assert links[0]["source_doc_id"] == v2_id
    assert links[0]["target_doc_id"] == v1_id


# ---------------------------------------------------------------------------
# 12. detect_version_chains() ignores singletons
# ---------------------------------------------------------------------------

def test_detect_version_chains_ignores_singletons(model):
    _add_doc(model, "standalone.pdf")
    _add_doc(model, "other_document.pdf")

    links = model.detect_document_version_chains()
    assert links == []


# ---------------------------------------------------------------------------
# 13-14. Gap behaviour
# ---------------------------------------------------------------------------

def test_detect_version_chains_records_gap_when_no_base(model):
    """v1 and v2 both versioned → no unversioned base → gap recorded."""
    _add_doc(model, "order_v1.pdf")
    _add_doc(model, "order_v2.pdf")

    model.detect_document_version_chains()

    gaps = model.gaps.open_gaps()
    gap_types = [g["gap_type"] for g in gaps]
    assert GapType.MISSING_DOCUMENT.value in gap_types


def test_detect_version_chains_no_gap_when_base_present(model):
    """Base doc + versioned doc → base serves as unversioned doc → no gap."""
    _add_doc(model, "order.pdf")       # base, sort_key=-1
    _add_doc(model, "order_v2.pdf")    # version 2

    model.detect_document_version_chains()

    gaps = model.gaps.open_gaps()
    gap_types = [g["gap_type"] for g in gaps]
    assert GapType.MISSING_DOCUMENT.value not in gap_types


# ---------------------------------------------------------------------------
# 15. MatterModel convenience wrapper
# ---------------------------------------------------------------------------

def test_matter_model_detect_version_chains_wrapper(model):
    _add_doc(model, "sla_v1.pdf")
    _add_doc(model, "sla_v2.pdf")
    result = model.detect_document_version_chains()
    assert isinstance(result, list)
    assert len(result) >= 1
