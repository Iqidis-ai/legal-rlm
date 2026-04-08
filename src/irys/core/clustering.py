"""Document clustering - group related documents by filename patterns."""

from pathlib import Path


def cluster_by_document_type(file_paths: list[str]) -> dict[str, list[str]]:
    """
    Simple clustering by document type based on filename patterns.

    Args:
        file_paths: List of file paths

    Returns:
        Dict mapping category to list of files
    """
    categories = {
        "contracts": ["contract", "agreement", "amendment", "addendum"],
        "pleadings": ["complaint", "answer", "motion", "brief", "memorandum"],
        "discovery": ["interrogator", "deposition", "request", "response"],
        "orders": ["order", "judgment", "ruling", "decision"],
        "correspondence": ["letter", "email", "memo", "correspondence"],
        "exhibits": ["exhibit", "attachment", "appendix"],
        "reports": ["report", "analysis", "summary", "review"],
    }

    result: dict[str, list[str]] = {cat: [] for cat in categories}
    result["other"] = []

    for file_path in file_paths:
        filename_lower = Path(file_path).stem.lower()
        categorized = False

        for category, keywords in categories.items():
            if any(kw in filename_lower for kw in keywords):
                result[category].append(file_path)
                categorized = True
                break

        if not categorized:
            result["other"].append(file_path)

    # Remove empty categories
    return {k: v for k, v in result.items() if v}
