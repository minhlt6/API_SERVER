import re
from typing import Set

_COLLECTION_SAFE_RE = re.compile(r"[^a-z0-9_]+")
_YEAR_PATTERN = re.compile(r"(20\d{2})")


def normalize_folder_key(folder_key: str) -> str:
    value = (folder_key or "").strip().lower()
    value = value.replace("-", "_")
    value = _COLLECTION_SAFE_RE.sub("_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "default"


def build_collection_name(folder_key: str, prefix: str = "rag") -> str:
    normalized = normalize_folder_key(folder_key)
    base = f"{prefix}_{normalized}"
    # Qdrant collection names should stay short and simple.
    return base[:63]


def extract_year_tokens(value: str) -> Set[str]:
    return {token for token in _YEAR_PATTERN.findall(value or "")}


def extract_folder_key_from_collection_name(collection_name: str, prefix: str = "rag") -> str | None:
    """
    Extract folder_key from collection name.
    E.g., 'rag_k63' -> 'k63', 'rag_2023_2024' -> '2023_2024'
    Returns None if collection_name doesn't match the expected pattern.
    """
    if not collection_name:
        return None
    
    prefix_with_underscore = f"{prefix}_"
    if collection_name.startswith(prefix_with_underscore):
        return collection_name[len(prefix_with_underscore):]
    
    return None


def collection_matches_cohort(collection_name: str, cohort_key: str, prefix: str = "rag") -> bool:
    """
    Check if collection matches the given cohort_key.
    E.g., collection='rag_k63', cohort_key='k63' -> True
    """
    if not cohort_key:
        return False
    
    extracted = extract_folder_key_from_collection_name(collection_name, prefix)
    if not extracted:
        return False
    
    return extracted.lower() == cohort_key.lower()


def collection_matches_year(collection_name: str, year_scope: str) -> bool:
    if not year_scope:
        return False

    collection_years = extract_year_tokens(collection_name)
    target_years = extract_year_tokens(year_scope)
    if not target_years:
        return False

    # For explicit ranges (e.g. 2022-2023), require all years to match.
    if len(target_years) >= 2:
        return target_years.issubset(collection_years)

    return bool(collection_years.intersection(target_years))
