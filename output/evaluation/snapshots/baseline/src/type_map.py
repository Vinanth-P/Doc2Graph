"""spaCy ↔ internal entity type mapping."""

# spaCy en_core_web_sm uses OntoNotes 5 labels.
# We unify them into a smaller internal set.
SPACY_TO_INTERNAL: dict[str, str] = {
    "PERSON":  "PERSON",
    "ORG":     "ORG",
    "GPE":     "GPE",   # geopolitical entity
    "LOC":     "GPE",   # natural locations → GPE for simplicity
    "FAC":     "GPE",   # buildings / airports → GPE
    "PRODUCT": "PRODUCT",
    "EVENT":   "EVENT",
    "LAW":     "LAW",
    "DATE":    "DATE",  # stored as property, not a node
    "MONEY":   "MONEY", # stored as property, not a node
}

# Internal types that create graph nodes (DATE/MONEY are properties)
NODE_TYPES: frozenset[str] = frozenset(
    {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW"}
)

# Internal type → Neo4j secondary label
TYPE_TO_LABEL: dict[str, str] = {
    "PERSON":  "Person",
    "ORG":     "Org",
    "GPE":     "Gpe",
    "PRODUCT": "Product",
    "EVENT":   "Event",
    "LAW":     "Law",
}
