"""
Domain schema configuration.

This module defines canonical schemas for each supported domain (customer, supplier, ...)
as DATA, not as branching business logic. Adding a new domain means adding a new
CanonicalSchema entry here -- the mapping/profiling/standardization engines never
branch on domain name or specific column names.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CanonicalField:
    name: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    identifier_hint: bool = False
    description: str = ""
    # Record-level METADATA (which system a row came from, when it was last
    # changed). Used by survivorship (source_priority / recency) and shown in
    # lineage, but it says nothing about WHO the entity is, so profiling never
    # recommends it as an ER MATCH/IDENTIFIER field.
    metadata_hint: bool = False


@dataclass(frozen=True)
class CanonicalSchema:
    domain: str
    fields: tuple[CanonicalField, ...]

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def get(self, name: str) -> Optional[CanonicalField]:
        for f in self.fields:
            if f.name == name:
                return f
        return None


CUSTOMER_SCHEMA = CanonicalSchema(
    domain="customer",
    fields=(
        CanonicalField("customer_id", ("account_number", "cust_account_id", "customerid",
                                        "account id", "acct number", "party number"),
                        identifier_hint=True),
        CanonicalField("customer_name", ("customername", "cust name", "account name", "name")),
        CanonicalField("legal_name", ("legalname", "legal entity name", "registered name")),
        CanonicalField("first_name", ("firstname", "fname", "given name")),
        CanonicalField("last_name", ("lastname", "lname", "surname", "family name")),
        CanonicalField("email", ("emailaddress", "email address", "e mail")),
        CanonicalField("phone", ("phonenumber", "phone number", "telephone", "tel")),
        CanonicalField("dob", ("dateofbirth", "date of birth", "birth date")),
        CanonicalField("address", ("address line 1", "street address", "addr")),
        CanonicalField("city", ("townname",)),
        CanonicalField("state", ("province", "region")),
        CanonicalField("country", ("countryname", "nation")),
        CanonicalField("postal_code", ("zip", "zipcode", "postcode", "postalcode")),
        CanonicalField("website", ("url", "web site", "homepage")),
        CanonicalField("duns", ("dunsnumber", "duns number", "d u n s number"), identifier_hint=True),
        CanonicalField("tax_identifier", ("tax id", "tax identification number",
                                           "tax registration number", "vat number", "ein"),
                        identifier_hint=True),
        CanonicalField("registration_number", ("reg number", "company registration number"),
                        identifier_hint=True),
        CanonicalField("source_system", ("source", "source name", "source system name", "src system",
                                          "system of record", "origin system", "data source"),
                        metadata_hint=True,
                        description="System the record came from (drives survivorship source priority)."),
        CanonicalField("last_updated", ("updated at", "last updated", "last modified", "modified date",
                                         "last update date", "updated date", "modified at", "update timestamp"),
                        metadata_hint=True,
                        description="When the source record was last changed (drives survivorship recency)."),
    ),
)


SUPPLIER_SCHEMA = CanonicalSchema(
    domain="supplier",
    fields=(
        CanonicalField("supplier_id", ("suppliernumber", "supplier number", "vendorid",
                                        "vendor id", "supplierid"), identifier_hint=True),
        CanonicalField("supplier_name", ("suppliername", "vendor name", "name", "supplier")),
        CanonicalField("legal_name", ("legalname", "legal entity name", "registered name")),
        CanonicalField("trade_name", ("trading_name", "trading name", "alternatename", "alternate name", "alias", "dba")),
        CanonicalField("email", ("email address",)),
        CanonicalField("phone", ("phone number", "telephone")),
        CanonicalField("address", ("address line 1", "street address")),
        CanonicalField("city", ("town name",)),
        CanonicalField("state", ("province", "region")),
        CanonicalField("country", ("country name", "nation")),
        CanonicalField("postal_code", ("zip", "zip code", "post code")),
        CanonicalField("website", ("url", "corporate web site", "homepage")),
        CanonicalField("duns", ("duns number", "d u n s number"),
                        identifier_hint=True),
        CanonicalField("tax_identifier", ("taxpayer id", "tax id", "tax identification number", "vat number"),
                        identifier_hint=True),
        CanonicalField("registration_number", ("reg number",), identifier_hint=True),
        CanonicalField("source_system", ("source", "source name", "source system name", "src system",
                                          "system of record", "origin system", "data source"),
                        metadata_hint=True,
                        description="System the record came from (drives survivorship source priority)."),
        CanonicalField("last_updated", ("updated at", "last updated", "last modified", "modified date",
                                         "last update date", "updated date", "modified at", "update timestamp"),
                        metadata_hint=True,
                        description="When the source record was last changed (drives survivorship recency)."),
    ),
)


SCHEMAS: dict[str, CanonicalSchema] = {
    "customer": CUSTOMER_SCHEMA,
    "supplier": SUPPLIER_SCHEMA,
}


def get_schema(domain: str) -> CanonicalSchema:
    try:
        return SCHEMAS[domain]
    except KeyError:
        raise ValueError(f"Unknown domain: {domain}. Known domains: {list(SCHEMAS)}")


# ---------------------------------------------------------------------------
# Presentation helpers (used by the data-entry form; never by the ER engines)
# ---------------------------------------------------------------------------
_LABEL_OVERRIDES = {
    "dob": "Date of birth",
    "duns": "D-U-N-S number",
    "tax_identifier": "Tax identifier",
    "postal_code": "Postal code",
    "source_system": "Source system",
    "last_updated": "Last updated",
}
_INPUT_TYPES = {"email": "email", "phone": "tel", "dob": "date", "website": "url",
                "last_updated": "datetime-local"}


def field_label(name: str) -> str:
    return _LABEL_OVERRIDES.get(name) or name.replace("_", " ").capitalize()


def input_type_for(name: str) -> str:
    return _INPUT_TYPES.get(name, "text")


def metadata_field_names(domain: str) -> set:
    """Canonical fields that describe the RECORD (source, timestamp) rather than the entity."""
    return {f.name for f in get_schema(domain).fields if f.metadata_hint}


# ---------------------------------------------------------------------------
# Display ordering helpers (presentation only; never used by the ER engines)
# ---------------------------------------------------------------------------
# Fields a steward recognises a record by, in the order they should be shown on
# list cards: identifier, name, then contact / address details.
PREFERRED_SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "customer": ("customer_id", "customer_name", "email", "phone", "address", "city", "country"),
    "supplier": ("supplier_id", "supplier_name", "email", "phone", "address", "city", "country"),
}
_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "customer": ("customer_name", "legal_name", "full_name"),
    "supplier": ("supplier_name", "legal_name", "trade_name"),
}


def ordered_field_names(domain: str, names) -> list:
    """Schema order first (id, name, legal name, ... source_system, last_updated),
    then any non-schema fields alphabetically. Golden records used to be built in
    alphabetical order, which put `address` before `customer_id` and `customer_name`."""
    names = list(dict.fromkeys(names))
    schema = SCHEMAS.get(domain)
    if schema is None:
        return sorted(names)
    pos = {n: i for i, n in enumerate(schema.field_names())}
    return sorted(names, key=lambda n: (pos.get(n, len(pos)), n))


def order_values(domain: str, values):
    """Return `values` as a new dict whose keys follow ordered_field_names()."""
    if not isinstance(values, dict):
        return values
    return {k: values[k] for k in ordered_field_names(domain, values.keys())}


def _present(v) -> bool:
    return v is not None and str(v).strip() != "" and str(v).strip().lower() not in {"nan", "none", "null"}


def summary_items(domain: str, values, limit: int = 6) -> list:
    """The few fields to show on a list card: [{field, label, value}] -- id, name,
    then contact/address -- skipping empty ones."""
    if not isinstance(values, dict):
        return []
    out = []
    for f in PREFERRED_SUMMARY_FIELDS.get(domain, ()):
        if _present(values.get(f)):
            out.append({"field": f, "label": field_label(f), "value": str(values[f])})
        if len(out) >= limit:
            break
    if not out:   # unknown domain / nothing preferred present: first non-empty fields in order
        for f in ordered_field_names(domain, values.keys()):
            if _present(values.get(f)):
                out.append({"field": f, "label": field_label(f), "value": str(values[f])})
            if len(out) >= limit:
                break
    return out


def display_name(domain: str, values) -> str | None:
    """Best human label for a record: its name field, else its id field."""
    if not isinstance(values, dict):
        return None
    for f in _NAME_FIELDS.get(domain, ()):
        if _present(values.get(f)):
            return str(values[f])
    for f in PREFERRED_SUMMARY_FIELDS.get(domain, ())[:1]:
        if _present(values.get(f)):
            return str(values[f])
    return None
