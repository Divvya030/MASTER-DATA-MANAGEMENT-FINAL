"""Realistic, OFFLINE augmentation for company-name records (label provenance: SYNTHETIC).

Standardization already fixes case, whitespace, missing tokens, phone/email formats, so this
generator does NOT re-create those. It models what is left in real master data: legal-form
variants (cleanco, the public legal-entity-suffix list, strips the suffix), abbreviations, token
drops, typos, source-system tags ("Sid 110833", "[4020]", "0577" -- patterns observed in
customer.csv), street-type abbreviations and moved/partial addresses.
"""
from __future__ import annotations
import random, re

LEGAL_GROUPS = [
    ["Inc", "Inc.", "Incorporated"], ["Ltd", "Ltd.", "Limited"], ["Llc", "L.l.c.", "LLC"],
    ["Corp", "Corp.", "Corporation"], ["Co", "Co.", "Company"],
    ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "Pvt Limited"], ["Pte Ltd", "Pte. Ltd.", "Private Limited"],
    ["Pty Ltd", "Pty Limited"], ["Sdn Bhd", "Sdn. Bhd."], ["Gmbh", "GmbH"], ["Sa", "S.a.", "S.A."],
    ["Plc", "PLC"], ["Llp", "L.l.p."], ["Bv", "B.v."], ["Ag", "AG"],
]
ABBR = [("Group", "Grp"), ("Company", "Co"), ("Corporation", "Corp"), ("International", "Intl"),
        ("Technologies", "Tech"), ("Technology", "Tech"), ("Management", "Mgmt"), ("Manufacturing", "Mfg"),
        ("Associates", "Assoc"), ("Services", "Svcs"), ("Systems", "Sys"), ("Industries", "Ind"),
        ("&", "And"), ("Center", "Centre"), ("Centre", "Center")]
STREET = [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Boulevard", "Blvd"), ("Drive", "Dr"),
          ("Suite", "Ste"), ("Floor", "Fl"), ("North", "N"), ("South", "S"), ("East", "E"), ("West", "W")]


def _strip_legal(name):
    try:
        from cleanco import basename
        b = basename(name)
        return b.strip() if b and b.strip() else name
    except Exception:  # noqa: BLE001
        return name


def _swap(a, b, text, rng):
    pat = re.compile(rf"\b{re.escape(a)}\b", re.I) if a.isalnum() else re.compile(re.escape(a))
    if pat.search(text) and rng.random() < .7:
        return pat.sub(b, text, count=1), True
    return text, False


def _typo(tok, rng):
    if len(tok) < 5:
        return tok
    i = rng.randrange(1, len(tok) - 1)
    k = rng.random()
    if k < .4: return tok[:i] + tok[i + 1:]
    if k < .7: return tok[:i] + tok[i] + tok[i:]
    return tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]


def corrupt_name(name, rng):
    n = name
    ops = rng.sample(["legal", "abbr", "drop", "typo", "tag", "strip"], rng.choice([1, 1, 2, 2, 3]))
    for op in ops:
        toks = n.split()
        if op == "legal":
            core = _strip_legal(n)
            grp = rng.choice(LEGAL_GROUPS)
            n = f"{core} {rng.choice(grp)}".strip()
        elif op == "strip":
            n = _strip_legal(n)
        elif op == "abbr":
            a, b = rng.choice(ABBR)
            n, ok = _swap(a, b, n, rng)
            if not ok:
                n, _ = _swap(b, a, n, rng)
        elif op == "drop" and len(toks) >= 3:
            del toks[rng.randrange(1, len(toks))]; n = " ".join(toks)
        elif op == "typo" and toks:
            j = rng.randrange(len(toks)); toks[j] = _typo(toks[j], rng); n = " ".join(toks)
        elif op == "tag":
            n = n + rng.choice([f" Sid {rng.randint(10000, 999999)}", f" {rng.randint(1, 9999):04d}", f" [{rng.randint(1000, 9999)}]"])
    n = re.sub(r"\s+", " ", n).strip()
    return n if len(n) >= 3 else name


def corrupt_address(addr, rng):
    if not addr:
        return addr
    a = addr
    if rng.random() < .5:
        x, y = rng.choice(STREET)
        a, ok = _swap(x, y, a, rng)
        if not ok: a, _ = _swap(y, x, a, rng)
    if rng.random() < .25:
        a = re.sub(r"\b(Ste|Suite|Fl|Floor)\b.*$", "", a, flags=re.I).strip()
    return a or addr


def realistic_corrupt(rec, rng, same_city_addresses=None):
    """Return a plausible duplicate of `rec`. Name stays recognisably the same entity."""
    out = dict(rec)
    if rec.get("customer_name"):
        out["customer_name"] = corrupt_name(rec["customer_name"], rng)
    r = rng.random()
    if r < .20:
        out["address"] = None
    elif r < .35 and same_city_addresses:
        out["address"] = rng.choice(same_city_addresses)       # same company, different site/address record
    else:
        out["address"] = corrupt_address(rec.get("address"), rng)
    if rng.random() < .10: out["city"] = None
    if rng.random() < .10: out["postal_code"] = None
    if rng.random() < .05: out["country"] = None
    return out
