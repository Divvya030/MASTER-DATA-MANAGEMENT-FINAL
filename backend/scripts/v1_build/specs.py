"""V1 model specifications -- the 'architecture' a v2 retrain should follow.
String = fuzzy (affine-gap + learned blocking); Exact = equality. All has_missing=True
(real master data is sparse; a missing value must not count as a mismatch)."""

SUPPLIER_V1 = {
    "domain": "supplier", "model_id": "TR-V1-SUPPLIER", "version": "v1.0.0",
    "fields": [("supplier_name", "String"), ("legal_name", "String"), ("email", "String"), ("phone", "String"),
               ("address", "String"), ("city", "String"), ("postal_code", "Exact"), ("country", "Exact"),
               ("duns", "Exact"), ("tax_identifier", "Exact"), ("registration_number", "Exact")],
    "identifier_fields": ["duns", "tax_identifier", "registration_number"],
    "name_fields": ["supplier_name", "legal_name"],
    "training_mapping": {"supplier_name": "supplier_name", "legal_name": "legal_name", "email": "email",
                         "phone": "phone", "address": "address_line_1", "city": "city", "postal_code": "postal_code",
                         "country": "country", "duns": "duns", "tax_identifier": "tax_identification_number",
                         "registration_number": "registration_number"},
}

CUSTOMER_V1 = {
    "domain": "customer", "model_id": "TR-V1-CUSTOMER", "version": "v1.0.0",
    # `state` is intentionally NOT compared: noisy (GJ vs GUJARAT, 855 'please select') and
    # redundant with city + postal_code + country. DUNS is NOT a model field: it defines the
    # weak training labels, so the model must not be able to read its own label.
    "fields": [("customer_name", "String"), ("address", "String"), ("city", "String"),
               ("postal_code", "Exact"), ("country", "Exact")],
    "identifier_fields": [], "name_fields": ["customer_name"],
    "training_mapping": {"customer_name": "customer_name", "address": "address1_p", "city": "city_p",
                         "state": "state_p", "postal_code": "postal_code_p", "country": "country_p",
                         "duns": "cust_duns_nbr"},
}


def make_variables(spec):
    from dedupe.variables import String, Exact
    ctor = {"String": String, "Exact": Exact}
    return [ctor[t](field=f, has_missing=True) for f, t in spec["fields"]]
