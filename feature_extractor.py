import json
import os
import re
import numpy as np
 
try:
    from fuzzywuzzy import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
 
# ---------------------------------------------------------------------------
# Load the canonical feature order (must match training order exactly)
# ---------------------------------------------------------------------------
_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "realtime_features.json")
with open(_FEATURES_PATH) as f:
    FEATURE_ORDER = json.load(f)
 
# ---------------------------------------------------------------------------
# Brand list used ONLY by brand_* and combo_* functions (25 brands)
# ---------------------------------------------------------------------------
KNOWN_BRANDS = [
    "paypal", "amazon", "apple", "google", "microsoft",
    "netflix", "facebook", "instagram", "twitter", "ebay",
    "linkedin", "dropbox", "spotify", "whatsapp", "gmail",
    "outlook", "yahoo", "chase", "wellsfargo", "bankofamerica",
    "github", "youtube", "wikipedia", "reddit", "stackoverflow"
]
 
 
def _split_domain_path(url: str):
    if "://" in url:
        after_scheme = url.split("://")[1]
        domain_part = after_scheme.split("/")[0]
        path_part = after_scheme.split("/", 1)[-1] if "/" in after_scheme else ""
    else:
        domain_part = url.split("/")[0]
        path_part = url.split("/", 1)[-1] if "/" in url else ""
    return domain_part, path_part
 
 
def extract_custom_features(url: str) -> dict:
    url_lower = url.lower()
    domain_part, path_part = _split_domain_path(url)
    total_len = max(len(url), 1)
    domain_len = max(len(domain_part), 1)
    vowels = sum(domain_part.lower().count(v) for v in "aeiou")
 
    # NOTE: this brand list is DIFFERENT (smaller) than KNOWN_BRANDS above —
    # that's intentional, matching the notebook exactly.
    suspicious_keywords = [
        "login", "signin", "verify", "update", "secure", "account",
        "banking", "confirm", "password", "credential", "suspend",
        "unusual", "access", "locked", "unlock", "validate", "submit",
        "paypal", "amazon", "apple", "google", "microsoft", "netflix",
        "instagram", "facebook", "twitter", "bank", "ebay", "support"
    ]
    brand_names = [
        "paypal", "amazon", "apple", "google", "microsoft",
        "netflix", "facebook", "instagram", "twitter", "ebay"
    ]
    suspicious_tlds = [
        ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
        ".club", ".online", ".site", ".info", ".biz", ".mom",
        ".pw", ".cc", ".ws", ".work", ".gd"
    ]
    urgency_words = [
        "urgent", "immediately", "expire", "suspend", "limited",
        "warning", "alert", "critical", "important", "verify-now"
    ]
    financial_words = [
        "payment", "invoice", "billing", "transaction", "wallet",
        "credit", "debit", "refund", "cashout", "withdraw", "deposit"
    ]
    free_hosting = [
        "wordpress", "wixsite", "weebly", "blogspot", "github.io",
        "netlify", "000webhostapp", "web.app", "firebaseapp",
        "vercel.app", "square.site", "wixstudio", "yolasite",
        "weeblysite", "pantheonsite", "workers.dev", "duckdns"
    ]
 
    try:
        return {
            "custom_num_hyphens":         url.count("-"),
            "custom_num_at":              url.count("@"),
            "custom_num_percent":         url.count("%"),
            "custom_num_underscores":     url.count("_"),
            "custom_num_slashes":         url.count("/"),
            "custom_uses_http":           int(url_lower.startswith("http://")),
            "custom_has_www":             1,  # dropped at training (constant) — filtered out later
            "custom_domain_has_hyphen":   int("-" in domain_part),
            "custom_domain_starts_num":   int(domain_part[0].isdigit() if domain_part else 0),
            "custom_domain_vowel_ratio":  vowels / domain_len,
            "custom_domain_digit_ratio":  sum(c.isdigit() for c in domain_part) / domain_len,
            "custom_has_multiple_tlds":   int(bool(re.search(r"\.(com|net|org|edu|gov)\.", domain_part))),
            "custom_path_depth":          len([p for p in path_part.split("/") if p]),
            "custom_path_has_exe":        int(any(ext in url_lower for ext in [".exe", ".zip", ".rar", ".bat", ".php", ".js"])),
            "custom_path_has_login":      int("login" in path_part.lower()),
            "custom_path_has_encoded":    int("%" in path_part),
            "custom_has_ip":              int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
            "custom_has_at_symbol":       int("@" in url),
            "custom_has_port":            int(bool(re.search(r":\d{4,5}", url))),
            "custom_has_hex":             int(bool(re.search(r"%[0-9a-f]{2}", url))),
            "custom_is_shortened":        int(bool(re.search(r"bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly", url_lower))),
            "custom_has_typosquat":       int(bool(re.search(r"paypa[^l]|go[^o]gle|gogle|amaz[^o]n|faceb[^o]ok", url_lower))),
            "custom_has_suspicious_kw":   int(any(w in url_lower for w in suspicious_keywords)),
            "custom_num_suspicious_kw":   sum(w in url_lower for w in suspicious_keywords),
            "custom_has_brand_name":      int(any(b in url_lower for b in brand_names)),
            "custom_has_suspicious_tld":  int(any(url_lower.endswith(t) for t in suspicious_tlds)),
            "custom_has_urgency_words":   int(any(w in url_lower for w in urgency_words)),
            "custom_has_financial_words": int(any(w in url_lower for w in financial_words)),
            "custom_is_free_hosting":     int(any(h in url_lower for h in free_hosting)),
            "custom_url_entropy":         -sum((url.count(c) / total_len) * np.log2(url.count(c) / total_len) for c in set(url) if url.count(c) > 0),
            "custom_ratio_digits":        sum(c.isdigit() for c in url) / total_len,
            "custom_ratio_special":       sum(not c.isalnum() for c in url) / total_len,
            "custom_ratio_alpha":         sum(c.isalpha() for c in url) / total_len,
            "custom_num_redirects":       url.count("http", 1),
        }
    except Exception:
        return {k: 0 for k in [
            "custom_num_hyphens", "custom_num_at", "custom_num_percent", "custom_num_underscores",
            "custom_num_slashes", "custom_uses_http", "custom_has_www", "custom_domain_has_hyphen",
            "custom_domain_starts_num", "custom_domain_vowel_ratio", "custom_domain_digit_ratio",
            "custom_has_multiple_tlds", "custom_path_depth", "custom_path_has_exe", "custom_path_has_login",
            "custom_path_has_encoded", "custom_has_ip", "custom_has_at_symbol", "custom_has_port",
            "custom_has_hex", "custom_is_shortened", "custom_has_typosquat", "custom_has_suspicious_kw",
            "custom_num_suspicious_kw", "custom_has_brand_name", "custom_has_suspicious_tld",
            "custom_has_urgency_words", "custom_has_financial_words", "custom_is_free_hosting",
            "custom_url_entropy", "custom_ratio_digits", "custom_ratio_special", "custom_ratio_alpha",
            "custom_num_redirects"
        ]}
 
 
def extract_lexical_features(url: str) -> dict:
    url_lower = url.lower()
    domain_part, path_part = _split_domain_path(url)
    words = re.findall(r'[a-zA-Z]+', url)
    domain_words = re.findall(r'[a-zA-Z]+', domain_part)
    total_len = max(len(url), 1)
    domain_len = max(len(domain_part), 1)
 
    try:
        return {
            "lex_num_words":              len(words),
            "lex_avg_word_length":        float(np.mean([len(w) for w in words])) if words else 0,
            "lex_longest_word_length":    max([len(w) for w in words]) if words else 0,
            "lex_shortest_word_length":   min([len(w) for w in words]) if words else 0,
            "lex_num_unique_chars":       len(set(url_lower)),
            "lex_char_diversity":         len(set(url_lower)) / total_len,
            "lex_consonant_ratio":        sum(c in "bcdfghjklmnpqrstvwxyz" for c in url_lower) / total_len,
            "lex_vowel_ratio":            sum(c in "aeiou" for c in url_lower) / total_len,
            "lex_domain_consonant_ratio": sum(c in "bcdfghjklmnpqrstvwxyz" for c in domain_part.lower()) / domain_len,
            "lex_has_random_string":      int(bool(re.search(r'[a-z0-9]{12,}', url_lower))),
            "lex_longest_consonant_seq":  max([len(m.group()) for m in re.finditer(r'[bcdfghjklmnpqrstvwxyz]+', url_lower)] or [0]),
            "lex_longest_digit_seq":      max([len(m.group()) for m in re.finditer(r'\d+', url)] or [0]),
            "lex_num_domain_words":       len(domain_words),
            "lex_avg_domain_word_len":    float(np.mean([len(w) for w in domain_words])) if domain_words else 0,
            "lex_has_repeated_chars":     int(bool(re.search(r'(.)\1{2,}', url_lower))),
            "lex_has_repeated_digits":    int(bool(re.search(r'(\d)\1{2,}', url))),
            "lex_num_numbers_in_domain":  len(re.findall(r'\d+', domain_part)),
            "lex_has_year_in_domain":     int(bool(re.search(r'20[0-9]{2}', domain_part))),
            "lex_num_path_words":         len(re.findall(r'[a-zA-Z]+', path_part)),
            "lex_path_has_hex_string":    int(bool(re.search(r'[0-9a-f]{8,}', path_part.lower()))),
        }
    except Exception:
        return {k: 0 for k in [
            "lex_num_words", "lex_avg_word_length", "lex_longest_word_length", "lex_shortest_word_length",
            "lex_num_unique_chars", "lex_char_diversity", "lex_consonant_ratio", "lex_vowel_ratio",
            "lex_domain_consonant_ratio", "lex_has_random_string", "lex_longest_consonant_seq",
            "lex_longest_digit_seq", "lex_num_domain_words", "lex_avg_domain_word_len",
            "lex_has_repeated_chars", "lex_has_repeated_digits", "lex_num_numbers_in_domain",
            "lex_has_year_in_domain", "lex_num_path_words", "lex_path_has_hex_string"
        ]}
 
 
def extract_brand_features(url: str) -> dict:
    url_lower = url.lower()
    domain_part, path_part = _split_domain_path(url)
    clean_domain = domain_part.replace("www.", "")
    domain_parts = clean_domain.split(".")
 
    try:
        brand_in_domain = int(any(b in clean_domain for b in KNOWN_BRANDS))
        brand_in_path = int(any(b in path_part.lower() for b in KNOWN_BRANDS))
        brand_in_subdomain = 0
        if len(domain_parts) > 2:
            subdomain = ".".join(domain_parts[:-2])
            brand_in_subdomain = int(any(b in subdomain for b in KNOWN_BRANDS))
 
        main_domain = domain_parts[-2] if len(domain_parts) >= 2 else clean_domain
 
        if FUZZY_AVAILABLE:
            max_fuzzy_score = max([fuzz.ratio(main_domain, brand) for brand in KNOWN_BRANDS], default=0)
            is_typosquat = int(max_fuzzy_score > 70 and max_fuzzy_score < 100)
        else:
            max_fuzzy_score = 0
            is_typosquat = int(bool(re.search(
                r'paypa[^l]|go[^o]gle|gogle|amaz[^o]n|faceb[^o]ok|micros[^o]ft|netfl[^i]x|twitt[^e]r',
                url_lower
            )))
 
        brands_count = sum(b in url_lower for b in KNOWN_BRANDS)
        brand_path_not_domain = int(brand_in_path and not brand_in_domain)
        brand_domain_impersonation = int(
            brand_in_domain and len(domain_parts) > 2 and not any(b in main_domain for b in KNOWN_BRANDS)
        )
 
        return {
            "brand_in_domain":            brand_in_domain,
            "brand_in_path":              brand_in_path,
            "brand_in_subdomain":         brand_in_subdomain,
            "brand_path_not_domain":      brand_path_not_domain,
            "brand_multiple_brands":      int(brands_count > 1),
            "brand_count":                brands_count,
            "brand_max_fuzzy_score":      max_fuzzy_score / 100,
            "brand_is_typosquat":         is_typosquat,
            "brand_domain_impersonation": brand_domain_impersonation,
        }
    except Exception:
        return {k: 0 for k in [
            "brand_in_domain", "brand_in_path", "brand_in_subdomain", "brand_path_not_domain",
            "brand_multiple_brands", "brand_count", "brand_max_fuzzy_score", "brand_is_typosquat",
            "brand_domain_impersonation"
        ]}
 
 
def extract_tld_features(url: str) -> dict:
    domain_part, _ = _split_domain_path(url)
    clean_domain = domain_part.replace("www.", "")
    parts = clean_domain.split(".")
 
    TOP_LEGITIMATE_TLDS = {
        'com': 0.9, 'org': 0.85, 'net': 0.8, 'edu': 0.95, 'gov': 0.98,
        'uk': 0.85, 'de': 0.85, 'fr': 0.85, 'au': 0.85, 'ca': 0.85, 'jp': 0.85
    }
    HIGH_RISK_TLDS = {
        'tk': 0.9, 'ml': 0.85, 'ga': 0.85, 'cf': 0.85, 'gq': 0.85, 'xyz': 0.7,
        'top': 0.7, 'club': 0.65, 'online': 0.6, 'site': 0.6, 'pw': 0.8
    }
 
    try:
        tld = parts[-1].lower() if parts else ""
        second_level = parts[-2].lower() if len(parts) >= 2 else ""
        return {
            "tld_is_high_risk":            int(tld in HIGH_RISK_TLDS),
            "tld_risk_score":              HIGH_RISK_TLDS.get(tld, 0.0),
            "tld_legitimate_score":        TOP_LEGITIMATE_TLDS.get(tld, 0.5),
            "tld_is_country_code":         int(len(tld) == 2),
            "tld_is_common_legit":         int(tld in TOP_LEGITIMATE_TLDS),
            "tld_length":                  len(tld),
            "tld_is_new_gtld":             int(tld not in TOP_LEGITIMATE_TLDS and len(tld) > 3),
            "tld_num_domain_parts":        len(parts),
            "tld_second_level_length":     len(second_level),
            "tld_second_level_has_digits": int(bool(re.search(r'\d', second_level))),
            "tld_second_level_has_hyphen": int("-" in second_level),
        }
    except Exception:
        return {k: 0 for k in [
            "tld_is_high_risk", "tld_risk_score", "tld_legitimate_score", "tld_is_country_code",
            "tld_is_common_legit", "tld_length", "tld_is_new_gtld", "tld_num_domain_parts",
            "tld_second_level_length", "tld_second_level_has_digits", "tld_second_level_has_hyphen"
        ]}
 
 
def extract_combination_features(url: str, row: dict) -> dict:
    url_lower = url.lower()
    has_https = int(url_lower.startswith("https"))
    has_login_kw = int("login" in url_lower or "signin" in url_lower)
    has_verify_kw = int("verify" in url_lower or "confirm" in url_lower)
    has_brand = int(any(b in url_lower for b in KNOWN_BRANDS))
    has_susp_tld = int(bool(re.search(r'\.(tk|ml|ga|cf|gq|xyz|top|club|online|site)$', url_lower)))
    url_length = len(url)
    domain_part, path_part = _split_domain_path(url_lower)
 
    try:
        brand_in_domain = any(b in domain_part for b in KNOWN_BRANDS)
        brand_in_path = any(b in path_part for b in KNOWN_BRANDS)
        combo_brand_path_not_domain = int(brand_in_path and not brand_in_domain)
        return {
            "combo_brand_suspicious_tld":  int(has_brand and has_susp_tld),
            "combo_login_no_https":        int(has_login_kw and not has_https),
            "combo_verify_no_https":       int(has_verify_kw and not has_https),
            "combo_brand_path_not_domain": combo_brand_path_not_domain,
            "combo_long_url_many_params":  int(url_length > 100 and url.count("?") > 0 and url.count("&") > 2),
            "combo_ip_with_path":          int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)) and "/" in url),
            "combo_free_host_brand":       int(any(h in url_lower for h in ["wordpress", "wixsite", "weebly", "blogspot", "web.app", "vercel.app", "workers.dev"]) and has_brand),
            "combo_has_password_no_https": int(row.get('HasPasswordField', 0) == 1 and not has_https),
            "combo_hidden_field_login":    int(row.get('HasHiddenFields', 0) == 1 and has_login_kw),
        }
    except Exception:
        return {k: 0 for k in [
            "combo_brand_suspicious_tld", "combo_login_no_https", "combo_verify_no_https",
            "combo_brand_path_not_domain", "combo_long_url_many_params", "combo_ip_with_path",
            "combo_free_host_brand", "combo_has_password_no_https", "combo_hidden_field_login"
        ]}
 
 
def extract_features(url: str) -> dict:
    """
    Build the 81-feature vector for a single URL, in the exact order
    realtime_model.pkl / realtime_scaler.pkl expect.
 
    Mirrors training: query string and fragment are stripped before
    feature extraction (training used df['URL'].split('?')[0].split('#')[0]).
    """
    url_clean = url.split("?")[0].split("#")[0]
 
    # Real-time inference has no page content — see module docstring.
    no_page_content_row = {"HasPasswordField": 0, "HasHiddenFields": 0}
 
    merged = {}
    merged.update(extract_custom_features(url_clean))
    merged.update(extract_lexical_features(url_clean))
    merged.update(extract_brand_features(url_clean))
    merged.update(extract_tld_features(url_clean))
    merged.update(extract_combination_features(url_clean, no_page_content_row))
 
    missing = [k for k in FEATURE_ORDER if k not in merged]
    if missing:
        raise ValueError(f"feature_extractor.py is missing features required by the model: {missing}")
 
    # Strictly reorder to FEATURE_ORDER, dropping anything not in the trained feature set
    # (e.g. custom_has_www, combo_long_url_many_params — both dropped as constants at training time)
    return {k: merged[k] for k in FEATURE_ORDER}