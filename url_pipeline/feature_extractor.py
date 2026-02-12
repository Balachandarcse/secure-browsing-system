import re
import math
from urllib.parse import urlparse

SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "bit.do"
]

SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top",
    "work", "click", "country", "stream"
]



def entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum([p * math.log2(p) for p in prob])


def has_ip(host):
    return int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", host)))


def digit_letter_mix(s):
    return int(any(c.isdigit() for c in s) and any(c.isalpha() for c in s))


def extract_features(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    features = {}

    features["url_length"] = len(url)
    features["hostname_length"] = len(host)
    features["path_length"] = len(path)

    features["dot_count"] = url.count(".")
    features["hyphen_count"] = url.count("-")
    features["underscore_count"] = url.count("_")
    features["slash_count"] = url.count("/")
    features["question_count"] = url.count("?")
    features["equal_count"] = url.count("=")
    features["at_count"] = url.count("@")
    features["amp_count"] = url.count("&")

    features["subdomain_count"] = host.count(".")
    features["url_depth"] = path.count("/")

    features["has_ip"] = has_ip(host)
    features["is_shortened"] = int(any(s in host for s in SHORTENERS))
    features["digit_letter_mix"] = digit_letter_mix(host)

    tld = host.split(".")[-1] if "." in host else ""
    features["suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)

    features["domain_entropy"] = entropy(host)

    return features
