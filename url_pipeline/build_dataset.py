import pandas as pd
import re
import math
from urllib.parse import urlparse

INPUT = "/app/dataset/urls.csv"
OUTPUT = "/app/dataset/url_features.csv"
CHUNK_SIZE = 50000


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

    # Length based
    features["url_length"] = len(url)
    features["hostname_length"] = len(host)
    features["path_length"] = len(path)

    # Character counts
    features["dot_count"] = url.count(".")
    features["hyphen_count"] = url.count("-")
    features["underscore_count"] = url.count("_")
    features["slash_count"] = url.count("/")
    features["question_count"] = url.count("?")
    features["equal_count"] = url.count("=")
    features["at_count"] = url.count("@")
    features["amp_count"] = url.count("&")

    # Structure
    features["subdomain_count"] = host.count(".")
    features["url_depth"] = path.count("/")

    # Suspicious patterns
    features["has_ip"] = has_ip(host)
    features["is_shortened"] = int(any(s in host for s in SHORTENERS))
    features["digit_letter_mix"] = digit_letter_mix(host)

    # TLD
    tld = host.split(".")[-1] if "." in host else ""
    features["suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)


    # Random looking domain
    features["domain_entropy"] = entropy(host)

    return features


def label_map(t):
    t = t.lower()
    if t == "benign":
        return 0
    if t == "phishing":
        return 1
    return 2  # defacement


def main():
    print("Building advanced phishing feature dataset...")

    writer = None

    for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE):
        chunk = chunk.dropna()

        rows = []
        for _, row in chunk.iterrows():
            feats = extract_features(str(row["url"]))
            feats["label"] = label_map(row["type"])
            rows.append(feats)

        df_feats = pd.DataFrame(rows)

        if writer is None:
            df_feats.to_csv(OUTPUT, index=False)
            writer = True
        else:
            df_feats.to_csv(OUTPUT, mode="a", header=False, index=False)

        print("Processed chunk...")

    print("✅ url_features.csv created successfully")


if __name__ == "__main__":
    main()
