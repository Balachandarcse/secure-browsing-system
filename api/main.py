from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
import requests
from urllib.parse import urlparse

from url_pipeline.feature_extractor import extract_features

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "chrome-extension://*",
    "https://www.google.com",
    "https://search.brave.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


model = joblib.load("/app/dataset/url_model.pkl")

with open("/app/api/feature_order.json") as f:
    FEATURE_ORDER = json.load(f)


class URLRequest(BaseModel):
    url: str


SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "buff.ly", "adf.ly", "is.gd", "bit.do"
}


def unshorten_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=5, allow_redirects=True)
        return r.url
    except:
        return url


def is_shortened(url: str) -> bool:
    domain = urlparse(url).netloc
    return domain in SHORTENER_DOMAINS


@app.post("/predict")
def predict_url(data: URLRequest):
    original_url = data.url

    shortened = is_shortened(original_url)
    final_url = unshorten_url(original_url) if shortened else original_url

    feats = extract_features(final_url)

    # 🔥 Correct feature order
    feature_vector = [feats[f] for f in FEATURE_ORDER]
    features = np.array([feature_vector])

    prediction = int(model.predict(features)[0])
    confidence = float(max(model.predict_proba(features)[0]))

    return {
        "original_url": original_url,
        "final_url": final_url,
        "is_shortened": shortened,
        "prediction": prediction,
        "confidence": confidence
    }
