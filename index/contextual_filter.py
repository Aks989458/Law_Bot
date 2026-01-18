import re
from collections import Counter

STOPWORDS = {
    "the", "is", "for", "under", "of", "and", "to",
    "what", "which", "who", "when", "how", "punishment",
    "section", "act", "ipc", "crpc"
}

def extract_query_keywords(query, min_len=4):
    words = re.findall(r"[a-zA-Z]+", query.lower())
    return [
        w for w in words
        if w not in STOPWORDS and len(w) >= min_len
    ]


def contextual_filter(chunks, query, min_matches=1):
    """
    Dynamically filters chunks based on query relevance.

    Rules:
    - Extract keywords from query
    - Keep chunk if it matches >= min_matches keywords
    - OR contains legal signal like 'punishment', 'imprisonment', 'fine'
    """

    keywords = extract_query_keywords(query)

    legal_signals = {
        "punishment", "imprisonment", "fine", "liable",
        "shall be punished", "sentence", "offence"
    }

    filtered = []

    for c in chunks:
        text = c["text"].lower()

        keyword_hits = sum(1 for k in keywords if k in text)
        legal_hits = sum(1 for s in legal_signals if s in text)

        # Dynamic decision
        if keyword_hits >= min_matches or legal_hits >= 1:
            filtered.append(c)

    return filtered
