import re, math, numpy as np

STOP = set("""a an the in on at of for to and or but if while as is are was were be been being from with by than then so because about into over after before under between out up down off near far very just not no nor""".split())

_SENT_SPLIT = re.compile(r"[.!?]+")
_WORD = re.compile(r"\b\w+\b", re.UNICODE)
_EMOJI = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def shannon_entropy(s: str) -> float:
    if not s: return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def sent_lengths(s: str):
    sents = [x.strip() for x in _SENT_SPLIT.split(s) if x.strip()]
    if not sents: return (0.0, 0.0)
    lens = [len(_WORD.findall(x)) for x in sents]
    return (float(np.mean(lens)), float(np.var(lens)))

def type_token_ratio(words):
    if not words: return 0.0
    return len(set(words)) / len(words)

def hapax_ratio(words):
    if not words: return 0.0
    from collections import Counter
    c = Counter(words)
    hapax = sum(1 for k,v in c.items() if v == 1)
    return hapax / len(c)

def function_word_ratio(words):
    if not words: return 0.0
    fw = sum(1 for w in words if w.lower() in STOP)
    return fw / len(words)

def repetition_index(words, n=3):
    # max frequency of any n-gram / total n-grams
    if len(words) < n: return 0.0
    from collections import Counter
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    cnt = Counter(grams)
    return max(cnt.values()) / max(1, len(grams))

def extract_features(text: str) -> np.ndarray:
    s = text.strip()
    words = _WORD.findall(s)
    chars = len(s) or 1
    sent_avg, sent_var = sent_lengths(s)

    features = [
        shannon_entropy(s),                         # char_entropy
        type_token_ratio(words),                    # ttr
        hapax_ratio(words),                         # hapax
        sent_avg,                                   # avg_sent_len
        sent_var,                                   # var_sent_len
        sum(1 for c in s if c in ".,;:!?")/chars,   # punct_rate
        sum(1 for c in s if c.isdigit())/chars,     # digit_rate
        len(_EMOJI.findall(s))/chars,               # emoji_rate
        function_word_ratio(words),                 # function_word_ratio
        repetition_index([w.lower() for w in words], n=3) # repetition_index
    ]
    return np.array(features, dtype=float)

FEATURE_NAMES = [
    "char_entropy","ttr","hapax","avg_sent_len","var_sent_len",
    "punct_rate","digit_rate","emoji_rate","function_word_ratio","repetition_index"
]
