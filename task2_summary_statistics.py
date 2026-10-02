#Task 2 
import streamlit as st

UCI = "age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal target".split()
MISSING = {"", "?", "na", "n/a", "nan", "null", "none"}

def val(t):
    if t.lower() in MISSING: return None
    for cast in (int, float):
        try: return cast(t)
        except ValueError: pass
    return t

def split(line, d):
    out, buf, q = [], "", False
    for ch in line:
        if ch == '"': q = not q
        elif ch == d and not q: out.append(buf.strip()); buf = ""
        else: buf += ch
    return out + [buf.strip()]

def load(raw):
    lines = [l for l in raw.decode("utf-8-sig", errors="replace").splitlines() if l.strip()]
    def score(d):
        c = [len(split(l, d)) for l in lines[:30]]; m = max(set(c), key=c.count)
        return (c.count(m) if m > 1 else 0, m)
    d = max(",;\t|", key=score)
    t = [split(l, d) for l in lines]
    header = not any(isinstance(val(c), (int, float)) for c in t[0])
    cols = t[0] if header else (UCI if len(t[0]) == 14 else [f"col{i+1}" for i in range(len(t[0]))])
    return cols, [[val(c) for c in r] for r in t[header:] if len(r) == len(cols)]

def msort(a):                                 
    if len(a) < 2: return a
    l, r, out = msort(a[:len(a) // 2]), msort(a[len(a) // 2:]), []
    while l and r: out.append(l.pop(0) if l[0] <= r[0] else r.pop(0))
    return out + l + r

def mean(v):
    t = 0
    for x in v: t += x
    return t / len(v)

def median(v):
    s, n = msort(list(v)), len(v)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def mode(v):                                
    c, best = {}, None
    for x in v: c[x] = c.get(x, 0) + 1
    for k in c:
        if best is None or c[k] > c[best] or (c[k] == c[best] and k < best): best = k
    return best

def std(v):                                  
    m, s = mean(v), 0
    for x in v: s += (x - m) ** 2
    return (s / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0

def min_max(v):
    lo = hi = v[0]
    for x in v: lo, hi = (x if x < lo else lo), (x if x > hi else hi)
    return lo, hi

st.title("Task 2 · Summary statistics (manual)")
f = st.file_uploader("Dataset", ["csv", "tsv", "txt", "data"])
if not f: st.stop()
cols, rows = load(f.getvalue())

out = {k: [] for k in ["feature", "count", "missing", "mean", "median", "mode", "min", "max", "std"]}
for j, c in enumerate(cols):
    v = [r[j] for r in rows if r[j] is not None]
    if v and all(isinstance(x, (int, float)) for x in v):
        lo, hi = min_max(v)
        for k, x in zip(out, [c, len(v), len(rows) - len(v), mean(v), median(v), mode(v), lo, hi, std(v)]):
            out[k].append(round(x, 3) if isinstance(x, float) else x)
st.dataframe(out, hide_index=True)
st.caption("Mean/std are meaningful for continuous features (age, trestbps, chol, thalach, oldpeak); for binary/categorical columns read the mode.")
