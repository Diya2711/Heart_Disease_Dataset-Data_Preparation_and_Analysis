#Task 1 
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
    def score(d):                             # delimiter = most consistent column count
        c = [len(split(l, d)) for l in lines[:30]]; m = max(set(c), key=c.count)
        return (c.count(m) if m > 1 else 0, m)
    d = max(",;\t|", key=score)
    t = [split(l, d) for l in lines]
    header = not any(isinstance(val(c), (int, float)) for c in t[0])
    cols = t[0] if header else (UCI if len(t[0]) == 14 else [f"col{i+1}" for i in range(len(t[0]))])
    return cols, [[val(c) for c in r] for r in t[header:] if len(r) == len(cols)], d

st.title("Task 1 · Load & browse dataset")
f = st.file_uploader("Dataset", ["csv", "tsv", "txt", "data"])
if not f: st.stop()
cols, rows, d = load(f.getvalue())
st.caption(f"{len(rows)} rows · {len(cols)} columns · delimiter {d!r}")

a, b, c = st.columns(3)
key = a.selectbox("Sort by", ["(none)"] + cols)
desc = b.checkbox("Descending")
size = c.selectbox("Rows per page", [10, 25, 50, 100])
if key != "(none)":
    j = cols.index(key)                      
    rows = sorted([r for r in rows if r[j] is not None], key=lambda r: (isinstance(r[j], str), r[j]), reverse=desc) \
           + [r for r in rows if r[j] is None]
pages = max(1, -(-len(rows) // size))
p = st.number_input(f"Page (1-{pages})", 1, pages, 1)
page = rows[(p - 1) * size: p * size]
data = {c: [r[i] for r in page] for i, c in enumerate(cols)}
st.dataframe({c: [str(x) for x in v] if any(isinstance(x, str) for x in v) else v for c, v in data.items()}, hide_index=True)
