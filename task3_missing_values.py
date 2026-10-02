#Task 3
import random
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

obs = lambda rows, j: [r[j] for r in rows if r[j] is not None]

def impute(rows, cont, how, tgt, g):
    n, out = len(cont), [r[:] for r in rows]
    sd = [(std(obs(rows, j)) or 1) if cont[j] else 1 for j in range(n)]
    def fill(sub, j):                         
        v = obs(sub, j) or obs(rows, j)
        return (mean(v) if how == "Mean / Mode" else median(v)) if cont[j] else mode(v)
    for i, r in enumerate(rows):
        miss = [j for j in range(n) if r[j] is None and j != tgt]
        if miss and how == "k-NN":            
            d = []
            for i2, r2 in enumerate(rows):
                s = [((r[j] - r2[j]) / sd[j]) ** 2 if cont[j] else float(r[j] != r2[j])
                     for j in range(n) if j != tgt and r[j] is not None and r2[j] is not None]
                if i2 != i and s: d.append((mean(s), i2))
            d.sort()
        for j in miss:
            if how == "k-NN":
                nb = [rows[i2][j] for _, i2 in d if rows[i2][j] is not None][:5]
                out[i][j] = (mean(nb) if cont[j] else mode(nb)) if nb else fill(rows, j)
            elif how == "Group-wise" and r[g] is not None:
                out[i][j] = fill([x for x in rows if x[g] == r[g]], j)
            else:
                out[i][j] = fill(rows, j)
    return out

@st.cache_data
def evaluate(rows, cont, tgt, g, methods):    
    sd = [(std(obs(rows, j)) or 1) if cont[j] else 1 for j in range(len(cont))]
    score = {m: [] for m in methods}
    cells = [(i, j) for i, r in enumerate(rows) for j in range(len(cont)) if j != tgt and r[j] is not None]
    for seed in range(3):
        pick = random.Random(seed).sample(cells, min(300, len(cells) // 10))
        masked = [r[:] for r in rows]
        for i, j in pick: masked[i][j] = None
        for m in methods:
            imp, err = impute(masked, cont, m, tgt, g), {}
            for i, j in pick:
                err.setdefault(j, []).append((imp[i][j] - rows[i][j]) ** 2 / sd[j] ** 2 if cont[j] else float(imp[i][j] != rows[i][j]))
            score[m].append(mean([mean(v) ** 0.5 if cont[j] else mean(v) for j, v in err.items()]))
    return {m: mean(v) for m, v in score.items()}

# app 
st.title("Task 3 · Missing values")
f = st.file_uploader("Dataset", ["csv", "tsv", "txt", "data"])
if not f: st.stop()
cols, rows = load(f.getvalue())
tgt = len(cols) - 1                           

zc = [cols.index(c) for c in ("trestbps", "chol", "thalach") if c in cols]
if st.checkbox("Treat 0 in trestbps / chol / thalach as missing", True):
    rows = [[None if (j in zc and v == 0) else v for j, v in enumerate(r)] for r in rows]
rows = [r for r in rows if r[tgt] is not None]
N, kept = len(rows), [r for r in rows if None not in r]       # kept = listwise deletion (manual dropna)
cont = [all(isinstance(v, (int, float)) for v in obs(rows, j)) and len(set(obs(rows, j))) > 10 for j in range(len(cols))]
miss = {c: N - len(obs(rows, j)) for j, c in enumerate(cols)}
total = 0
for m in miss.values(): total += m

st.subheader("1 · Identification")
a, b, c = st.columns(3)
a.metric("Missing cells", total); b.metric("Rows with a gap", N - len(kept)); c.metric("Lost by deletion", f"{100 * (N - len(kept)) / N:.1f}%")
l, r = st.columns(2)
l.dataframe({"column": list(miss), "missing": list(miss.values()), "%": [round(100 * m / N, 2) for m in miss.values()]}, hide_index=True)
r.bar_chart(miss)

st.subheader("2 · Imputation techniques")
g = cols.index("sex") if "sex" in cols else 0
methods = ("Mean / Mode", "Median / Mode", "Group-wise", "k-NN")
with st.spinner("Hiding known values and testing every method…"):
    res = evaluate(rows, cont, tgt, g, methods)
rank = sorted(res, key=res.get); best = rank[0]
st.dataframe({"rank": range(1, 5), "method": rank, "error (lower is better)": [round(res[m], 4) for m in rank]}, hide_index=True)
st.caption(f"Test: 10% of known values hidden (3 random masks), imputed, compared with the truth. Group-wise uses '{cols[g]}'.")
choice = st.selectbox("Apply method", rank)
final = impute(rows, cont, choice, tgt, g)
st.dataframe({c: [round(x[i], 2) if isinstance(x[i], float) else x[i] for x in final[:30]] for i, c in enumerate(cols)}, hide_index=True)
st.download_button("Download imputed CSV", "\n".join(",".join(map(str, r)) for r in [cols] + final), "heart_imputed.csv")

st.subheader("3 · Report - why this method")
base = res["Mean / Mode"]
st.markdown(f"""
**Finding:** {total} missing cells ({N - len(kept)} of {N} rows affected), detected with plain `None` checks - no `isna()/dropna()`.

**Chosen: {best}** - lowest error in the hidden-value test ({res[best]:.3f}{f", {100 * (base - res[best]) / base:.1f}% better than mean/mode" if best != "Mean / Mode" else ""}).

1. **Accuracy** - it reconstructs known values best, the only objective test since real gaps have no ground truth.
2. **No data loss** - all {N} rows are kept; deletion would drop {N - len(kept)} ({100 * (N - len(kept)) / N:.1f}%) and may bias the sample.
3. **Fit** - {"k-NN beating constants shows the features carry information about each other (age ↔ blood pressure, cholesterol, heart rate)." if best == "k-NN" else "a simple statistic was as good as the complex methods here, so the simplest option is preferred."}
4. **No leakage** - the label is never imputed or used as a predictor.

*Limitation:* assumes values are missing at random; when training the classifier, fit the imputer on the training split only.
""")
