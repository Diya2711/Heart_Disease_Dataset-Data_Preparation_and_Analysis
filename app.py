import os, random
import streamlit as st
UCI = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
MISSING = {"", "?", "na", "n/a", "nan", "null", "none"}
isnum = lambda v: isinstance(v, (int, float))

# Task 1
def split_line(line, d):
    out, buf, quoted = [], "", False
    for ch in line:                                   # quote-aware tokenizer
        if ch == '"': quoted = not quoted
        elif ch == d and not quoted: out.append(buf.strip()); buf = ""
        else: buf += ch
    out.append(buf.strip())
    return [f for f in out if f] if d == " " else out

def to_val(t):                                        # missing marker -> None, else int / float / str
    if t.lower() in MISSING: return None
    for cast in (int, float):
        try: return cast(t)
        except ValueError: pass
    return t

@st.cache_data
def load(raw):
    try: text = raw.decode("utf-8-sig")
    except UnicodeDecodeError: text = raw.decode("latin-1")
    lines = [l for l in text.splitlines() if l.strip()]
    def score(d):                                     # delimiter = most consistent column count
        c = [len(split_line(l, d)) for l in lines[:30]]; m = max(set(c), key=c.count)
        return (c.count(m) if m > 1 else 0, m)
    delim = max(",;\t| ", key=score)
    table = [split_line(l, delim) for l in lines]
    header = not any(isnum(to_val(c)) for c in table[0])
    cols = table[0] if header else (UCI if len(table[0]) == 14 else [f"col{i+1}" for i in range(len(table[0]))])
    rows = [[to_val(c) for c in r] for r in table[header:] if len(r) == len(cols)]
    return cols, rows, delim, header

def tbl(cols, rows):                                  # column dict for st.dataframe (no pandas in our code)
    d = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    return {c: v if all(v_ is None or isnum(v_) for v_ in v) else [str(x) for x in v] for c, v in d.items()}

# Task 2:
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
    c = {}
    for x in v: c[x] = c.get(x, 0) + 1
    best = None
    for k in c:
        if best is None or c[k] > c[best] or (c[k] == c[best] and k < best): best = k
    return best

def extremes(v):
    lo = hi = v[0]
    for x in v:
        lo, hi = (x if x < lo else lo), (x if x > hi else hi)
    return lo, hi

def std(v):
    m, s = mean(v), 0
    for x in v: s += (x - m) ** 2
    return (s / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0

obs = lambda rows, j: [r[j] for r in rows if r[j] is not None]

# Task 3:
def impute(rows, kinds, how, tgt, g=None, k=5):
    n = len(kinds); out = [r[:] for r in rows]
    sd = {j: std(obs(rows, j)) or 1.0 for j in range(n) if kinds[j] == "cont"}
    def center(sub, j):
        v = obs(sub, j)
        if not v: return None
        return (mean(v) if how == "Mean / Mode" else median(v)) if kinds[j] == "cont" else mode(v)
    for i, r in enumerate(rows):
        miss = [j for j in range(n) if r[j] is None and j != tgt]
        if not miss: continue
        if how == "k-NN (k=5)":
            d = []
            for i2, r2 in enumerate(rows):
                s = [((r[j] - r2[j]) / sd[j]) ** 2 if kinds[j] == "cont" else float(r[j] != r2[j])
                     for j in range(n) if j != tgt and r[j] is not None and r2[j] is not None]
                if i2 != i and s: d.append((mean(s), i2))
            d.sort()
        for j in miss:
            if how == "k-NN (k=5)":
                nb = [rows[i2][j] for _, i2 in d if rows[i2][j] is not None][:k]
                out[i][j] = (mean(nb) if kinds[j] == "cont" else mode(nb)) if nb else center(rows, j)
            else:
                sub = [x for x in rows if x[g] == r[g]] if how.startswith("Group") and r[g] is not None else rows
                c = center(sub, j)
                out[i][j] = c if c is not None else center(rows, j)
    return out

@st.cache_data
def evaluate(rows, kinds, tgt, g, methods):
    """Hide 10% of known values, impute, compare with the truth (NRMSE for numeric, error rate for categorical)."""
    sd = {j: std(obs(rows, j)) or 1.0 for j in range(len(kinds)) if kinds[j] == "cont"}
    score = {m: [] for m in methods}
    for seed in range(3):
        rnd = random.Random(seed)
        cells = [(i, j) for i, r in enumerate(rows) for j in range(len(kinds)) if j != tgt and r[j] is not None]
        pick = rnd.sample(cells, min(300, len(cells) // 10))
        masked = [r[:] for r in rows]
        for i, j in pick: masked[i][j] = None
        for m in methods:
            imp, err = impute(masked, kinds, m, tgt, g), {}
            for i, j in pick:
                e = (imp[i][j] - rows[i][j]) ** 2 / sd[j] ** 2 if kinds[j] == "cont" else float(imp[i][j] != rows[i][j])
                err.setdefault(j, []).append(e)
            score[m].append(mean([mean(v) ** 0.5 if kinds[j] == "cont" else mean(v) for j, v in err.items()]))
    return {m: mean(v) for m, v in score.items()}

# Streamlit app for all tasks
st.set_page_config(page_title="Heart Disease - Data Preparation", page_icon="❤️", layout="wide")
st.title("🫀Heart Disease Prediction - Data Preparation")
up = st.sidebar.file_uploader("Upload dataset (csv / tsv / txt / .data)")
path = st.sidebar.text_input("…or local file path", "heart.csv")
raw = up.getvalue() if up else open(path, "rb").read() if os.path.isfile(path) else None
if raw is None: st.info("Upload a dataset or put `heart.csv` next to this script."); st.stop()

cols, rows, delim, header = load(raw)
tgt = cols.index(st.sidebar.selectbox("Label column", cols, index=len(cols) - 1))
zero_missing = st.sidebar.checkbox("Treat 0 in trestbps / chol / thalach as missing", True)
t1, t2, t3 = st.tabs(["1 · Load & browse", "2 · Summary statistics", "3 · Missing values"])

with t1:
    a, b, c, d = st.columns(4)
    a.metric("Rows", len(rows)); b.metric("Columns", len(cols))
    c.metric("Delimiter", {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe", " ": "space"}[delim])
    d.metric("Header in file", "yes" if header else "no (assigned)")
    s1, s2, s3 = st.columns([2, 1, 1])
    key = s1.selectbox("Sort by", ["(none)"] + cols)
    desc = s2.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Descending"
    size = s3.selectbox("Rows per page", [10, 25, 50, 100], 1)
    view = rows
    if key != "(none)":
        j = cols.index(key)
        have = sorted([r for r in rows if r[j] is not None], key=lambda r: (isinstance(r[j], str), str(r[j]) if isinstance(r[j], str) else r[j]), reverse=desc)
        view = have + [r for r in rows if r[j] is None]          
    pages = max(1, -(-len(view) // size))
    page = st.number_input(f"Page (1-{pages})", 1, pages, 1)
    st.dataframe(tbl(cols, view[(page - 1) * size: page * size]), hide_index=True)
kinds = ["cont" if all(isnum(v) for v in obs(rows, j)) and len(set(obs(rows, j))) > 10 else "cat" for j in range(len(cols))]

with t2:
    st.caption("Mean, median, mode, min, max and std are computed with hand-written functions.")
    out = {k: [] for k in ["feature", "count", "missing", "mean", "median", "mode", "min", "max", "std"]}
    for j, c in enumerate(cols):
        v = obs(rows, j)
        if v and all(isnum(x) for x in v):
            lo, hi = extremes(v)
            for k, x in zip(out, [c, len(v), len(rows) - len(v), mean(v), median(v), mode(v), lo, hi, std(v)]):
                out[k].append(round(x, 3) if isinstance(x, float) else x)
    st.dataframe(out, hide_index=True)
    st.caption("Mean / std are meaningful for continuous features (age, trestbps, chol, thalach, oldpeak); for binary / categorical columns read the mode.")

with t3:
    work = [r[:] for r in rows]
    if zero_missing:
        for c in ("trestbps", "chol", "thalach"):
            if c in cols:
                for r in work:
                    if r[cols.index(c)] == 0: r[cols.index(c)] = None
    work = [r for r in work if r[tgt] is not None]              
    N, kept = len(work), [r for r in work if None not in r]     
    miss = {c: len(work) - len(obs(work, j)) for j, c in enumerate(cols)}
    total = 0
    for m in miss.values(): total += m

    st.subheader("Identification")
    a, b, c = st.columns(3)
    a.metric("Missing cells", total); b.metric("Rows with ≥1 gap", N - len(kept))
    c.metric("Rows lost by deletion", f"{100 * (N - len(kept)) / N:.1f}%")
    left, right = st.columns(2)
    left.dataframe({"column": list(miss), "type": ["continuous" if k == "cont" else "categorical" for k in kinds],
                 "missing": list(miss.values()), "% missing": [round(100 * m / N, 2) for m in miss.values()]}, hide_index=True)
    right.bar_chart(miss)

    st.subheader("Imputation techniques & selection")
    g = cols.index("sex") if "sex" in cols else next(j for j in range(len(cols)) if kinds[j] == "cat" and j != tgt)
    methods = ("Mean / Mode", "Median / Mode", "Group-wise", "k-NN (k=5)")
    with st.spinner("Hiding known values and testing every method…"):
        res = evaluate(work, kinds, tgt, g, methods)
    ranking = sorted(res, key=res.get)
    best = ranking[0]
    st.dataframe({"rank": range(1, len(res) + 1), "method": [f"{m} by {cols[g]}" if m == "Group-wise" else m for m in ranking], "error (lower = better)": [round(res[m], 4) for m in ranking]}, hide_index=True)
    st.caption("Experiment: 10 % of known values hidden at random (3 masks) → imputed → compared with the truth. Numeric = normalised RMSE, categorical = error rate.")
    chosen = st.selectbox("Method to apply", ranking, 0)
    final = impute(work, kinds, chosen, tgt, g)
    st.dataframe(tbl(cols, final[:50]), hide_index=True)
    csv = "\n".join([",".join(cols)] + [",".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in r) for r in final])
    st.download_button("⬇ Download imputed dataset", csv, "heart_imputed.csv", "text/csv")

    st.subheader("Report - method justification")
    base = res["Mean / Mode"]; gain = 100 * (base - res[best]) / base
    report = f"""
**Findings.** {total} missing cells were found without `isna()/dropna()` (own `None` checks on the parsed markers `?`, `NA`, blank{' and impossible zeros' if zero_missing else ''}); {N - len(kept)} of {N} rows contain a gap. Most affected: {', '.join(c for c in sorted(miss, key=miss.get, reverse=True)[:3] if miss[c])  or 'none'}.

**Compared:** listwise deletion, mean/mode, median/mode, group-wise by `{cols[g]}`, and k-NN - all implemented manually.

**Chosen: {best}** - lowest error in the masked-value test ({res[best]:.3f}{f', {gain:.1f}% better than mean/mode ({base:.3f})' if best != 'Mean / Mode' else ''}).

**Justification**
1. *Accuracy:* it reconstructed hidden values best, which is the only objective test available because real gaps have no ground truth.
2. *Data retention:* every row is kept, whereas deletion would discard {N - len(kept)} rows ({100 * (N - len(kept)) / N:.1f}%) and can bias the sample.
3. *Fit to the data:* {'k-NN beating constants shows the clinical variables carry information about each other (age ↔ blood pressure, cholesterol, max heart rate), so similar patients give better estimates.' if best.startswith('k-NN') else 'the simple statistic was as accurate as the complex methods here, so the simplest transparent option is preferred.'}
4. *Pipeline safety:* the label is never imputed or used as a predictor, avoiding target leakage.

**Limitations.** The test assumes values are missing at random; when training the classifier, fit the imputer on the *training split only* and consider missing-indicator features.
"""
    st.markdown(report)
    st.download_button(" Download report", report, "task3_report.md")
