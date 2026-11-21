import json, os
import numpy as np, pandas as pd, streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, Draw, rdFingerprintGenerator
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AChE Inhibitor Predictor", page_icon="🧠", layout="wide")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

RES = json.load(open("results.json")) if os.path.exists("results.json") else None
SCAF_AUC = RES["scaffold"]["roc_auc"] if RES else 0.873

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&display=swap');
h1, h2, h3 {font-family: 'Source Serif 4', Georgia, serif !important; color: #1F2328;}
.hero {background: linear-gradient(120deg, #1F2328 0%, #2B3138 60%, #3A424B 100%);
    padding: 1.8rem 2.2rem; border-radius: 12px; margin-bottom: 1.2rem; border-left: 6px solid #F59E0B;}
.hero .tag {color: #FCD34D; font-size: .78rem; letter-spacing: .14em; font-weight: 600;}
.hero h1 {color: #FFFFFF !important; margin: .2rem 0 .3rem 0; font-size: 2.1rem;}
.hero p {color: #D5D9DE; margin: 0; font-size: 1rem;}
.hero .meta {color: #98A1AC; font-size: .85rem; margin-top: .7rem;}
[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E2E2DF; border-radius: 10px;
    padding: .8rem 1rem; box-shadow: 0 1px 3px rgba(31,35,40,.07);}
[data-testid="stMetricLabel"] {color: #6B7280;}
[data-testid="stMetricValue"] {color: #1F2328;}
.stButton > button {border: 1px solid #B45309; color: #B45309; background: #FFFFFF; border-radius: 8px;}
.stButton > button:hover {background: #B45309; color: #FFFFFF; border-color: #B45309;}
.stTabs [data-baseweb="tab"] {font-weight: 600; color: #6B7280;}
.stTabs [aria-selected="true"] {color: #B45309;}
.block-container {padding-top: 2rem; max-width: 1150px;}
</style>
<div class="hero">
  <div class="tag">CHEMINFORMATICS · NEUROPHARMACOLOGY</div>
  <h1>AChE Inhibitor Predictor</h1>
  <p>Machine learning-based prioritization of brain-penetrant acetylcholinesterase inhibitors</p>
  <div class="meta">Random forest · Morgan fingerprints · ChEMBL training data</div>
</div>
""", unsafe_allow_html=True)

def to_arr(fp):
    a = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a

@st.cache_resource
def load():
    train = pd.read_csv("ache_clean.csv")
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train["smiles"]]
    X = np.array([to_arr(f) for f in fps])
    model = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X, train["label"].values)
    return model, fps, train

@st.cache_resource
def pains_catalog():
    p = FilterCatalogParams()
    p.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(p)

model, train_fps, train = load()
pains_cat = pains_catalog()

def props(mol):
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol)
    tpsa, qed = rdMolDescriptors.CalcTPSA(mol), QED.qed(mol)
    lip = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return dict(mw=mw, logp=logp, hbd=hbd, hba=hba, tpsa=tpsa, qed=qed, lip=lip,
                bbb=(tpsa < 90 and mw < 450), pains=len(pains_cat.GetMatches(mol)))

def predict(mol):
    fp = gen.GetFingerprint(mol)
    p = model.predict_proba(to_arr(fp).reshape(1, -1))[0, 1]
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    return p, sims

def verdict(p, sim, bbb):
    if sim < 0.4:
        return "⚠️ Unreliable: far from training data", "warning"
    if p >= 0.5 and bbb:
        return "✅ Priority candidate: likely active and brain-penetrant", "success"
    if p >= 0.5:
        return "🟡 Likely active, but may not reach the brain", "warning"
    return "❌ Likely inactive", "error"

def find_col(df, *keys):
    for c in df.columns:
        name = str(c).lower()
        if all(k in name for k in keys):
            return c
    return None

def num(v, fallback):
    v = pd.to_numeric(v, errors="coerce")
    return fallback if pd.isna(v) else int(v)

EXAMPLES = {
    "Galantamine": "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
    "Donepezil": "COc1cc2c(cc1OC)C(=O)C(C2)CC1CCN(CC1)Cc1ccccc1",
    "Rivastigmine": "CCN(C)C(=O)Oc1cccc(c1)[C@H](C)N(C)C",
    "Tacrine": "Nc1c2CCCCc2nc2ccccc12",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}
if "smiles" not in st.session_state:
    st.session_state.smiles = EXAMPLES["Galantamine"]

def set_smiles(s):
    st.session_state.smiles = s

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔬 Single molecule", "📊 Model performance",
                                        "🧬 Structural features", "🧪 Screening & shortlist", "ℹ️ About"])

# ---------------- Objective 1: classifier, single molecule ----------------
with tab1:
    st.caption("Try an example, or paste your own SMILES:")
    cols = st.columns(len(EXAMPLES))
    for c, (name, s) in zip(cols, EXAMPLES.items()):
        c.button(name, on_click=set_smiles, args=(s,), use_container_width=True)
    smiles = st.text_input("SMILES", key="smiles")

    mol = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
    if smiles.strip() and mol is None:
        st.error("Invalid SMILES. Please check the structure.")
    elif mol is not None:
        p, sims = predict(mol)
        pr = props(mol)
        top = sims.argsort()[::-1][:3]
        best = sims[top[0]]

        left, right = st.columns([1, 1.3])
        with left:
            st.image(Draw.MolToImage(mol, size=(420, 320)))
        with right:
            st.metric("Predicted probability of AChE activity", f"{p:.0%}")
            st.progress(float(p))
            text, kind = verdict(p, best, pr["bbb"])
            getattr(st, kind)(text)
            st.write(f"**Brain-penetration quick rule:** "
                     f"{'✅' if pr['tpsa'] < 90 else '❌'} TPSA < 90 ({pr['tpsa']:.0f})  ·  "
                     f"{'✅' if pr['mw'] < 450 else '❌'} MW < 450 ({pr['mw']:.0f})")
            st.write(f"**PAINS alerts:** {pr['pains']}  ·  **Similarity to nearest training molecule:** {best:.2f}")

        st.subheader("Drug-likeness")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Mol. weight", f"{pr['mw']:.0f}")
        c2.metric("logP", f"{pr['logp']:.1f}")
        c3.metric("H-donors", pr["hbd"])
        c4.metric("H-acceptors", pr["hba"])
        c5.metric("QED", f"{pr['qed']:.2f}")
        c6.metric("Lipinski violations", pr["lip"])

        st.subheader("Most similar molecules the model learned from")
        nbrs = [Chem.MolFromSmiles(train["smiles"].iloc[i]) for i in top]
               def tag(i):
            nm = "" if "name" not in train.columns or pd.isna(train["name"].iloc[i]) else str(train["name"].iloc[i])[:22]
            return f"{nm} | {'Active' if train['label'].iloc[i] == 1 else 'Inactive'} | {sims[i]:.2f}"
        legends = [tag(i) for i in top]
        st.image(Draw.MolsToGridImage(nbrs, molsPerRow=3, subImgSize=(340, 220), legends=legends))
        st.caption(f"This is a computer prediction, not a lab result. On unseen scaffolds the model's "
                   f"ROC-AUC was {SCAF_AUC}, so expect errors.")

# ---------------- Objective 2: honest evaluation ----------------
with tab2:
    if RES is None:
        st.info("Upload results.json and roc.png (made in Colab) to the GitHub repo to show this tab.")
    else:
        st.write(f"Dataset: {RES['n_molecules']} molecules, {RES['active_rate']}% active. "
                 "Same model settings, evaluated two ways.")
        rows = []
        for k, label in [("random", "Random split"), ("scaffold", "Scaffold split")]:
            r = RES[k]
            rows.append({"Split": label, "Train": r["n_train"], "Test": r["n_test"],
                         "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"], "Accuracy": r["accuracy"],
                         "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"], "MCC": r["mcc"]})
        st.dataframe(pd.DataFrame(rows).set_index("Split"), use_container_width=True)
        gap = RES["random"]["roc_auc"] - RES["scaffold"]["roc_auc"]
        c1, c2 = st.columns([1, 1])
        if os.path.exists("roc.png"):
            c1.image("roc.png")
        c2.markdown(f"""
**How to read this**

- A **random split** puts molecules from the same chemical series in both training and test sets, so it tends to look optimistic.
- A **scaffold split** keeps whole scaffolds out of training, which is closer to screening genuinely new chemistry. This is the more honest number.
- ROC-AUC gap (random minus scaffold): **{gap:.3f}**. The bigger the gap, the more the model depends on recognizing close relatives of molecules it has already seen.
- Precision is the share of predicted actives that were truly active. Recall is the share of true actives that were found.
""")

# ---------------- Objective 3: structural features ----------------
with tab3:
    if RES is None or "features" not in RES:
        st.info("Upload results.json and features.png (made in Colab) to the GitHub repo to show this tab.")
    else:
        st.write("The 10 Morgan fingerprint bits the random forest relies on most. Highlighted atoms show "
                 "the fragment each bit encodes, in an example molecule. The legend gives the share of "
                 "training molecules with that bit that are active.")
        if os.path.exists("features.png"):
            st.image("features.png")
        ft = pd.DataFrame(RES["features"]).rename(columns={
            "rank": "Rank", "bit": "Bit", "importance": "Importance", "n_molecules": "Molecules with bit",
            "pct_active": "% active", "enriched_in": "More common in"})
        st.dataframe(ft.set_index("Rank"), use_container_width=True)
        st.subheader("Known AChE chemistry to compare with")
        st.markdown("""
- **Basic or quaternary nitrogen** (protonated amine): binds the catalytic anionic site through a cation–π interaction with Trp86 (galantamine, donepezil, tacrine).
- **Benzyl-piperidine and other aromatic groups**: π–π stacking with Trp286 at the peripheral anionic site; donepezil spans both sites.
- **Carbamate group**: carbamoylates the catalytic Ser203 (rivastigmine, physostigmine).
- **Planar fused aromatic rings** such as aminoacridine or aminoquinoline (tacrine): stack inside the active-site gorge.
- **Dual-site designs**: two aromatic ends joined by a linker that spans the gorge.
""")
        st.caption("Check whether the highlighted fragments resemble these. Fingerprint bits are hashed, so one bit "
                   "can stand for more than one fragment. Treat the comparison as a sanity check, not proof.")

# ---------------- Objective 4: screening and shortlist ----------------
with tab4:
    st.write("Upload a compound library as a CSV: a plain list with a **smiles** column, or a **SwissADME export**. "
             "For a SwissADME file, its BBB, GI absorption, P-gp, Lipinski and PAINS columns are used directly. "
             "Otherwise the app falls back to RDKit calculations. The first 1000 rows are screened.")
    up = st.file_uploader("Library CSV", type="csv", key="lib")

    f1, f2, f3 = st.columns(3)
    thr = f1.slider("Minimum AChE probability", 0.30, 0.90, 0.50, 0.05)
    max_lip = f1.number_input("Max Lipinski violations", 0, 4, 1)
    need_bbb = f2.checkbox("Require BBB permeable", True)
    no_pains = f2.checkbox("Exclude PAINS alerts", True)
    need_sim = f3.checkbox("Exclude molecules far from training data (similarity < 0.4)", True)
    need_gi = f3.checkbox("Require high GI absorption (SwissADME file only)", False)
    no_pgp = f3.checkbox("Exclude P-gp substrates (SwissADME file only)", False)

    if up is not None:
        df = pd.read_csv(up)
        scol = "smiles" if "smiles" in df.columns else next(
            (c for c in df.columns if "smiles" in str(c).lower()), None)
        if scol is None:
            st.error("The CSV needs a column containing 'smiles' in its name.")
        else:
            df = df.head(1000).reset_index(drop=True)
            name_c = find_col(df, "molecule") or find_col(df, "name") or find_col(df, "id")
            bbb_c, lip_c = find_col(df, "bbb"), find_col(df, "lipinski")
            pains_c, gi_c = find_col(df, "pains"), find_col(df, "gi", "absorption")
            pgp_c = find_col(df, "pgp") or find_col(df, "p-gp")

            valid_idx, mols, fps = [], [], []
            for i, s in enumerate(df[scol].astype(str)):
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    valid_idx.append(i); mols.append(m); fps.append(gen.GetFingerprint(m))
            n_invalid = len(df) - len(valid_idx)

            if not mols:
                st.error("No valid SMILES found in this file.")
            else:
                probs = model.predict_proba(np.array([to_arr(f) for f in fps]))[:, 1]
                rows = []
                for k, (i, m, fp) in enumerate(zip(valid_idx, mols, fps)):
                    pr = props(m)
                    sim = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
                    rec = df.iloc[i]
                    rows.append({
                        "Name": str(rec[name_c]) if name_c else f"mol_{i + 1}",
                        "SMILES": str(rec[scol]),
                        "AChE_prob": round(float(probs[k]), 3),
                        "Similarity": round(float(sim), 2),
                        "BBB_pass": (str(rec[bbb_c]).strip().lower() == "yes") if bbb_c else pr["bbb"],
                        "Lipinski_viol": num(rec[lip_c], pr["lip"]) if lip_c else pr["lip"],
                        "PAINS_alerts": num(rec[pains_c], pr["pains"]) if pains_c else pr["pains"],
                        "GI_high": (str(rec[gi_c]).strip().lower() == "high") if gi_c else None,
                        "Pgp_substrate": (str(rec[pgp_c]).strip().lower() == "yes") if pgp_c else None,
                        "QED": round(pr["qed"], 2), "MW": round(pr["mw"]), "TPSA": round(pr["tpsa"]),
                    })
                out = pd.DataFrame(rows)

                stages, cur = [("Library (valid SMILES)", out)], out
                cur = cur[cur["AChE_prob"] >= thr]
                stages.append((f"Predicted active (probability ≥ {thr:.2f})", cur))
                if need_sim:
                    cur = cur[cur["Similarity"] >= 0.4]
                    stages.append(("Within the model's reliable range", cur))
                cur = cur[cur["Lipinski_viol"] <= max_lip]
                stages.append((f"Drug-like (Lipinski violations ≤ {max_lip})", cur))
                if need_bbb:
                    cur = cur[cur["BBB_pass"] == True]
                    stages.append(("BBB permeable", cur))
                if no_pains:
                    cur = cur[cur["PAINS_alerts"] == 0]
                    stages.append(("No PAINS alerts", cur))
                if need_gi and gi_c:
                    cur = cur[cur["GI_high"] == True]
                    stages.append(("High GI absorption", cur))
                if no_pgp and pgp_c:
                    cur = cur[cur["Pgp_substrate"] != True]
                    stages.append(("Not a P-gp substrate", cur))

                st.caption(f"BBB source: {'SwissADME column' if bbb_c else 'TPSA < 90 and MW < 450 rule'}  ·  "
                           f"Lipinski and PAINS: {'SwissADME columns' if (lip_c and pains_c) else 'RDKit'}  ·  "
                           f"Invalid SMILES skipped: {n_invalid}")
                st.subheader("Screening funnel")
                st.dataframe(pd.DataFrame({"Stage": [s for s, _ in stages],
                                           "Compounds": [len(d) for _, d in stages]}),
                             use_container_width=True, hide_index=True)

                short = cur.sort_values("AChE_prob", ascending=False).reset_index(drop=True)
                short.index = short.index + 1
                st.subheader(f"Ranked shortlist ({len(short)} compounds)")
                if short.empty:
                    st.warning("No compounds pass all filters. Try loosening one of them.")
                else:
                    st.dataframe(short, use_container_width=True)
                    st.download_button("Download shortlist (CSV)", short.to_csv(index_label="Rank"),
                                       "ache_shortlist.csv")
                    t = short.head(6)
                    st.image(Draw.MolsToGridImage(
                        [Chem.MolFromSmiles(s) for s in t["SMILES"]], molsPerRow=3, subImgSize=(260, 200),
                        legends=[f"{n} | p={p:.2f}" for n, p in zip(t["Name"], t["AChE_prob"])]))
                st.caption("The shortlist is for further validation (docking, molecular dynamics, experiments). "
                           "It is not proof that any compound works.")

# ---------------- About ----------------
with tab5:
    st.write("**Model:** random forest (500 trees, balanced class weights) on Morgan fingerprints "
             "(radius 2, 2048 bits).")
    st.write("**Training data:** ChEMBL bioactivity data for human acetylcholinesterase.")
    st.write(f"**Performance:** ROC-AUC {SCAF_AUC} on unseen scaffolds (see the Model performance tab).")
    st.write("**What is machine learning here:** only the AChE activity prediction. "
             "Drug-likeness, BBB and PAINS are rule-based filters, and ADMET values come from SwissADME when you upload its export.")
    st.write("**Quick BBB rule:** TPSA < 90 and molecular weight < 450.")
    st.write("**Reliability:** if similarity to the training set is below 0.4, the prediction is flagged as unreliable.")
    st.info("Limits: activity is predicted from structure alone. The app does not model docking, toxicity or real "
            "blood-brain barrier transport. Treat results as a way to rank molecules, not as proof.")
