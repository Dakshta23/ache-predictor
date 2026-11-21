import numpy as np, pandas as pd, streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, Draw, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AChE inhibitor predictor", page_icon="🧠", layout="wide")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

st.markdown("""
<style>
.hero {background: linear-gradient(90deg,#6C4DF6,#9B7BFF); padding: 1.4rem 1.8rem;
       border-radius: 14px; color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {margin: .3rem 0 0 0; opacity: .92;}
</style>
<div class="hero"><h1>🧠 AChE inhibitor predictor</h1>
<p>Prioritizing brain-penetrant acetylcholinesterase inhibitors with machine learning</p></div>
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

model, train_fps, train = load()

def props(mol):
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol)
    tpsa, qed = rdMolDescriptors.CalcTPSA(mol), QED.qed(mol)
    lip = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    return dict(mw=mw, logp=logp, hbd=hbd, hba=hba, tpsa=tpsa, qed=qed, lip=lip,
                bbb=(tpsa < 90 and mw < 450))

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

tab1, tab2, tab3 = st.tabs(["🔬 Single molecule", "📋 Batch screening", "ℹ️ About the model"])

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
            st.write(f"**Similarity to nearest training molecule:** {best:.2f}")

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
        legends = [f"{'Active' if train['label'].iloc[i] == 1 else 'Inactive'}  |  similarity {sims[i]:.2f}"
                   for i in top]
        st.image(Draw.MolsToGridImage(nbrs, molsPerRow=3, subImgSize=(260, 200), legends=legends))

        st.caption("This is a computer prediction, not a lab result. On unseen scaffolds the model's "
                   "ROC-AUC was 0.873, so expect errors (about 1 in 6 predicted actives was wrong).")

with tab2:
    st.write("Upload a CSV with a column named **smiles**. You get a ranked table back (first 500 rows).")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "smiles" not in df.columns:
            st.error("The CSV needs a column named 'smiles'.")
        else:
            rows = []
            for s in df["smiles"].astype(str).head(500):
                m = Chem.MolFromSmiles(s)
                if m is None:
                    rows.append({"smiles": s, "probability": None, "note": "invalid SMILES"})
                    continue
                p, sims = predict(m)
                pr = props(m)
                v, _ = verdict(p, sims.max(), pr["bbb"])
                rows.append({"smiles": s, "probability": round(p, 3), "brain_rule": "pass" if pr["bbb"] else "fail",
                             "MW": round(pr["mw"]), "logP": round(pr["logp"], 1),
                             "TPSA": round(pr["tpsa"]), "QED": round(pr["qed"], 2),
                             "nearest_similarity": round(float(sims.max()), 2), "verdict": v})
            out = pd.DataFrame(rows).sort_values("probability", ascending=False)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False), "ache_predictions.csv")

with tab3:
    st.write("**Model:** random forest (500 trees, balanced class weights) on Morgan fingerprints "
             "(radius 2, 2048 bits).")
    st.write("**Training data:** ChEMBL bioactivity data for human acetylcholinesterase.")
    st.write("**Performance:** ROC-AUC 0.873 on unseen scaffolds.")
    st.write("**Brain-penetration rule:** TPSA < 90 and molecular weight < 450.")
    st.write("**Reliability:** if a molecule's similarity to the training set is below 0.4, "
             "the prediction is flagged as unreliable.")
    st.info("Limits: this predicts activity from structure alone. It does not model docking, toxicity "
            "or real blood-brain barrier transport. Treat results as a way to rank molecules, not as proof.")
