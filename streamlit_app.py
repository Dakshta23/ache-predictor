import numpy as np, pandas as pd, streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AChE inhibitor predictor")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

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
    return model, fps

model, train_fps = load()

st.title("AChE inhibitor predictor")
st.write("Paste a molecule as a SMILES string. Model: random forest on Morgan fingerprints, "
         "trained on ChEMBL data for human acetylcholinesterase.")
smiles = st.text_input("SMILES", "COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2")

if smiles.strip():
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        st.error("Invalid SMILES. Please check the structure.")
    else:
        fp = gen.GetFingerprint(mol)
        p = model.predict_proba(to_arr(fp).reshape(1, -1))[0, 1]
        best = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
        hbd, hba = rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol)
        tpsa, qed = rdMolDescriptors.CalcTPSA(mol), QED.qed(mol)
        lip = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        bbb = "pass" if (tpsa < 90 and mw < 450) else "fail"
        st.subheader(f"Predicted probability of AChE activity: {p:.2f}")
        st.write(f"- Molecular weight {mw:.0f}, logP {logp:.1f}, donors {hbd}, acceptors {hba}")
        st.write(f"- TPSA {tpsa:.0f}, QED {qed:.2f}, Lipinski violations: {lip}")
        st.write(f"- Blood-brain barrier quick rule: {bbb}")
        st.write(f"- Similarity to nearest training molecule: {best:.2f}")
        if best < 0.4:
            st.warning("Far from the training data, so this prediction is unreliable.")
        st.caption("This is a computer prediction, not a lab result. On unseen scaffolds the model's "
                   "ROC-AUC was 0.873, so expect errors (about 1 in 6 predicted actives was wrong).")
