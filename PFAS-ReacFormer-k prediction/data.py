"""
Data utilities for k prediction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.ML.Descriptors import MoleculeDescriptors
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


sel_descs = ['FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
             'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
             'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
             'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
             'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
             'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert',
             'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
             'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
             'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
             'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo',
             'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine',
             'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
             'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss',
             'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom',
             'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
             'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
             'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone',
             'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane',
             'fr_urea']
desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(sel_descs)
RDLogger.DisableLog('rdApp.*')


def get_fp(mol, radius=2, nBits=2048, useChirality=True, fp_type='morgan'):
    if fp_type.lower() == 'morgan':
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits,
                                                            useChirality=useChirality, )
    elif fp_type.lower() == 'atompair':
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits, )
    elif fp_type.lower() == 'toptorsion':
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits, )
    elif fp_type.lower() == 'rdfp':
        fp = Chem.RDKFingerprint(mol, fpSize=nBits, )

    return np.array(list(map(eval, list(fp.ToBitString()))))


def get_rdkit_desc(mol, **kwargs):
    return np.array(list(desc_calc.CalcDescriptors(mol)))


def ext_feat_gen(mol, params={'radius': 2, 'nBits': 2048, 'useChirality': True}, desc_type='Morgan',
                 multi_readout='mean'):
    if '++' in desc_type:
        desc_type_lst = desc_type.split('++')
        desc_lst = []
        for desc_type in desc_type_lst:
            desc_lst.append(get_fp(mol, fp_type=desc_type.lower(), **params))
        if multi_readout.lower() == 'mean':
            return np.mean(desc_lst, axis=0)
        elif multi_readout.lower() == 'concat':
            return np.concatenate(desc_lst, axis=0)
        else:
            raise ValueError('Invalid multi_readout type, please choose from "mean", "concat"')
    else:
        if desc_type.lower() in ['morgan', 'atompair', 'toptorsion', 'rdfp']:
            return get_fp(mol, fp_type=desc_type, **params)
        elif desc_type.lower() == 'rdkit':
            return get_rdkit_desc(mol, **params)
        else:
            raise ValueError(
                'Invalid descriptor type, please choose from "Morgan", "RDKit", "AtomPair", "TopTorsion", "RDFP"')


RDLogger.DisableLog('rdApp.*')

NUM_ATOM_TYPE = 65
NUM_DEGRESS_TYPE = 11
NUM_FORMCHRG_TYPE = 5
NUM_HYBRIDTYPE = 6
NUM_CHIRAL_TYPE = 3
NUM_AROMATIC_NUM = 2
NUM_VALENCE_TYPE = 7
NUM_Hs_TYPE = 5
NUM_RS_TPYE = 3

NUM_BOND_TYPE = 6
NUM_BOND_DIRECTION = 3
NUM_BOND_STEREO = 3
NUM_BOND_INRING = 2
NUM_BOND_ISCONJ = 2
ATOM_FEAT_DIMS = [NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,
                    NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE]
BOND_FEAT_DIME = [NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ]
ATOM_LST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LST)}
MAX_NEIGHBORS = 10
CHIRAL_TAG_LST = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                  Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG_LST)}
HYBRIDTYPE_LST = [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D,Chem.rdchem.HybridizationType.SP3D2,Chem.rdchem.HybridizationType.UNSPECIFIED]
HYBRIDTYPE_DICT = {hb: i for i, hb in enumerate(HYBRIDTYPE_LST)}
VALENCE_LST = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE_LST)}
NUM_Hs_LST = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs_LST)}
BOND_TYPE_LST = [Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC,
                 Chem.rdchem.BondType.DATIVE,
                 Chem.rdchem.BondType.UNSPECIFIED]
BOND_DIR_LST = [
                Chem.rdchem.BondDir.NONE,
                Chem.rdchem.BondDir.ENDUPRIGHT,
                Chem.rdchem.BondDir.ENDDOWNRIGHT]
BOND_STEREO_LST = [Chem.rdchem.BondStereo.STEREONONE,
                   Chem.rdchem.BondStereo.STEREOE,
                   Chem.rdchem.BondStereo.STEREOZ,
                   ]
FORMAL_CHARGE_LST = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE_LST)}
RS_TAG_LST = ["R","S","None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG_LST)}


HALOGENS = {9, 17, 35, 53, 85}


def tpsa_atomic_contribs(mol: Chem.Mol):
    try:
        contribs = rdMD._CalcTPSAContribs(mol)


        if isinstance(contribs, (list, tuple)) and len(contribs) >= 1:
            vals = contribs[0]
        else:
            vals = contribs


        if isinstance(vals, (int, float)):

            return [float(vals)] * mol.GetNumAtoms()
        else:

            return [float(v) for v in vals]

    except Exception as e:

        num_atoms = mol.GetNumAtoms()
        return [0.0] * num_atoms

def gasteiger_charges(mol: Chem.Mol):
    tmp = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(tmp)
    out = []
    for a in tmp.GetAtoms():
        q = a.GetProp('_GasteigerCharge') if a.HasProp('_GasteigerCharge') else '0.0'
        try:
            q = float(q)
        except Exception:
            q = 0.0
        if q != q:
            q = 0.0
        out.append(q)
    return out

def electronegativity_pauling(mol: Chem.Mol):
    PT = Chem.GetPeriodicTable()
    vals = []
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        try:
            v = PT.GetElectronegativity(z)
            if v is None or v != v:
                v = 0.0
        except Exception:
            v = 0.0
        vals.append(float(v))
    return vals

def adjacent_hetero_counts(mol: Chem.Mol):
    res = []
    for a in mol.GetAtoms():
        O = N = S = HAL = 0
        total = 0
        for nb in a.GetNeighbors():
            z = nb.GetAtomicNum()
            if z not in (1, 6):
                total += 1
            if z == 8: O += 1
            if z == 7: N += 1
            if z == 16: S += 1
            if z in HALOGENS: HAL += 1
        res.append([total, O, N, S, HAL])
    return res


BOND_FG_CLASSES = [
    "other",
    "amide",
    "acyl_halide",
    "carboxylic_acid",
    "ester",
    "nitrile",
    "sulfonyl",
    "aromatic",
    "alkene",
    "alkyne",
    "ether",
    "carbonyl"
]
FG2IDX = {n:i for i, n in enumerate(BOND_FG_CLASSES)}
NUM_BOND_FGCLASS = len(BOND_FG_CLASSES)


FG_SMARTS_PRIORITY = [
    ("amide",           "[#6X3](=O)[NX3]"),
    ("acyl_halide",     "[#6X3](=O)[F,Cl,Br,I]"),
    ("carboxylic_acid", "[#6X3](=O)[O;H1]"),
    ("ester",           "[#6X3](=O)O[#6]"),
    ("nitrile",         "[#6]#[#7]"),
    ("sulfonyl",        "[#16X4](=O)(=O)"),
    ("aromatic",        "a:a"),
    ("alkene",          "C=C"),
    ("alkyne",          "C#C"),
    ("ether",           "[#6]-O-[#6]"),
    ("carbonyl",        "[#6]=O"),
]

FG_QUERIES = [(name, Chem.MolFromSmarts(s)) for name, s in FG_SMARTS_PRIORITY]
for name, q in FG_QUERIES:
    if q is None:
        raise ValueError(f"SMARTS for {name} failed to compile.")

def bond_fg_labels_via_smarts(mol: Chem.Mol):
    nB = mol.GetNumBonds()
    labels = [-1] * nB

    for name, q in FG_QUERIES:
        cls_idx = FG2IDX[name]

        for match in mol.GetSubstructMatches(q, uniquify=True):

            for qb in q.GetBonds():
                qa = qb.GetBeginAtomIdx(); qb_ = qb.GetEndAtomIdx()
                a = match[qa]; b = match[qb_]
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is None:
                    continue
                bid = bond.GetIdx()
                if labels[bid] == -1:
                    labels[bid] = cls_idx


    return [FG2IDX["other"] if x == -1 else x for x in labels]


def calc_batch_graph_distance(batch, edge_index, task):

    assert task in ["forward_prediction", "retrosynthesis"], "task must be 'forward_prediction' or 'retrosynthesis'"
    num_nodes = batch.size(0)
    num_graphs = batch.max().item() + 1
    max_len = int(torch.bincount(batch).max())


    full_adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int32, device=batch.device)
    src = edge_index[0]
    dest = edge_index[1]
    full_adj_matrix[src, dest] = 1


    graph_masks = batch.unsqueeze(0) == batch.unsqueeze(1)


    adj_matrices = full_adj_matrix * graph_masks.float()

    distances = []
    for i in range(num_graphs):

        graph_mask = batch == i
        adj_matrix = adj_matrices[graph_mask][:, graph_mask]


        max_power = max_len
        identity = torch.eye(adj_matrix.size(0),
                             dtype=torch.int32, device=adj_matrix.device)
        dist = torch.full_like(adj_matrix, float('inf'))
        dist[adj_matrix > 0] = 1
        power = adj_matrix.clone()

        for _ in range(2, max_power + 1):
            power = torch.matmul(power, adj_matrix)
            new_paths = (power > 0) & (dist == float('inf'))
            dist[new_paths] = _


        dist[(dist > 8) & (dist < 15)] = 8
        dist[dist >= 15] = 9
        if task == "forward_prediction":
            dist[dist == 0] = 10
        dist.fill_diagonal_(0)


        padded_dist = torch.full((max_len, max_len), 11 if task == "forward_prediction" else 10,
                                 dtype=torch.int32, device=dist.device)
        actual_size = dist.size(0)
        padded_dist[:actual_size, :actual_size] = dist
        distances.append(padded_dist)

    distances = torch.stack(distances)
    return distances

def get_agraph(node_num, edge_index):

    a_graphs = [[] for _ in range(node_num)]
    edge_dict = {}

    for u, v, in edge_index.T:
        eid = len(edge_dict)
        edge_dict[(u, v)] = eid
        a_graphs[v].append(eid)
    for a_graph in a_graphs:
        while len(a_graph) < 11:
            a_graph.append(1e9)
    a_graphs = torch.tensor(a_graphs).long()
    return a_graphs, edge_dict

def gen_onehot(features,feature_dims):
    assert len(features) == len(feature_dims), "size of 'features' and 'feature_dims' should be same"
    onehot = []
    for feat,feat_dim in zip(features,feature_dims):
        f_oh = np.zeros(feat_dim)
        f_oh[feat] = 1
        onehot.append(f_oh)
    return np.concatenate(onehot)

def get_bgraphs(edge_index, edge_dict):

    src_tgt_lst_map = {}
    for src, tgt in edge_index.T:
        if not src in src_tgt_lst_map:
            src_tgt_lst_map[src] = [tgt]
        else:
            src_tgt_lst_map[src].append(tgt)


    b_graphs = [[] for _ in range(len(edge_index.T))]
    for u, v, in edge_index.T:
        u = int(u)
        v = int(v)
        eid = edge_dict[(u, v)]

        for w in src_tgt_lst_map[u]:
            if not w == v:
                b_graphs[eid].append(edge_dict[(w, u)])

    for b_graph in b_graphs:
        while len(b_graph) < 11:
            b_graph.append(1e9)
    b_graphs = torch.tensor(b_graphs).long()
    return b_graphs


def mol2graphinfo(mol):
    fg_labels = bond_fg_labels_via_smarts(mol)
    NUM_BOND_FGCLASS = len(BOND_FG_CLASSES)

    atom_features_list = []
    atom_oh_features_list = []
    atom_mass_list = []

    atom_feat_dims = [NUM_ATOM_TYPE, NUM_DEGRESS_TYPE, NUM_FORMCHRG_TYPE, NUM_HYBRIDTYPE, NUM_CHIRAL_TYPE,
                      NUM_AROMATIC_NUM, NUM_VALENCE_TYPE, NUM_Hs_TYPE, NUM_RS_TPYE]
    bond_feat_dims = [NUM_BOND_TYPE, NUM_BOND_DIRECTION, NUM_BOND_STEREO, NUM_BOND_INRING, NUM_BOND_ISCONJ,NUM_BOND_FGCLASS]

    for atom in mol.GetAtoms():
        atom_feature = [ATOM_DICT.get(atom.GetSymbol(), ATOM_DICT["unk"]),
                        min(atom.GetDegree(), MAX_NEIGHBORS),
                        FC_DICT.get(atom.GetFormalCharge(), 4),
                        HYBRIDTYPE_DICT.get(atom.GetHybridization(), 5),
                        CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2),
                        int(atom.GetIsAromatic()),
                        VALENCE_DICT.get(atom.GetTotalValence(), 6),
                        NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4),
                        RS_TAG_DICT.get(atom.GetPropsAsDict().get("_CIPCode", "None"), 2)]
        atom_oh_feature = gen_onehot(atom_feature, atom_feat_dims)
        atom_mass = atom.GetMass()
        atom_features_list.append(atom_feature)
        atom_oh_features_list.append(atom_oh_feature)
        atom_mass_list.append(atom_mass)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    x_oh = torch.tensor(np.array(atom_oh_features_list), dtype=torch.long)
    atom_mass = torch.from_numpy(np.array(atom_mass_list))


    Chem.AssignStereochemistry(mol, force=True)

    tpsa_vals = tpsa_atomic_contribs(mol)
    gchg_vals = gasteiger_charges(mol)
    en_vals = electronegativity_pauling(mol)
    het_feat = adjacent_hetero_counts(mol)

    x_cont_np = np.column_stack([
        np.asarray(tpsa_vals, dtype=np.float32),
        np.asarray(gchg_vals, dtype=np.float32),
        np.asarray(en_vals, dtype=np.float32),
        np.asarray(het_feat, dtype=np.float32),
    ])
    x_cont = torch.tensor(x_cont_np, dtype=torch.float)


    num_bond_features = 6
    num_oh_bond_features = sum(bond_feat_dims)
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        edge_oh_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            fg_idx = fg_labels[bond.GetIdx()]
            edge_feature = [BOND_TYPE_LST.index(bond.GetBondType()),
                            BOND_DIR_LST.index(bond.GetBondDir()),
                            BOND_STEREO_LST.index(bond.GetStereo()),
                            int(bond.IsInRing()),
                            int(bond.GetIsConjugated()),
                            fg_idx]
            edge_oh_feature = gen_onehot(edge_feature, bond_feat_dims)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            edge_oh_features_list.append(edge_oh_feature)

        edge_index = np.array(edges_list).T

        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
        edge_oh_attr = torch.tensor(np.array(edge_oh_features_list),
                                    dtype=torch.long)
    else:
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        edge_oh_attr = torch.empty((0, num_oh_bond_features), dtype=torch.long)

    a_graphs, edge_dict = get_agraph(len(x), edge_index)
    b_graphs = get_bgraphs(edge_index, edge_dict)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return x, edge_index, edge_attr, atom_mass, x_oh, edge_oh_attr, a_graphs, b_graphs,x_cont


AGG = {"mean": np.mean, "sum": np.sum, "max": np.max}


def reactants_desc_vec(
    rct_smiles: str,
    ext_feat_type: str = "rdkit",
    ext_feat_param: dict = None,
    readout: str = "mean",
) -> torch.Tensor:
    assert ext_feat_gen is not None,
    ext_feat_param = ext_feat_param or {"radius": 2, "nBits": 2048, "useChirality": True}

    parts = [s for s in str(rct_smiles).split(".") if s]
    vecs = []
    for s in parts:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue

        v = ext_feat_gen(m, params=ext_feat_param, desc_type=ext_feat_type, multi_readout=readout)

        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)

        v = np.asarray(v, dtype=np.float32).reshape(-1)
        vecs.append(v)

    if not vecs:
        dim = ext_feat_param.get("nBits", 2048)
        return torch.zeros((dim,), dtype=torch.float)

    V = AGG.get(readout, np.mean)(np.stack(vecs, 0), axis=0)
    return torch.from_numpy(np.asarray(V, dtype=np.float32)).float()


class PFASDataset(InMemoryDataset):

    def __init__(
        self,
        csv_path: str,
        root: str = ".",
        target_col: str = "k",
        target_transform: str = "log10",
        use_desc: bool = True,
        ext_feat_type: str = "rdkit",
        ext_feat_param: dict = None,
        readout: str = "mean",
        filtered_out_csv: str = None,
        transform=None,
        pre_transform=None,
    ):
        self.csv_path = csv_path
        self.target_col = target_col
        self.target_transform = target_transform
        self.use_desc = use_desc
        self.ext_feat_type = ext_feat_type
        self.ext_feat_param = ext_feat_param or {"radius": 2, "nBits": 2048, "useChirality": True}
        self.readout = readout
        self.filtered_out_csv = filtered_out_csv

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        pass

    @property
    def processed_file_names(self):
        trans_tag = self.target_transform if self.target_transform is not None else "raw"
        if self.use_desc:
            tag = f"{self.target_col}_{trans_tag}_desc_{self.ext_feat_type}_{self.readout}"
        else:
            tag = f"{self.target_col}_{trans_tag}_nodesc"

        base = Path(self.csv_path).stem
        name = base + f"_pfas_{tag}.pt"
        return [name]

    def process(self):
        df = pd.read_csv(self.csv_path, sep=",", engine="python")

        pass
        for col in df.columns:
            pass

        required_cols = ["SMILES", self.target_col]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"CSV 中缺少必要列: {c}")


        df["__original_index"] = np.arange(len(df))
        df["__target_raw"] = df[self.target_col]
        df["__target_num"] = pd.to_numeric(df[self.target_col], errors="coerce")

        filtered_records = []


        smiles_missing_mask = df["SMILES"].isna() | (df["SMILES"].astype(str).str.strip() == "") | (df["SMILES"].astype(str).str.lower() == "nan")
        for _, r in df[smiles_missing_mask].iterrows():
            filtered_records.append({
                "original_index": int(r["__original_index"]),
                "SMILES": r.get("SMILES", ""),
                "target_col": self.target_col,
                "target_raw": r.get("__target_raw", ""),
                "target_num": r.get("__target_num", np.nan),
                "reason": "SMILES 缺失或为空",
            })


        target_bad_mask = df["__target_num"].isna()
        for _, r in df[target_bad_mask & (~smiles_missing_mask)].iterrows():
            filtered_records.append({
                "original_index": int(r["__original_index"]),
                "SMILES": r.get("SMILES", ""),
                "target_col": self.target_col,
                "target_raw": r.get("__target_raw", ""),
                "target_num": r.get("__target_num", np.nan),
                "reason": f"目标列 {self.target_col} 缺失或无法转成数字",
            })


        df = df[(~smiles_missing_mask) & (~target_bad_mask)].copy().reset_index(drop=True)
        df[self.target_col] = df["__target_num"]

        pass
        pass
        pass

        candidate_cond_cols = [
            "initial photoinitiator concentration",
            "initial PFAS concentration",
            "pH",
            "temperature",
            "contact time",
            "UV",
            "HA",
            "TOC",
            "Ccl",
            "Cphosphate",
            "CNO3",
            "CHCO3",
            "CSO4",
            "CCO3",
        ]

        cond_cols = [c for c in candidate_cond_cols if c in df.columns]

        for col in cond_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if len(cond_cols) > 0:
            pass
            cond_raw = df[cond_cols].to_numpy(dtype=np.float32)
            cond_miss = (~np.isfinite(cond_raw)).astype(np.float32)
            cond_mat = np.nan_to_num(cond_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        else:
            pass
            cond_mat = None
            cond_miss = None

        if "System" in df.columns:
            sys_codes, sys_uniques = pd.factorize(df["System"])
            system_ids = torch.tensor(sys_codes, dtype=torch.long)
            pass
        else:
            system_ids = None

        data_list = []
        desc_list = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building PFASDataset"):
            d = self.row_to_data(row, system_ids[idx] if system_ids is not None else None)
            if d is None:
                filtered_records.append({
                    "original_index": int(row["__original_index"]),
                    "SMILES": row.get("SMILES", ""),
                    "target_col": self.target_col,
                    "target_raw": row.get("__target_raw", ""),
                    "target_num": row.get(self.target_col, np.nan),
                    "reason": "RDKit 无法解析 SMILES / 建图失败",
                })
                continue

            target_value = float(row[self.target_col])

            if self.target_transform == "log10":
                if target_value <= 0:
                    filtered_records.append({
                        "original_index": int(row["__original_index"]),
                        "SMILES": row.get("SMILES", ""),
                        "target_col": self.target_col,
                        "target_raw": row.get("__target_raw", ""),
                        "target_num": target_value,
                        "reason": "log10 变换要求原始目标值 > 0，当前目标值 <= 0",
                    })
                    continue
                target_value = np.log10(target_value)
            elif self.target_transform == "raw":
                pass
            else:
                raise ValueError(f"不支持的 target_transform: {self.target_transform}")

            d.y = torch.tensor([target_value], dtype=torch.float)

            d.target_name = self.target_col
            d.target_transform = self.target_transform

            if self.use_desc:
                rct_smiles = row["SMILES"]
                mol_desc = reactants_desc_vec(
                    rct_smiles,
                    ext_feat_type=self.ext_feat_type,
                    ext_feat_param=self.ext_feat_param,
                    readout=self.readout,
                )
                d.mol_desc = mol_desc.unsqueeze(0)
                desc_list.append(mol_desc)

            if cond_mat is not None:
                d.cond_vec = torch.from_numpy(cond_mat[idx])
                d.cond_miss = torch.from_numpy(cond_miss[idx])

            if system_ids is not None:
                d.system_id = system_ids[idx]

            data_list.append(d)


        if self.filtered_out_csv is not None:
            filtered_path = Path(self.filtered_out_csv)
        else:
            filtered_path = Path(self.root) / f"{Path(self.csv_path).stem}_{self.target_col}_{self.target_transform}_filtered_out.csv"
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(filtered_records).to_csv(filtered_path, index=False, encoding="utf-8-sig")
        pass
        pass

        if len(data_list) == 0:
            raise ValueError("没有任何有效样本，请查看 filtered_out.csv 中的过滤原因。")

        data, slices = self.collate(data_list)

        if self.use_desc and len(desc_list) > 0:
            desc_tensor = torch.stack(desc_list)
            desc_out = f"{Path(self.csv_path).stem}_{self.target_col}_{self.target_transform}_{self.ext_feat_type}_desc_tensor.pt"
            torch.save(desc_tensor, desc_out)
            pass

        torch.save((data, slices), self.processed_paths[0])
        pass

    @staticmethod
    def row_to_data(row, system_id=None):
        smi_r = str(row["SMILES"])
        if not smi_r or smi_r.lower() == "nan":
            return None

        m = Chem.MolFromSmiles(smi_r)
        if m is None:
            pass
            return None

        x, edge_index, edge_attr, *_ = mol2graphinfo(m)

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if system_id is not None:
            d.system_id = system_id

        d.reactant_smiles = smi_r
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="包含 SMILES 和目标列的 CSV 文件")
    ap.add_argument("--root", default=".", help="PyG 数据缓存目录")
    ap.add_argument("--target_col", default="k", help="要预测的目标列名，如 k 或 deF")
    ap.add_argument("--target_transform", default="log10", choices=["raw", "log10"],
                    help="目标变换方式: raw 或 log10")
    ap.add_argument("--use_desc", action="store_true", help="是否计算分子描述符 mol_desc")
    ap.add_argument("--ext_feat_type", default="rdkit", help="描述符类型，如 rdkit / morgan")
    ap.add_argument("--readout", default="mean", help="多分子聚合方式: mean/sum/max")
    ap.add_argument("--out", default=None, help="可选：额外保存一个 (data,slices) pt 文件名")
    ap.add_argument("--filtered_out_csv", default=None, help="可选：保存被过滤样本及原因的 CSV 文件路径")
    args = ap.parse_args()

    ds = PFASDataset(
        csv_path=args.csv,
        root=args.root,
        target_col=args.target_col,
        target_transform=args.target_transform,
        use_desc=args.use_desc,
        ext_feat_type=args.ext_feat_type,
        readout=args.readout,
        filtered_out_csv=args.filtered_out_csv,
    )

    if args.out:
        torch.save((ds._data, ds.slices), args.out)
        pass

    pass
    if len(ds) > 0:
        pass

        pass
        for i in range(min(5, len(ds))):
            y_val = ds[i].y.item()
            pass


if __name__ == "__main__":
    main()
