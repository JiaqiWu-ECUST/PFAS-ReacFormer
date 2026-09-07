import argparse
import csv
import json
import os
from typing import Dict, Iterator, List, Set, Tuple

from rdkit import Chem
from rxnmapper import BatchedMapper


def detok(text: str) -> str:
    """Remove quotes, invisible characters, and whitespace from a SMILES string."""
    text = text.strip()
    for ch in ['“', '”', '„', '‟', '«', '»', '"', "'", "\u200b", "\ufeff"]:
        text = text.replace(ch, "")
    return "".join(text.split())


def smiles_ok(smiles: str) -> bool:
    """Return True if all dot-separated SMILES fragments can be parsed by RDKit."""
    if not smiles:
        return False
    try:
        return all(Chem.MolFromSmiles(part) is not None for part in smiles.split("."))
    except Exception:
        return False


def canon(smiles: str) -> str:
    """Return RDKit-canonical SMILES, or an empty string if parsing fails."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol else ""
    except Exception:
        return ""


def read_pairs(src_path: str, tgt_path: str) -> Iterator[Tuple[int, str, str]]:
    """
    Read aligned reactant/product files line by line.

    Raises
    ------
    ValueError
        If the two files contain different numbers of lines.
    """
    with open(src_path, encoding="utf-8", errors="ignore") as fs, \
         open(tgt_path, encoding="utf-8", errors="ignore") as ft:
        idx = 0
        while True:
            src_line = fs.readline()
            tgt_line = ft.readline()

            if not src_line and not tgt_line:
                break
            if not src_line or not tgt_line:
                raise ValueError(
                    "Source and target files contain different numbers of lines."
                )

            idx += 1
            yield idx, detok(src_line), detok(tgt_line)


def is_heavy_atom(atom) -> bool:
    """Return True for non-hydrogen atoms."""
    return atom.GetAtomicNum() > 1


def smi_to_mol(smiles: str):
    """Parse a SMILES string safely."""
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def split_side(side: str) -> List[str]:
    """Split a dot-separated reaction side into molecular fragments."""
    return [x for x in side.split(".") if x] if side is not None else []


def parse_mapped_reaction(mapped_rxn: str):
    """Parse mapped reaction SMILES in reactants>reagents>products format."""
    parts = mapped_rxn.split(">")
    if len(parts) != 3:
        raise ValueError("Reaction SMILES must have the form reactants>reagents>products.")

    reactants = [smi_to_mol(s) for s in split_side(parts[0])]
    products = [smi_to_mol(s) for s in split_side(parts[2])]

    reactants = [mol for mol in reactants if mol is not None]
    products = [mol for mol in products if mol is not None]
    return reactants, products


def has_any_mapping(mol) -> bool:
    """Return True if the molecule contains at least one atom-map number."""
    return any(atom.GetAtomMapNum() > 0 for atom in mol.GetAtoms())


def count_total_atoms(mol, heavy_only: bool = True) -> int:
    """Count atoms, optionally excluding hydrogens."""
    return sum(
        1 for atom in mol.GetAtoms()
        if not heavy_only or is_heavy_atom(atom)
    )


def count_mapped_atoms(mol, heavy_only: bool = True) -> int:
    """Count mapped atoms, optionally excluding hydrogens."""
    return sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
        and (not heavy_only or is_heavy_atom(atom))
    )


def element_count(mols, heavy_only: bool = True) -> Dict[str, int]:
    """Count element occurrences across molecules."""
    counts: Dict[str, int] = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if heavy_only and not is_heavy_atom(atom):
                continue
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def bonds_by_map(mol) -> Dict[Tuple[int, int], Tuple[str, int]]:
    """Represent mapped bonds by atom-map-number pairs."""
    bonds = {}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetAtomMapNum()
        j = bond.GetEndAtom().GetAtomMapNum()

        if i > 0 and j > 0:
            if i > j:
                i, j = j, i
            bonds[(i, j)] = (
                str(bond.GetBondType()),
                int(bond.GetIsAromatic()),
            )
    return bonds


def sanitize_ok(mol) -> bool:
    """Check whether a molecule passes RDKit sanitization."""
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
        return True
    except Exception:
        return False


def qc_on_mapped_rxn(
    mapped_rxn: str,
    min_cov: float = 0.90,
    min_changes: int = 1,
    max_changes: int = 8,
    ignore_H: bool = True,
    expand_neighbors: bool = True,
) -> Dict:
    """
    Run QC on a mapped reaction and identify reaction-center atoms/bonds.

    The checks include:
    - atom-map bijection between reactants and products
    - atom-mapping coverage
    - element conservation
    - RDKit sanitization
    - number of bond changes
    """
    out = {
        "ok_bijection": False,
        "reactant_cov": None,
        "product_cov": None,
        "ok_coverage": False,
        "ok_element_conservation": False,
        "ok_sanitize_R": True,
        "ok_sanitize_P": True,
        "bond_formed": 0,
        "bond_broken": 0,
        "bond_order_changed": 0,
        "bond_changes_total": 0,
        "ok_change_count_range": False,
        "center_atoms": [],
        "center_atoms_with_neighbors": [],
        "center_bonds": [],
    }

    reactants, products = parse_mapped_reaction(mapped_rxn)
    mapped_reactants = [mol for mol in reactants if has_any_mapping(mol)]
    mapped_products = [mol for mol in products if has_any_mapping(mol)]

    if not mapped_reactants or not mapped_products:
        return out

    maps_r: Set[int] = set()
    maps_p: Set[int] = set()

    for mol in mapped_reactants:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                if map_num in maps_r:
                    return out
                maps_r.add(map_num)

    for mol in mapped_products:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                if map_num in maps_p:
                    return out
                maps_p.add(map_num)

    out["ok_bijection"] = (maps_r == maps_p) and len(maps_r) > 0

    cov_r = (
        sum(count_mapped_atoms(mol, True) for mol in mapped_reactants)
        / max(1, sum(count_total_atoms(mol, True) for mol in mapped_reactants))
    )
    cov_p = (
        sum(count_mapped_atoms(mol, True) for mol in mapped_products)
        / max(1, sum(count_total_atoms(mol, True) for mol in mapped_products))
    )

    out["reactant_cov"] = round(cov_r, 4)
    out["product_cov"] = round(cov_p, 4)
    out["ok_coverage"] = cov_r >= min_cov and cov_p >= min_cov

    elem_r = element_count(mapped_reactants, heavy_only=ignore_H)
    elem_p = element_count(mapped_products, heavy_only=ignore_H)
    out["ok_element_conservation"] = elem_r == elem_p

    out["ok_sanitize_R"] = all(sanitize_ok(mol) for mol in mapped_reactants)
    out["ok_sanitize_P"] = all(sanitize_ok(mol) for mol in mapped_products)

    bonds_r = {}
    bonds_p = {}

    for mol in mapped_reactants:
        bonds_r.update(bonds_by_map(mol))
    for mol in mapped_products:
        bonds_p.update(bonds_by_map(mol))

    formed = 0
    broken = 0
    changed = 0
    center_bonds = []

    for key in set(bonds_r) | set(bonds_p):
        if key not in bonds_r:
            formed += 1
            center_bonds.append(key)
        elif key not in bonds_p:
            broken += 1
            center_bonds.append(key)
        elif bonds_r[key] != bonds_p[key]:
            changed += 1
            center_bonds.append(key)

    total_changes = formed + broken + changed

    out["bond_formed"] = formed
    out["bond_broken"] = broken
    out["bond_order_changed"] = changed
    out["bond_changes_total"] = total_changes
    out["ok_change_count_range"] = min_changes <= total_changes <= max_changes

    center_atoms = sorted(
        {i for i, _ in center_bonds}
        | {j for _, j in center_bonds}
    )

    out["center_bonds"] = [f"{i}-{j}" for i, j in center_bonds]
    out["center_atoms"] = center_atoms

    if expand_neighbors and center_atoms:
        map_to_atom = {}

        for mol in mapped_reactants:
            for atom in mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0 and map_num not in map_to_atom:
                    map_to_atom[map_num] = (mol, atom.GetIdx())

        expanded = set(center_atoms)

        for map_num in center_atoms:
            if map_num not in map_to_atom:
                continue

            mol, atom_idx = map_to_atom[map_num]
            for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbor_map = neighbor.GetAtomMapNum()
                if neighbor_map > 0:
                    expanded.add(neighbor_map)

        out["center_atoms_with_neighbors"] = sorted(expanded)
    else:
        out["center_atoms_with_neighbors"] = center_atoms

    return out


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of an output path if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Clean, atom-map, quality-control, and annotate reaction data."
    )
    parser.add_argument("--src", default="src-train.txt", help="Reactant SMILES file.")
    parser.add_argument("--tgt", default="tgt-train.txt", help="Product SMILES file.")
    parser.add_argument("--out_clean", default="output/clean_dataset.tsv")
    parser.add_argument("--out_rejects", default="output/rejects.tsv")
    parser.add_argument("--out_qc", default="output/qc_report.tsv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument(
        "--canon",
        action="store_true",
        help="Canonicalize reactant and product SMILES before atom mapping.",
    )
    parser.add_argument(
        "--conf_th",
        type=float,
        default=0.0,
        help="Reject RXNMapper results below this confidence threshold.",
    )
    parser.add_argument("--min_cov", type=float, default=0.99)
    parser.add_argument("--min_changes", type=int, default=1)
    parser.add_argument("--max_changes", type=int, default=8)
    parser.add_argument("--no_neighbor_expand", action="store_true")
    args = parser.parse_args()

    for path in (args.out_clean, args.out_rejects, args.out_qc):
        ensure_parent_dir(path)

    mapper = BatchedMapper(batch_size=args.batch_size)
    seen_rxn: Set[str] = set()
    buffer: List[Tuple[int, str, str, str]] = []

    with open(args.out_clean, "w", newline="", encoding="utf-8") as fclean, \
         open(args.out_rejects, "w", encoding="utf-8") as frej, \
         open(args.out_qc, "w", newline="", encoding="utf-8") as fqc:

        clean_writer = csv.writer(fclean, delimiter="\t")
        qc_writer = csv.writer(fqc, delimiter="\t")

        clean_writer.writerow([
            "reactants",
            "products",
            "rxn_std",
            "mapped_rxn",
            "confidence",
            "reactant_cov",
            "product_cov",
            "bond_formed",
            "bond_broken",
            "bond_order_changed",
            "bond_changes_total",
            "center_atoms",
            "center_atoms_with_neighbors",
            "center_bonds",
        ])

        frej.write(
            "idx\treactants\tproducts\treason\tconfidence\tmapped_rxn\n"
        )

        qc_writer.writerow([
            "rxn_std",
            "mapped_rxn",
            "confidence",
            "ok_bijection",
            "reactant_cov",
            "product_cov",
            "ok_coverage",
            "ok_element_conservation",
            "ok_sanitize_R",
            "ok_sanitize_P",
            "bond_formed",
            "bond_broken",
            "bond_order_changed",
            "bond_changes_total",
            "ok_change_count_range",
        ])

        def handle_one(
            idx: int,
            reactants: str,
            products: str,
            rxn_std: str,
            result: dict,
        ) -> None:
            mapped = result.get("mapped_rxn", "")
            confidence = float(result.get("confidence", 0.0) or 0.0)
            error = result.get("error", "")

            if error:
                frej.write(
                    f"{idx}\t{reactants}\t{products}\t"
                    f"mapping_error:{error}\t{confidence}\t{mapped}\n"
                )
                return

            if args.conf_th and confidence < args.conf_th:
                frej.write(
                    f"{idx}\t{reactants}\t{products}\t"
                    f"low_conf<{args.conf_th}\t{confidence}\t{mapped}\n"
                )
                return

            qc = qc_on_mapped_rxn(
                mapped,
                min_cov=args.min_cov,
                min_changes=args.min_changes,
                max_changes=args.max_changes,
                ignore_H=True,
                expand_neighbors=not args.no_neighbor_expand,
            )

            qc_writer.writerow([
                rxn_std,
                mapped,
                confidence,
                qc["ok_bijection"],
                qc["reactant_cov"],
                qc["product_cov"],
                qc["ok_coverage"],
                qc["ok_element_conservation"],
                qc["ok_sanitize_R"],
                qc["ok_sanitize_P"],
                qc["bond_formed"],
                qc["bond_broken"],
                qc["bond_order_changed"],
                qc["bond_changes_total"],
                qc["ok_change_count_range"],
            ])

            passed = (
                qc["ok_bijection"]
                and qc["ok_coverage"]
                and qc["ok_change_count_range"]
                and qc["ok_sanitize_R"]
                and qc["ok_sanitize_P"]
                and qc["ok_element_conservation"]
            )

            if not passed:
                reasons = []
                if not qc["ok_bijection"]:
                    reasons.append("non_bijection")
                if not qc["ok_coverage"]:
                    reasons.append("low_coverage")
                if not qc["ok_change_count_range"]:
                    reasons.append("abnormal_change_count")
                if not qc["ok_sanitize_R"] or not qc["ok_sanitize_P"]:
                    reasons.append("sanitize_fail")
                if not qc["ok_element_conservation"]:
                    reasons.append("element_nonconservation")

                frej.write(
                    f"{idx}\t{reactants}\t{products}\t"
                    f"{'|'.join(reasons)}\t{confidence}\t{mapped}\n"
                )
                return

            clean_writer.writerow([
                reactants,
                products,
                rxn_std,
                mapped,
                confidence,
                qc["reactant_cov"],
                qc["product_cov"],
                qc["bond_formed"],
                qc["bond_broken"],
                qc["bond_order_changed"],
                qc["bond_changes_total"],
                json.dumps(qc["center_atoms"], ensure_ascii=False),
                json.dumps(qc["center_atoms_with_neighbors"], ensure_ascii=False),
                json.dumps(qc["center_bonds"], ensure_ascii=False),
            ])

        def flush() -> None:
            nonlocal buffer

            if not buffer:
                return

            reactions = [rxn for _, _, _, rxn in buffer]

            try:
                results = list(mapper.map_reactions_with_info(reactions))
            except Exception:
                for idx, reactants, products, rxn_std in buffer:
                    try:
                        result = list(
                            mapper.map_reactions_with_info([rxn_std])
                        )[0]
                    except Exception as exc:
                        frej.write(
                            f"{idx}\t{reactants}\t{products}\t"
                            f"mapping_exception:{exc}\t\t\n"
                        )
                        continue

                    handle_one(
                        idx,
                        reactants,
                        products,
                        rxn_std,
                        result,
                    )

                buffer.clear()
                fclean.flush()
                frej.flush()
                fqc.flush()
                return

            for item, result in zip(buffer, results):
                idx, reactants, products, rxn_std = item
                handle_one(
                    idx,
                    reactants,
                    products,
                    rxn_std,
                    result,
                )

            buffer.clear()
            fclean.flush()
            frej.flush()
            fqc.flush()

        for idx, src_smiles, tgt_smiles in read_pairs(args.src, args.tgt):
            src_valid = smiles_ok(src_smiles)
            tgt_valid = smiles_ok(tgt_smiles)

            if not (src_valid and tgt_valid):
                reasons = []
                if not src_valid:
                    reasons.append("bad_src")
                if not tgt_valid:
                    reasons.append("bad_tgt")

                frej.write(
                    f"{idx}\t{src_smiles}\t{tgt_smiles}\t"
                    f"{','.join(reasons)}\t\t\n"
                )
                continue

            reactants = canon(src_smiles) if args.canon else src_smiles
            products = canon(tgt_smiles) if args.canon else tgt_smiles
            rxn_std = f"{reactants}>>{products}"

            if rxn_std in seen_rxn:
                frej.write(
                    f"{idx}\t{reactants}\t{products}\t"
                    "duplicate_reaction\t\t\n"
                )
                continue

            seen_rxn.add(rxn_std)
            buffer.append((idx, reactants, products, rxn_std))

            if len(buffer) >= args.chunk_size:
                flush()

        flush()

    print(f"[DONE] clean -> {os.path.abspath(args.out_clean)}")
    print(f"[DONE] rejects -> {os.path.abspath(args.out_rejects)}")
    print(f"[DONE] qc -> {os.path.abspath(args.out_qc)}")


if __name__ == "__main__":
    main()
