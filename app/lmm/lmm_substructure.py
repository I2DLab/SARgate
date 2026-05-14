"""
==========================
lmm_substructure.py
==========================

Substructure detection and analysis.

Performs substructure searches to identify scaffolds or molecular cores shared
across the dataset. Provides functions to extract Bemis-Murcko scaffolds,
calculate Bemis-Murcko scaffold's minimal substructures, performs substructure clustering and subsequent MCS calculation.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Scaffold analysis
# 3. Calculate minimal substructures

import os
import csv
import numpy as np
from collections import Counter
from typing import Any
from scipy.cluster.hierarchy import linkage, fcluster
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from app.lmm.lmm_gui import update_scaffold_analysis_status
from app.lmm.lmm_abort import confirm_cancellation
from app.utils.callbacks import append_to_log


# -----------------------------------------------------------------------------
# 1.1. Mol smiles cache
# -----------------------------------------------------------------------------
def _mol_smiles_cached(mol: Any, cache: dict[int, str]) -> str:
    """
    Return canonical isomeric SMILES with a per-run cache.
    """
    key = id(mol)
    if key not in cache:
        cache[key] = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return cache[key]


# -----------------------------------------------------------------------------
# 1.2. Murcko cache
# -----------------------------------------------------------------------------
def _murcko_scaffold_cached(mol: Any, cache: dict[int, Any]) -> Any:
    """
    Return Murcko scaffold with a per-run cache.
    """
    key = id(mol)
    if key not in cache:
        cache[key] = MurckoScaffold.GetScaffoldForMol(mol)
    return cache[key]


# -----------------------------------------------------------------------------
# 2. Scaffold analysis
# -----------------------------------------------------------------------------
def scaffold_analysis(state: dict[str, Any]) -> Any:
    """
    Perform scaffold-based analysis and clustering of the molecular dataset.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    checkbox_states = state["checkbox_states"]               
    selected_file = state["selected_file_name"]            
    subset_dir = state["subset_dir"]                        
    report_dir = state["report_dir"]                        
    suppl = state["supplier"]                                
    collecting_method = checkbox_states.get("Subsets collection method", "")
    mcs_timeout_raw = str(checkbox_states.get("MCS timeout", "60s") or "60s").strip()
    if mcs_timeout_raw.lower() == "unlimited":
        mcs_timeout_seconds = None
    else:
        try:
            mcs_timeout_seconds = max(1, int(mcs_timeout_raw.rstrip("sS ").strip()))
        except Exception:
            mcs_timeout_seconds = 60
    mol_smiles_cache: dict[int, str] = {}
    murcko_cache: dict[int, Any] = {}

    if collecting_method == "User-defined scaffold" and checkbox_states.get("Scaffold SMILES"):
        update_scaffold_analysis_status("FILTERING LIBRARY BY USER-DEFINED SCAFFOLD", state, step_id=True)
        append_to_log(state, f"Filtering library by user-defined scaffold: {checkbox_states['Scaffold SMILES']}.")

        # Try multiple formats (SMILES, SMARTS, InChI, MolBlock) until one parses successfully.
        scaffold_input = checkbox_states["Scaffold SMILES"]
        append_to_log(state, f"User-defined scaffold: {scaffold_input}")

        if scaffold_input.strip() == "":
            append_to_log(state, "❌ User-defined scaffold is empty.")
            confirm_cancellation(state)
            return

        parsing_attempts = [
            ("SMILES", Chem.MolFromSmiles),
            ("SMARTS", Chem.MolFromSmarts),
            ("InChI", Chem.MolFromInchi),
            ("MolBlock", lambda s: Chem.MolFromMolBlock(s, sanitize=False))
        ]

        scaffold_mol = None
        for fmt, func in parsing_attempts:
            try:
                mol = func(scaffold_input)
                if mol is not None:
                    scaffold_mol = mol
                    append_to_log(state, f"✅ Scaffold parsed as {fmt}.")
                    break
            except Exception:
                continue

        if scaffold_mol is None:
            append_to_log(state, "❌ User-defined scaffold is not valid.")
            update_scaffold_analysis_status("INVALID SCAFFOLD SMILES", state, separator=True)
            confirm_cancellation(state)
            return

        Chem.SanitizeMol(scaffold_mol)                        
        scaffold_smiles = Chem.MolToSmiles(scaffold_mol, isomericSmiles=True, canonical=True)

        # Keep only molecules that contain the scaffold (matching with chirality).
        filtered_mols = []
        for mol in suppl:
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            
            if mol is not None and mol.HasSubstructMatch(scaffold_mol, useChirality=True):
                filtered_mols.append(mol)

        # No matches found → abort with message
        if not filtered_mols:
            append_to_log(state, f"⚠️ No molecules matched the user-defined scaffold.")
            update_scaffold_analysis_status("NO MOLECULES MATCHED THE SCAFFOLD", state, separator=True)
            state["scaffold_id"] = 0
            state["abort_analysis"] = True
            return
        else:
            scaffold_dict = {scaffold_smiles: filtered_mols}
            append_to_log(state, f"{len(filtered_mols)} molecules matched the user-defined scaffold.")
            update_scaffold_analysis_status(f"{len(filtered_mols)} MOLECULES MATCHED", state, separator=True)


    elif collecting_method in ["User-defined generalized scaffold", "User-defined generalized scaffold + similarity"] and checkbox_states.get("Generalized Scaffold SMILES"):

        update_scaffold_analysis_status("FILTERING LIBRARY BY USER-DEFINED GENERALIZED SCAFFOLD", state, step_id=True)
        append_to_log(state, f"Filtering library by user-defined generalized scaffold: {checkbox_states['Generalized Scaffold SMILES']}.")

        scaffold_input = checkbox_states["Generalized Scaffold SMILES"]

        if scaffold_input.strip() == "":
            append_to_log(state, "❌ User-defined generalized scaffold is empty.")
            confirm_cancellation(state)
            return

        parsing_attempts = [
            ("SMILES", Chem.MolFromSmiles),
            ("SMARTS", Chem.MolFromSmarts),
            ("InChI", Chem.MolFromInchi),
            ("MolBlock", lambda s: Chem.MolFromMolBlock(s, sanitize=False)),
        ]

        query_mol = None
        for fmt, func in parsing_attempts:
            try:
                mol = func(scaffold_input)
                if mol is not None:
                    query_mol = mol
                    append_to_log(state, f"✅ Generalized scaffold parsed as {fmt}.")
                    break
            except Exception:
                continue

        if query_mol is None:
            append_to_log(state, "❌ User-defined generalized scaffold is not valid.")
            update_scaffold_analysis_status("INVALID GENERALIZED SCAFFOLD", state, separator=True)
            confirm_cancellation(state)
            return

        try:
            Chem.SanitizeMol(query_mol)
        except Exception:
            try:
                Chem.SanitizeMol(query_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
            except Exception:
                append_to_log(state, "❌ Generalized scaffold sanitization failed.")
                update_scaffold_analysis_status("INVALID GENERALIZED SCAFFOLD", state, separator=True)
                confirm_cancellation(state)
                return

        def generalize_connectivity_from_core(core: Any) -> Any:
            """
            Execute the generalize connectivity from core routine.
            
            Args:
                core (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            try:
                if core is None:
                    return None
                rw = Chem.RWMol(core)
                for a in rw.GetAtoms():
                    a.SetAtomicNum(6)
                return rw.GetMol()
            except Exception as e:
                return None


        def topo_key(gen_mol: Any) -> Any:
            """
            Build a canonical topology-only key from a 'generalized' mol:.
            
            Args:
                gen_mol (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """

            if gen_mol is None:
                return None
            
            rw = Chem.RWMol(gen_mol)

            for a in rw.GetAtoms():
                a.SetIsAromatic(False)
                a.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                a.SetFormalCharge(0)

            for b in rw.GetBonds():
                b.SetIsAromatic(False)
                b.SetBondType(Chem.rdchem.BondType.SINGLE)

            m = rw.GetMol()

            Chem.SanitizeMol(
                m,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            )
            return m


        # Query Murcko
        query_scaf = MurckoScaffold.GetScaffoldForMol(query_mol)
        if query_scaf is None or query_scaf.GetNumAtoms() == 0:
            query_scaf = query_mol

        query_gen = generalize_connectivity_from_core(query_scaf)
        if query_gen is None:
            append_to_log(state, "❌ Generalized scaffold could not be generalized.")
            update_scaffold_analysis_status("INVALID GENERALIZED SCAFFOLD", state, separator=True)
            confirm_cancellation(state)
            return

        query_gen_key = topo_key(query_gen)


        sim_mode = (collecting_method == "User-defined generalized scaffold + similarity")
        sim_thr = None
        query_fp = None
        similarity_fp_gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)

        if sim_mode:
            try:
                sim_thr = float(checkbox_states.get("Scaffold Similarity threshold", 0)) / 100.0
            except Exception:
                sim_thr = 0.0
            query_fp = similarity_fp_gen.GetFingerprint(query_gen)

        matching_scaffolds = []
        seen = set()
        matched = 0
        filtered_suppl = []

        for mol in suppl:
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            if mol is None:
                continue

            murcko = _murcko_scaffold_cached(mol, murcko_cache)
            if murcko is None or murcko.GetNumAtoms() == 0:
                continue

            murcko_gen = generalize_connectivity_from_core(murcko)
            murcko_gen_key = topo_key(murcko_gen)
            if murcko_gen is None:
                continue

            # Check for match against the user-defined generalized scaffold using substructure search
            # and, if enabled, Tanimoto similarity on the generalized Murcko scaffold.
            if collecting_method == "User-defined generalized scaffold":
                match = murcko_gen_key.HasSubstructMatch(query_gen_key, useChirality=False)

            elif collecting_method == "User-defined generalized scaffold + similarity":
                match = murcko_gen_key.HasSubstructMatch(query_gen_key, useChirality=False)

                if (not match) and sim_mode:
                    fp = similarity_fp_gen.GetFingerprint(murcko_gen_key)
                    sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                    match = (sim >= sim_thr)


            if (not match) and sim_mode:
                fp = similarity_fp_gen.GetFingerprint(murcko_gen)
                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                match = (sim >= sim_thr)

            if match:
                matched += 1
                filtered_suppl.append(mol)

                # collect ORIGINAL Murcko scaffold as Mol object (unique by canonical smiles)
                smi = _mol_smiles_cached(murcko, mol_smiles_cache)
                if smi not in seen:
                    seen.add(smi)
                    matching_scaffolds.append(smi)
            
            suppl = filtered_suppl


        if not matching_scaffolds:
            append_to_log(state, "⚠️ No molecules matched the user-defined generalized scaffold (contain/similarity).")
            update_scaffold_analysis_status("No molecules matched the user-defined generalized scaffold.", state, separator=True)
            state["scaffold_id"] = 0
            state["abort_analysis"] = True
            return
        

        append_to_log(state, f"{matched} molecules matched the generalized scaffold filter.")
        append_to_log(state, f"{len(matching_scaffolds)} unique scaffolds extracted from {matched} molecules.")
        update_scaffold_analysis_status(f"   {matched} molecules matched the user-defined\n generalized scaffold", state)
        update_scaffold_analysis_status(f"   {len(matching_scaffolds)} unique scaffolds\n   extracted from {matched} molecules", state, separator=True)

        minimal_substructures = calculate_minimal_substructures(matching_scaffolds, state)
        

    elif collecting_method in ["Unique Bemis-Murcko scaffolds (BMS)", "BMS minimal substructures (MinBMS)", "MinBMS clustering (MSC)"]:
        # Generate Bemis–Murcko scaffolds, track unique SMILES and their frequencies, and remove linear molecules.
        update_scaffold_analysis_status("EXTRACTING UNIQUE MURCKO SCAFFOLDS", state, step_id=True)
        murcko_smiles = set()
        scaffold_freqs = Counter()
        cleaned_suppl = []

        # Extract unique Murcko scaffolds from dataset
        for mol in suppl:
            if mol is not None:
                murcko_scaffold = _murcko_scaffold_cached(mol, murcko_cache)
                if murcko_scaffold is not None and murcko_scaffold.GetNumAtoms() > 0:
                    murcko_smi = _mol_smiles_cached(murcko_scaffold, mol_smiles_cache)
                    murcko_smiles.add(murcko_smi)
                    scaffold_freqs[murcko_smi] += 1
                    cleaned_suppl.append(mol)  # Keep only valid scaffold-containing molecules

        removed_count = len(suppl) - len(cleaned_suppl)
        suppl = cleaned_suppl  # Remove linear molecules with no Murcko scaffold

        if removed_count > 0:
            append_to_log(state, f"{removed_count} molecules removed (no Murcko scaffold found)")  
            update_scaffold_analysis_status(f"   {removed_count} molecules removed (no Murcko scaffold found)\n", state)
        append_to_log(state, f"{len(murcko_smiles)} unique scaffolds extracted from {len(suppl)} molecules")
        update_scaffold_analysis_status(f"   {len(murcko_smiles)} unique scaffolds\n   extracted from {len(suppl)} molecules", state)

        # --- STEP 1.1: COMPUTE SCAFFOLD DOMINANCE INDEX (DI) ---
        # Combine richness and dominance signals to quantify scaffold concentration and diversity.
        def compute_scaffold_dominance_index(
            scaffold_freqs: Any,
            state: dict[str, Any],
            w: Any,
            print_top: int = 10
        ) -> Any:
            """
            Compute the scaffold Dominance Index (DI) combining richness and dominance:
                DI = w * D + (1 - w) * R
            where:
                - M = total molecules (after deduplication)
                - S = number of unique scaffolds
                - n_i = count of molecules for scaffold i
                - p_i = n_i / M
                - R = 1 - S/M                                  (richness deficit; higher when fewer scaffolds)
                - D = (sum_i p_i^2 - 1/S) / (1 - 1/S), S > 1   (Simpson/Herfindahl normalized on S: 0 uniform, 1 mono-dominant)

            Args:
                scaffold_freqs (collections.Counter): counts per scaffold.
                state: logging/state handle used by append_to_log/update_scaffold_analysis_status.
                w (float): weight for the dominance term D (0..1). Typical 0.2–0.4.
                print_top (int): number of top scaffolds to display.

            Returns:
                tuple: (DI, R, D) with DI ∈ [0, 1].
            """
            # --- Basic counts
            counts = [c for c in scaffold_freqs.values() if c > 0]
            M = sum(counts)
            S = len(counts)

            # --- Edge cases
            if M <= 1 or S == 0:
                DI = 0.0
                return DI, float("0"), float("1")
            if S == 1:
                DI = 1.0
                return DI, float("1"), float("1")

            # --- Optional print/log of scaffold frequencies (sorted)
            sorted_freq = scaffold_freqs.most_common()
            append_to_log(state, f"Scaffold frequencies (first {print_top}, sorted by frequency):")
            for i, (scf, cnt) in enumerate(sorted_freq[:print_top], 1):
                line = f"     Scaffold {i} : {cnt}"
                append_to_log(state, line)

            # --- Components
            R = 1.0 - (S / M)                                  # Richness
            p2_sum = sum((c / M) ** 2 for c in counts)         # Simpson/Herfindahl
            D = (p2_sum - (1.0 / S)) / (1.0 - (1.0 / S))       # Normalised dominance for S > 1

            # --- Final index
            DI = w * D + (1.0 - w) * R
            DI = max(0.0, min(1.0, DI))                        # Clamp to [0, 1]

            # --- Log details
            append_to_log(state, "Calculating Dominance Index (DI) of scaffold distribution:")
            append_to_log(state, f"     M () = {M}")
            append_to_log(state, f"     S = {S}")
            append_to_log(state, f"     R = {R}")
            append_to_log(state, f"     D = {D}")
            append_to_log(state, f"     w = {w}")
            append_to_log(state, f"     Dominance Index, DI = {DI:.2f}")

            return DI, R, D

        DI, R, D = compute_scaffold_dominance_index(scaffold_freqs, state, w=0.2)
        update_scaffold_analysis_status(f"      Richness (R) = {R:.2f}", state, separator=False)
        update_scaffold_analysis_status(f"      Dominance (D) = {D:.2f}", state, separator=False)
        update_scaffold_analysis_status(f"   Dominance Index, DI = {DI:.2f}", state, separator=True)

        # Reduce to minimal substructures via iterative substructure filtering (if requested).
        if collecting_method in ["BMS minimal substructures (MinBMS)", "MinBMS clustering (MSC)"]:
            minimal_substructures = calculate_minimal_substructures(murcko_smiles, state)


        # For plain Murcko analysis, treat Murcko scaffolds as the working substructures.
        elif collecting_method == "Unique Bemis-Murcko scaffolds (BMS)":
            minimal_substructures = []
            for smi in murcko_smiles:
                murcko_mol = Chem.MolFromSmiles(smi)
                if murcko_mol is not None:
                    minimal_substructures.append(murcko_mol)
            minimal_substructures.sort(key=lambda mol: mol.GetNumHeavyAtoms(), reverse=True)

        # If requested, group minimal substructures by similarity then compute an MCS per group.
        if collecting_method == "MinBMS clustering (MSC)":
            update_scaffold_analysis_status("CLUSTERING MINIMAL SUBSTRUCTURES\n     BY TANIMOTO AND CALCULATING MSC", state, step_id=True)

            similarity_threshold = checkbox_states["Similarity threshold"] / 100  # e.g., 0.80 for 80%

            # Morgan fingerprints (bit vectors) for Tanimoto similarities
            substructure_fp_gen = GetMorganGenerator(radius=2, fpSize=4096, includeChirality=True)
            fps = [substructure_fp_gen.GetFingerprint(mol) for mol in minimal_substructures]
            n = len(fps)

            # Pairwise Tanimoto similarity matrix
            similarity_matrix = np.zeros((n, n), dtype=float)
            for i in range(n):
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
                for offset, sim in enumerate(sims, start=1):
                    j = i + offset
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

            # Condensed distance matrix (1 - similarity) for clustering
            distance_matrix = 1.0 - similarity_matrix
            condensed_dist = distance_matrix[np.triu_indices(n, k=1)]

            # Complete-linkage to enforce threshold within clusters
            linkage_matrix = linkage(condensed_dist, method='complete')

            # Threshold in distance space
            cut_distance = 1.0 - similarity_threshold
            cluster_labels = fcluster(linkage_matrix, t=cut_distance, criterion='distance')

            # Group minimal substructures by cluster label
            groups = {}
            for i, label in enumerate(cluster_labels):
                groups.setdefault(label, []).append(minimal_substructures[i])
            groups = list(groups.values())

            clusters = sum(1 for g in groups if len(g) > 1)
            append_to_log(state, f"{clusters} groups of minimal substructures (with length > 1) found with Tanimoto ≥ {similarity_threshold}.")

            clusters = 0
            for group_mols in groups:
                if len(group_mols) > 1:
                    clusters += 1
                    for mol in group_mols:
                        append_to_log(state, f"Cluster {clusters} | Minimal substructure: {Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)}")

            # Compute MCS per group; fallback to original substructures if MCS invalid or times out.
            final_scaffolds = []
            i = 0
            for group_mols in groups:
                if len(group_mols) == 1:  # singleton
                    final_scaffolds.append(group_mols[0])
                    continue

                i += 1
                append_to_log(state, f"Processing cluster {i}, containing {len(group_mols)} minimal substructures")
                update_scaffold_analysis_status(f"Processing cluster {i}, containing {len(group_mols)} minimal substructures", state, temp=True)
                try:
                    params = rdFMCS.MCSParameters()
                    if hasattr(params, "AtomTyper"):
                        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
                    if hasattr(params, "BondTyper"):
                        params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
                    if hasattr(params, "AtomCompareParameters"):
                        params.AtomCompareParameters.MatchValences = True
                        params.AtomCompareParameters.MatchChiralTag = True
                        params.AtomCompareParameters.CompleteRingsOnly = True
                        params.AtomCompareParameters.RingMatchesRingOnly = True
                    if hasattr(params, "BondCompareParameters"):
                        params.BondCompareParameters.CompleteRingsOnly = True
                        params.BondCompareParameters.RingMatchesRingOnly = True
                    if hasattr(params, "MaximizeBonds"):
                        params.MaximizeBonds = False
                    if hasattr(params, "Threshold"):
                        params.Threshold = 1.0
                    elif hasattr(params, "threshold"):
                        params.threshold = 1.0
                    if mcs_timeout_seconds is not None:
                        if hasattr(params, "Timeout"):
                            params.Timeout = mcs_timeout_seconds
                        elif hasattr(params, "timeout"):
                            params.timeout = mcs_timeout_seconds
                    group_mcs = rdFMCS.FindMCS(group_mols, parameters=params)
                except Exception:
                    fallback_kwargs = dict(
                        atomCompare=rdFMCS.AtomCompare.CompareElements,
                        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                        matchValences=True,
                        ringMatchesRingOnly=True,
                        completeRingsOnly=True,
                        threshold=1.0,
                    )
                    if mcs_timeout_seconds is not None:
                        fallback_kwargs["timeout"] = mcs_timeout_seconds
                    group_mcs = rdFMCS.FindMCS(group_mols, **fallback_kwargs)
                append_to_log(state, f"     MCS SMARTS: {group_mcs.smartsString}")

                mcs_mol_from_smarts = Chem.MolFromSmarts(group_mcs.smartsString)

                # Outcome handling: timeout, invalid MCS, or valid SMILES conversion.
                if group_mcs.canceled:
                    append_to_log(state, f"     ❌ Cluster {i}: Timeout reached ({mcs_timeout_seconds}s)")
                    final_scaffolds.extend(group_mols)

                elif (mcs_mol_from_smarts is None) or (mcs_mol_from_smarts.GetNumAtoms() == 0):
                    append_to_log(state, f"     ❌ Cluster {i} - No valid MCS found")
                    final_scaffolds.extend(group_mols)

                else:
                    try:
                        mcs_smiles = Chem.MolToSmiles(mcs_mol_from_smarts, isomericSmiles=True, canonical=True)
                        append_to_log(state, f"     MCS SMILES: {mcs_smiles}")
                        mcs_mol_from_smiles = Chem.MolFromSmiles(mcs_smiles, sanitize=True)
                        if mcs_mol_from_smiles is None:
                            mcs_mol_from_smiles = Chem.MolFromSmiles(mcs_smiles, sanitize=False)
                            if mcs_mol_from_smiles is not None:
                                try:
                                    Chem.SanitizeMol(
                                        mcs_mol_from_smiles,
                                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                                    )
                                except Exception:
                                    pass
                        if mcs_mol_from_smiles is not None:
                            append_to_log(state, f"     ✅ Cluster {i} - MCS calculated correctly")
                            final_scaffolds.append(mcs_mol_from_smiles)
                        else:
                            append_to_log(state, f"     ⚠️ Cluster {i} - Falling back to SMARTS-derived MCS (non-kekulized)")
                            final_scaffolds.append(Chem.Mol(mcs_mol_from_smarts))
                    except Exception:
                        try:
                            append_to_log(state, f"     ⚠️ Cluster {i} - Falling back to SMARTS-derived MCS after SMILES conversion failure")
                            final_scaffolds.append(Chem.Mol(mcs_mol_from_smarts))
                        except Exception:
                            append_to_log(state, f"     ❌ Cluster {i} - MCS conversion to SMILES failed")
                            final_scaffolds.extend(group_mols)

            # Update minimal substructures with unique final scaffolds and sort by size
            minimal_substructures = final_scaffolds

            unique_smiles = set()
            unique_mols = []
            for mol in minimal_substructures:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                if smi not in unique_smiles:
                    unique_smiles.add(smi)
                    unique_mols.append(mol)
            minimal_substructures = unique_mols

            minimal_substructures.sort(key=lambda mol: mol.GetNumHeavyAtoms(), reverse=True)

            append_to_log(state, f"{len(minimal_substructures)} minimal substructures after clustering by similarity (Tanimoto ≥ {similarity_threshold})")
            update_scaffold_analysis_status(f"   {len(minimal_substructures)} minimal substructures after\n   clustering by similarity (Tanimoto >= {similarity_threshold})\n", state, separator=True)


    ################################################
    ### COMMON STEPS FOR ALL PATHS CONTINUE HERE ###
    ################################################

    if collecting_method in ["Unique Bemis-Murcko scaffolds (BMS)", "BMS minimal substructures (MinBMS)", "MinBMS clustering (MSC)",
                             "User-defined generalized scaffold", "User-defined generalized scaffold + similarity"]:
        
        # For user-defined scaffold mode, we already have scaffold_dict built directly from the initial filtering steps, so we can skip directly to sorting and finalizing subsets (Step 7) and subsequent

        # Create subsets by matching molecules to each minimal substructure.
        update_scaffold_analysis_status("POPULATING THE SUBSETS", state, step_id=True)
        temp_dict = {}
        for min_substruct in minimal_substructures:
            temp_dict[min_substruct] = []


        # Populate each subset with matching molecules (handling heavy-atom threshold)
        append_to_log(state, "Populating subsets with molecules matching minimal substructures...")
        update_scaffold_analysis_status(f"   Matching {len(suppl)} mols vs {len(minimal_substructures)} minimal substructures:", state)

        assigned_ids = set()  # Track molecule indices assigned when substructure below threshold

        for sub_id, scaff in enumerate(list(temp_dict.keys()), 1):
            heavy_sub = scaff.GetNumHeavyAtoms() >= checkbox_states["Heavy atoms threshold"]  # Rule toggle
            plswumo=checkbox_states.get("Populate light substructures with unassigned molecules only", True)

            update_scaffold_analysis_status(f"   Populating Subset {sub_id}", state, temp=True)

            for mol_id, mol in enumerate(suppl):
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return

                if mol is not None and mol.HasSubstructMatch(scaff, useChirality=True):
                    if heavy_sub:
                        temp_dict[scaff].append(mol)
                        if plswumo: assigned_ids.add(mol_id)
                    else:
                        if not plswumo or mol_id not in assigned_ids:
                            temp_dict[scaff].append(mol)
                            assigned_ids.add(mol_id)


        # Secondary pass: find molecules that match without chirality and assign via Murcko scaffold fallback.
        if len(assigned_ids) < len(suppl):
            chirality_unmatching_mols = []
            for mol_id, mol in enumerate(suppl):
                if mol_id not in assigned_ids:
                    for scaff in temp_dict.keys():
                        if mol is not None and mol.HasSubstructMatch(scaff, useChirality=False): # Check matches ignoring chirality
                            chirality_unmatching_mols.append(mol)
                            break
                        elif mol is None:
                            continue
                        
            append_to_log(state, f"     {len(chirality_unmatching_mols)}/{len(suppl)-len(assigned_ids)} molecules matched at least one minimal substructure without considering chirality.")

            # Recompute Murcko scaffolds for unmatched molecules and merge into existing subsets when equivalent
            secondary_scaffold_dict = {}
            for mol in chirality_unmatching_mols:
                scaffold = _murcko_scaffold_cached(mol, murcko_cache)
                if scaffold is not None:
                    smi = _mol_smiles_cached(scaffold, mol_smiles_cache)
                    scaffold_mol = Chem.MolFromSmiles(smi)
                    if scaffold_mol is not None:
                        if scaffold_mol not in secondary_scaffold_dict:
                            secondary_scaffold_dict[scaffold_mol] = []
                        secondary_scaffold_dict[scaffold_mol].append(mol)

            # Combine primary and secondary assignments (use mutual substructure check to merge)
            for new_scaffold, new_mols in secondary_scaffold_dict.items():
                found_match = False
                for existing_scaffold in temp_dict.keys():
                    if new_scaffold.HasSubstructMatch(existing_scaffold) and existing_scaffold.HasSubstructMatch(new_scaffold):
                        for mol in new_mols:
                            if mol not in temp_dict[existing_scaffold]:
                                temp_dict[existing_scaffold].append(mol)
                        found_match = True
                        break
                if not found_match:
                    temp_dict[new_scaffold] = new_mols

        # Sort subsets by size (descending) and build final scaffold dictionary keyed by SMILES.
        subsets = []
        for scaffold, mols in temp_dict.items():
            subsets.append((_mol_smiles_cached(scaffold, mol_smiles_cache), mols))
        subsets = sorted(subsets, key=lambda item: len(item[1]), reverse=True)
        scaffold_dict = dict(subsets)

        associated_mols = sum(len(mols) for mols in scaffold_dict.values())
        duplicates = max(0, associated_mols - len(suppl))
        singlets = len(suppl) - duplicates

        append_to_log(state, f"     Molecules matching at least one minimal substructure: {associated_mols}")
        append_to_log(state, f"     Molecules matching one minimal substructure: {singlets}")
        append_to_log(state, f"     Molecules matching more than one minimal substructure: {duplicates}")
        update_scaffold_analysis_status(f"   Total matches = {associated_mols}\n   Single matches = {singlets}\n   Multiple matches = {duplicates}", state, separator=True)


    # For each subset, rank molecules by the most common activity type and then by activity value.
    for scaffold, molecules in scaffold_dict.items():
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return
        
        bioactivity_types = Counter()  # Count activity types within the subset
        bio_by_mol = {}                # Map mol → list of (type, value) pairs from "Activity"

        for mol in molecules:
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            
            props = [p for p in mol.GetPropNames() if p == "Activity"]
            for p in props:
                activity = mol.GetProp(p).split(" ")
                if len(activity) >= 3:
                    bioactivity_types[activity[0]] += 1
                    bio_by_mol.setdefault(mol, []).append((activity[0], activity[2]))

        ordered_types = [btype for btype, _ in bioactivity_types.most_common()]  # Priority by frequency

        def get_sort_key(mol: Any) -> Any:
            """
            Produce a sorting key for the given molecule based on the group’s bioactivity profile.

            Molecules are ranked by:
              1) the order of most frequent bioactivity types in the subset;
              2) numeric activity values with direction:
                 - ascending for IC50-like (nM, ug/mL, etc.);
                 - descending for dimensionless/percentage activities.

            Args:
                mol (rdkit.Chem.Mol): Molecule to evaluate.

            Returns:
                tuple: (priority_index, signed_value) compatible with Python sort.
            """
            activities = bio_by_mol.get(mol, [])
            for i, activity_type in enumerate(ordered_types):
                if state.get("abort_analysis", False):
                    confirm_cancellation(state)
                    return
                
                for act_type, act_val in activities:
                    if act_type == activity_type:
                        try:
                            val = float(act_val)
                            direction = 1 if activity_type in state["nM_activity_types"] or activity_type in state["ug/mL_activities"] else -1
                            return (i, direction * val)
                        except:
                            continue
            return (len(ordered_types), float("inf"))

        scaffold_dict[scaffold] = sorted(molecules, key=get_sort_key)

    def _coerce_scaffold_mol(scaffold_value: Any) -> Any:
        if scaffold_value is None:
            return None
        if isinstance(scaffold_value, str):
            return Chem.MolFromSmiles(scaffold_value)
        if hasattr(scaffold_value, "GetNumAtoms"):
            return scaffold_value
        return None

    def _coerce_scaffold_smiles(scaffold_value: Any) -> str:
        scaffold_mol = _coerce_scaffold_mol(scaffold_value)
        if scaffold_mol is None:
            return str(scaffold_value)
        try:
            return Chem.MolToSmiles(scaffold_mol, isomericSmiles=True, canonical=True)
        except Exception:
            return str(scaffold_value)

    normalized_scaffold_dict = {_coerce_scaffold_smiles(scaffold): molecules for scaffold, molecules in scaffold_dict.items()}
    scaffold_dict = normalized_scaffold_dict


    # Retain only subsets with at least the configured number of molecules.
    update_scaffold_analysis_status("FILTERING THE SUBSETS BY SIZE", state, step_id=True)
    filtered_scaffolds = [(s, m) for s, m in scaffold_dict.items() if len(m) >= checkbox_states["Filtering threshold"]]

    if filtered_scaffolds:
        append_to_log(state, f"{len(scaffold_dict) - len(filtered_scaffolds)} subsets filtered out")
        append_to_log(state, f"{len(filtered_scaffolds)} subset{'s' if len(filtered_scaffolds) > 1 else ''} retained")
        update_scaffold_analysis_status(f"   {len(scaffold_dict) - len(filtered_scaffolds)} subsets filtered out\n   {len(filtered_scaffolds)} subset{'s' if len(filtered_scaffolds) > 1 else ''} retained", state, separator=True)
    else:
        append_to_log(state, f"\nAny common scaffold found in the dataset {selected_file}")
        update_scaffold_analysis_status("0 VALID SUBSETS", state, step_id=True, separator=True)
        confirm_cancellation(state)


    # For each retained subset, write a subset SDF and a corresponding scaffold SDF; also collect all scaffolds into a single SDF.
    update_scaffold_analysis_status("WRITING SDF AND CSV REPORTS", state, step_id=True)
    for scaffold_id, (scaffold, molecules) in enumerate(filtered_scaffolds, start=1):
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return

        output_sdf_file = os.path.join(subset_dir, f"subset_{scaffold_id}.sdf")
        with Chem.SDWriter(output_sdf_file) as writer:
            for mol in molecules:
                writer.write(mol)

        scaffold_sdf_file = os.path.join(subset_dir, f"scaffold_{scaffold_id}.sdf")
        with Chem.SDWriter(scaffold_sdf_file) as writer:
            scaffold_mol = _coerce_scaffold_mol(scaffold)
            if scaffold_mol is not None:
                writer.write(scaffold_mol)

    all_scaffolds_path = os.path.join(subset_dir, f"{state['selected_file_name'][:-4]}_AllScaffolds.sdf")
    with Chem.SDWriter(all_scaffolds_path) as writer:
        for scaffold, _ in filtered_scaffolds:
            scaffold_mol = _coerce_scaffold_mol(scaffold)
            if scaffold_mol is not None:
                writer.write(scaffold_mol)
            

    state["scaffold_id"] = scaffold_id if filtered_scaffolds else 0
    append_to_log(state, f"SDF files written in the 'sdf_files' folder")


    # 1) subset_counts.csv with per-scaffold sizes; 2) clustering.csv mapping molecules to subsets with activity columns.
    scaffold_counts = [(s, len(m)) for s, m in scaffold_dict.items()]
    sorted_counts = sorted(scaffold_counts, key=lambda x: x[1], reverse=True)

    with open(os.path.join(report_dir, "subset_counts.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subset_ID", "Common_Substructure_SMILES", "Subset_Size"])
        for i, (s, count) in enumerate(sorted_counts, start=1):
            writer.writerow([i, s, count])

    csv_file_2 = os.path.join(report_dir, "clustering.csv")
    with open(csv_file_2, mode="w", newline="") as file:
        writer = csv.writer(file)
        bio_types = set()
        for sid, (s, mols) in enumerate(filtered_scaffolds, start=1):
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            for mol in mols:
                if mol.HasProp("Activity"):
                    bio_data = mol.GetProp("Activity").split(" ")
                    bio_types.add(bio_data[0])

        bio_types = sorted(bio_types)
        writer.writerow(["Molecule_Name", "Molecule_SMILES", "Subset", "Common_Substructure_SMILES"] + bio_types)
        for sid, (s, mols) in enumerate(filtered_scaffolds, start=1):
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            for mol in mols:
                smi = _mol_smiles_cached(mol, mol_smiles_cache)
                name = mol.GetProp("Name") if mol.HasProp("Name") else mol.GetProp("_Name") if mol.HasProp("_Name") else "N/A"
                values = {b: "N/A" for b in bio_types}
                if mol.HasProp("Activity"):
                    try:
                        bio_data = mol.GetProp("Activity").split(" ")
                        values[bio_data[0]] = bio_data[2]
                    except:
                        pass
                writer.writerow([name, smi, sid, s] + [values[b] for b in bio_types])


    # Persist the scaffold dictionary in the application state and confirm CSV output.
    state["scaffold_dict"] = scaffold_dict
    append_to_log(state, f"CSV reports 'subset_counts.csv', 'clustering.csv' saved in the 'reports' folder")


# -----------------------------------------------------------------------------
# 3. Calculate minimal substructures
# -----------------------------------------------------------------------------
def calculate_minimal_substructures(murcko_smiles: str, state: dict[str, Any]) -> Any:
    """
    Execute the calculate minimal substructures routine.
    
    Args:
        murcko_smiles (Any): Parameter accepted by this routine.
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """
    append_to_log(state, "Identifying minimal substructures...")
    update_scaffold_analysis_status("IDENTIFYING MINIMAL SUBSTRUCTURES", state, step_id=True)

    checkbox_states = state["checkbox_states"]

    # Convert SMILES to Mol objects
    murcko_molobjects = [murcko_mol for smi in murcko_smiles if (murcko_mol := Chem.MolFromSmiles(smi)) is not None]

    # Iterative filtering to retain only minimal scaffolds
    current_list = murcko_molobjects
    cycle = 1
    unchanged_cycles = 0
    max_cycles = 10  # Guard to avoid infinite loops

    heavy_cache = {id(mol): mol.GetNumHeavyAtoms() for mol in current_list}

    while unchanged_cycles < 2 and cycle <= max_cycles:
        reduced_list = []

        for i, scaff1 in enumerate(current_list):
            is_substructure = False
            for j, scaff2 in enumerate(current_list):
                if i == j:
                    continue
                if heavy_cache[id(scaff2)] < checkbox_states["Heavy atoms threshold"]:
                    continue
                if scaff1.HasSubstructMatch(scaff2, useChirality=True):
                    is_substructure = True
                    break
            if not is_substructure:
                reduced_list.append(scaff1)

        if len(reduced_list) == len(current_list):
            unchanged_cycles += 1
        else:
            unchanged_cycles = 0

        current_list = reduced_list
        cycle += 1

    minimal_substructures = current_list
    minimal_substructures.sort(key=lambda mol: mol.GetNumHeavyAtoms(), reverse=True)

    append_to_log(state, f"     {len(minimal_substructures)} minimal substructures calculated")
    update_scaffold_analysis_status(f"   {len(minimal_substructures)} minimal substructures found\n", state, separator=True)

    return minimal_substructures
