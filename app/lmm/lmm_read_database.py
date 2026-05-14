"""
============================
lmm_read_database.py
============================

Public database import module.

Enables retrieval of molecular and bioactivity data directly from public sources
(ChEMBL or PubChem). Handles API queries, ID lookups, and response parsing,
automatically generating a comprehensive dataset for analysis (in SDF format).
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Create sdf from chembl
# 3. Create sdf from pubchem
# 4. Merge fetched sdfs

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import csv
import re
import io
import time
import shutil
import random
import requests
from typing import Any
from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import SDWriter
from app.lmm.lmm_activity_curation import (
    activity_curation,
    calculate_pvalue,
    classify_assay
)
from app.utils.callbacks import append_to_log
from app.lmm.lmm_gui import update_library_preparation_status
from app.lmm.lmm_abort import confirm_cancellation


# -----------------------------------------------------------------------------
# 2. Create sdf from chembl
# -----------------------------------------------------------------------------
def create_sdf_from_chembl(state: dict[str, Any]) -> None:
    """
    Download biological activity data from ChEMBL for the provided ChEMBL target.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        None: This routine updates state or performs side effects in place.
    """

    update_library_preparation_status("[CHEMBL] SEARCHING FOR BIOACTIVITY DATA", state, step_id=True)
    checkbox_states = state["checkbox_states"]
    state["checkbox_states"]["File extension"] = "sdf"  # Force file extension to .sdf
    input_dir = state["input_dir"]
    subset_dir = state["subset_dir"]
    target_chembl_id = checkbox_states["CHEMBL target ID"].upper().strip()
    job_name = checkbox_states["Job name"]

    # Paginate through ChEMBL's REST API, collecting all activity entries for the target.
    base_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_chembl_id}"
    limit = 1000
    offset = 0
    all_molecules = []
    show_total_activities = True

    append_to_log(state, f"Fetching activities for target {target_chembl_id}...")
    update_library_preparation_status(f"   Fetching activities for target {target_chembl_id}", state)

    try:
        while True:
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return
            url = f"{base_url}&limit={limit}&offset={offset}"
            append_to_log(state, f"Fetching data from: {url}")
            response = requests.get(url)
            data = response.json()
            all_molecules += data["activities"]

            if show_total_activities:
                append_to_log(state, f"{data['page_meta']['total_count']} molecules with bioactivities found for target {target_chembl_id}")
                update_library_preparation_status(f"   {data['page_meta']['total_count']} molecules found for target {target_chembl_id}", state)
                show_total_activities = False

            if offset + limit < data["page_meta"]["total_count"]:
                append_to_log(state, f"Downloading {offset}-{offset+limit}")
                update_library_preparation_status(f"   Downloaded {offset}-{offset+limit}", state, temp=True)
            else:
                append_to_log(state, f"Downloading {offset}-{data['page_meta']['total_count']}")
                update_library_preparation_status(f"   Downloaded {offset}-{data['page_meta']['total_count']}", state, temp=True)
                break
            offset += limit

    except requests.exceptions.JSONDecodeError:
        append_to_log(state, "❌ Error: Unable to decode JSON response. It can be due to server errors. Please also check the target ID")
        update_library_preparation_status("   Error: Unable to decode\n   It can be due to server errors\n   Please check the target ChEMBL_ID", state, separator=True)
        return


    # Extract per-entry fields, clean/curate activity, compute pActivity, and classify assay.
    molecules = []
    for entry in all_molecules:
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return

        name = entry.get("molecule_pref_name", "N/A")
        chembl_id = entry["molecule_chembl_id"]
        smiles = entry.get("canonical_smiles", "N/A")
        bioactivity_type = entry.get("standard_type", "N/A")
        bioactivity_relation = entry.get("standard_relation", "N/A")
        bioactivity_value = entry.get("standard_value", "N/A")
        bioactivity_units = entry.get("standard_units", "N/A")

        action_info = entry.get("action_type") or {}
        action_type = action_info.get("action_type", "N/A")
        action_description = action_info.get("description", "N/A")
        assay_description = entry.get("assay_description", "N/A")
        assay_chembl_id = entry.get("assay_chembl_id", "N/A")
        assay_type = entry.get("assay_type", "N/A")
        bao_label = entry.get("bao_label", "N/A")
        bao_format = entry.get("bao_format", "N/A")
        target_organism = entry.get("target_organism", "N/A")
        target_molecule = entry.get("target_pref_name", "N/A")
        target_chembl_id = entry.get("target_chembl_id", "N/A")
        max_phase = entry.get("max_phase", "N/A")
        comment = entry.get("activity_comment", "N/A")

        # Normalise common variants of activity type strings to a consistent form.
        if bioactivity_type in ["ResidualActivity", "Residual Activity", "Residual activity", "Residual_Activity", "Residual_activity"]:
            bioactivity_type = "Residualactivity"
        elif bioactivity_type in [r"%ofcontrol", r"% Of Control", r"% of control", r"% of Control"]:
            bioactivity_type = "%Control"
        elif bioactivity_type in ["flu intensity", "Flu intensity", "Flu Intensity"]:
            bioactivity_type = "Fluintensity"

        # Curate activity values and optionally compute pActivity (inverted relation) for nM types.
        if bioactivity_type and bioactivity_relation and bioactivity_value and bioactivity_units:

            # Step 3.2.1: Skip ambiguous activities if user disabled them
            if (checkbox_states["Enable ambiguous activities"] == False) and bioactivity_relation in ["<", "<=", ">=", ">"]:
                bioactivity = ""
                log_activity = ""

            else:
                # Step 3.2.2: Standardise activity fields (type/value/units)
                bioactivity_type, bioactivity_value, bioactivity_units = activity_curation(
                    bioactivity_type, bioactivity_value, bioactivity_units
                )
                bioactivity = f"{bioactivity_type} {bioactivity_relation} {bioactivity_value} {bioactivity_units}"

                # Step 3.2.3: Compute pActivity if in nM and type is supported
                if bioactivity_type in state["nM_activity_types"] and bioactivity_units == "nM":
                    pvalue = calculate_pvalue(bioactivity_type, bioactivity_value, bioactivity_units)
                    inverted_relation = (
                        ">" if bioactivity_relation == "<" else
                        "<" if bioactivity_relation == ">" else
                        ">=" if bioactivity_relation == "<=" else
                        "<=" if bioactivity_relation == ">=" else
                        "="
                    )
                    log_activity = f"p{bioactivity_type} {inverted_relation} {pvalue}"
                else:
                    log_activity = ""
        else:
            bioactivity = ""
            log_activity = ""

        # Convert BAO and assay type metadata into a concise assay label (e.g., Binding/Functional).
        assay = classify_assay(bao_label, assay_type)

        # Collect parsed and curated fields into a per-molecule dictionary to be written to SDF.
        molecule_entry = {
            "name": name,
            "chembl_id": chembl_id,
            "smiles": smiles,
            "bioactivity": bioactivity,
            "log_activity": log_activity,
            "action_type": action_type,
            "action_description": action_description,
            "assay_description": assay_description,
            "assay_chembl_id": assay_chembl_id,
            "assay type": assay_type,
            "bao label": bao_label,
            "bao format": bao_format,
            "assay": assay,
            "target": target_molecule,
            "target_chembl_id": target_chembl_id,
            "organism": target_organism,
            "max_phase": max_phase,
            "comment": comment
        }

        molecules.append(molecule_entry)

    print(f"\nGot biological activitiy data for {len(molecules)} compound entries")

    # Build SDF path and stream all curated molecules to disk while updating UI.
    print("Writing molecules in SDF...")
    sdf_filename = f"{job_name}_ChEMBL.sdf"
    output_path = os.path.join(input_dir, sdf_filename)

    with SDWriter(output_path) as writer:
        for idx, mol_data in enumerate(molecules, start=1):
            if state.get("abort_analysis", False):
                confirm_cancellation(state)
                return

            smiles = mol_data.get("smiles")
            if not smiles or smiles in ["N/A", ""]:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Prepare all SD fields to annotate the molecule with assay/target and activity metadata.
            props = {
                "_Name": mol_data.get("name"),
                "ChEMBL_ID": mol_data.get("chembl_id"),
                "Smiles": mol_data.get("smiles"),
                "Activity": mol_data.get("bioactivity"),
                "pValue": mol_data.get("log_activity"),
                "Action_Type": mol_data.get("action_type"),
                "Action_Description": mol_data.get("action_description"),
                "Assay_Description": mol_data.get("assay_description"),
                "Assay_ChEMBL_ID": mol_data.get("assay_chembl_id"),
                "Assay_Type": mol_data.get("assay type"),
                "BAO_Label": mol_data.get("bao label"),
                "BAO_Format": mol_data.get("bao format"),
                "Assay": mol_data.get("assay"),
                "Target": mol_data.get("target"),
                "Target_ChEMBL_ID": mol_data.get("target_chembl_id"),
                "Organism": mol_data.get("organism"),
                "Max_Phase": mol_data.get("max_phase"),
                "Comment": mol_data.get("comment")
            }

            # Only set non-empty values to keep the SDF concise.
            for key, value in props.items():
                if value not in [None, "N/A", ""]:
                    mol.SetProp(str(key), str(value))

            writer.write(mol)

            if idx % 100 == 0 or idx == len(molecules):
                print(f"   Writing molecule {idx}/{len(molecules)}")
                update_library_preparation_status(f"   Writing molecule {idx}/{len(molecules)}", state, temp=True)

    append_to_log(state, f"SDF file {sdf_filename}.sdf created with {len(molecules)} molecules and saved in: {subset_dir} (one copy of the SDF file was saved in {input_dir})")
    update_library_preparation_status(f"   SDF saved as: {sdf_filename}", state, separator=True)
    
    # Copy SDF into subset_dir as a backup and verify it contains valid molecules.
    sdf_path = os.path.join(input_dir, sdf_filename)
    backup_path = os.path.join(subset_dir, sdf_filename)
    shutil.copyfile(sdf_path, backup_path)

    state["selected_file_path"] = sdf_path
    state["output_sdf"] = backup_path
    add_recent_file = state.get("add_recent_file")
    if callable(add_recent_file):
        add_recent_file(sdf_path)
        add_recent_file(backup_path)
    suppl = Chem.SDMolSupplier(sdf_path)

    valid_mols = [mol for mol in suppl if mol is not None]
    if len(valid_mols) == 0:
        append_to_log(state, f"❌ The SDF file '{sdf_filename}' is empty or contains no valid molecules")
        update_library_preparation_status("   SERVER ERROR\n   ANALYSIS ABORTED", state, separator=True)
        confirm_cancellation(state)
        return


# -----------------------------------------------------------------------------
# 3. Create sdf from pubchem
# -----------------------------------------------------------------------------
def create_sdf_from_pubchem(state: dict[str, Any]) -> Any:
    """
    Download biological assay data from PubChem for a target resolvable via.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    # -----------------------------------------------------------------------------
    # 3.1. Explain exc
    # -----------------------------------------------------------------------------
    def _explain_exc(e: Any, ctx: Any) -> Any:
        """
        Execute the explain exc routine.
        
        Args:
            e (Any): Parameter accepted by this routine.
            ctx (str): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        if isinstance(e, requests.HTTPError):
            r = getattr(e, "response", None)
            code = getattr(r, "status_code", None)
            if code == 404: return f"{ctx}: not found (404). Check the ID or accession."
            if code == 400: return f"{ctx}: bad request (400)."
            if code == 401: return f"{ctx}: unauthorized (401)."
            if code == 403: return f"{ctx}: forbidden (403)."
            if code in (429, 500, 502, 503, 504): return f"{ctx}: server busy/temporary error ({code})."
            return f"{ctx}: HTTP error ({code})."
        if isinstance(e, requests.Timeout):      return f"{ctx}: network timeout."
        if isinstance(e, requests.ConnectionError):   return f"{ctx}: network connection error."
        return f"{ctx}: unexpected error."


    update_library_preparation_status("[PUBCHEM] SEARCHING FOR BIOACTIVITY DATA", state, step_id=True)
    checkbox_states = state["checkbox_states"]
    state["checkbox_states"]["File extension"] = "sdf"  # Force file extension to .sdf
    input_dir = state["input_dir"]
    subset_dir = state["subset_dir"]
    target_chembl_id = checkbox_states["CHEMBL target ID"].upper().strip()
    job_name = checkbox_states["Job name"]

    append_to_log(state, f"Fetching activities for target {target_chembl_id}...")
    update_library_preparation_status(f"   Fetching activities for target {target_chembl_id}", state)

    # Define headers, endpoints, and resilient GET helpers for JSON/text responses.
    H = {"User-Agent":"Mozilla/5.0","Accept":"application/json,text/plain,*/*"}
    URL_CHEMBL_TGT   = "https://www.ebi.ac.uk/chembl/api/data/target/{tid}.json"
    URL_UNIPROT_JSON = "https://rest.uniprot.org/uniprotkb/{acc}.json"
    URL_TGT_ACC_CSV  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/accession/{acc}/concise/CSV"
    URL_TGT_SYM_CSV  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/genesymbol/{sym}/concise/CSV"
    URL_TGT_GID_CSV  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/geneid/{gid}/concise/CSV"
    URL_PUGVIEW_COMPOUND = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    URL_AID_CONCISE_CSV = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/concise/CSV"

    # -----------------------------------------------------------------------------
    # 3.2. Get
    # -----------------------------------------------------------------------------
    def _get(url: str, accept: Any = None, retry: int = 7, base_sleep: float = 1.0) -> Any:
        """
        Return the requested value.
        
        Args:
            url (str): Parameter accepted by this routine.
            accept (Any): Parameter accepted by this routine. Defaults to the configured value.
            retry (Any): Parameter accepted by this routine. Defaults to the configured value.
            base_sleep (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            Any: Value produced by the routine.
        """
        hdr = dict(H);  hdr["Accept"] = accept or H["Accept"]
        last = None
        for n in range(retry+1):
            try:
                r = requests.get(url, headers=hdr, timeout=30)
                if r.status_code == 200:
                    return r
                # Apply backoff on 429/5xx, honoring Retry-After when available.
                if r.status_code in (429, 500, 502, 503, 504) and n < retry:
                    ra = r.headers.get("Retry-After")
                    wait = float(ra) if ra and ra.isdigit() else base_sleep * (2 ** n)
                    wait += random.uniform(0.0, 0.5)  # jitter
                    time.sleep(min(wait, 30))
                    continue
                r.raise_for_status()
            except Exception as e:
                last = e
                if n < retry:
                    wait = base_sleep * (2 ** n) + random.uniform(0.0, 0.5)
                    time.sleep(min(wait, 30))
                    continue
                raise
        raise last
    

    # -----------------------------------------------------------------------------
    # 3.3. Jget
    # -----------------------------------------------------------------------------
    def jget(url: str) -> Any:
        """
        Retrieve and decode a JSON payload from the provided endpoint.

        Args:
            url (str): Endpoint to query.

        Returns:
            Any: Parsed JSON payload returned by the endpoint.
        """
        return _get(url, "application/json").json()

    # -----------------------------------------------------------------------------
    # 3.4. Tget
    # -----------------------------------------------------------------------------
    def tget(url: str) -> Any:
        """
        Retrieve a plain-text payload from the provided endpoint.

        Args:
            url (str): Endpoint to query.

        Returns:
            str: Response body returned as plain text.
        """
        return _get(url, "text/plain,*/*").text

    # --- Target resolution helpers (ChEMBL → UniProt, then gene symbol/GeneID) ---
    # -----------------------------------------------------------------------------
    # 3.5. Chembl to uniprot
    # -----------------------------------------------------------------------------
    def chembl_to_uniprot(tid: Any) -> Any:
        """
        Execute the chembl to uniprot routine.
        
        Args:
            tid (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        d = jget(URL_CHEMBL_TGT.format(tid=quote(tid)))
        if d.get("target_chembl_id","").upper()!=tid.upper():
            raise requests.HTTPError("ChEMBL ID mismatch")
        if str(d.get("target_type","")).upper()!="SINGLE PROTEIN":
            raise ValueError("Target type is not SINGLE PROTEIN")
        for c in d.get("target_components") or []:
            if c.get("component_type")=="PROTEIN" and c.get("accession"):
                return c["accession"], d.get("organism"), d.get("pref_name")
        raise ValueError("No UniProt accession for target")

    # -----------------------------------------------------------------------------
    # 3.6. Uniprot gene
    # -----------------------------------------------------------------------------
    def uniprot_gene(acc: Any) -> Any:
        """
        Execute the uniprot gene routine.
        
        Args:
            acc (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        d = jget(URL_UNIPROT_JSON.format(acc=quote(acc)))
        sym=None
        try: sym=(d.get("genes",[{}])[0].get("geneName") or {}).get("value")
        except: pass
        gid=None
        for x in d.get("uniProtKBCrossReferences",[]) or []:
            if x.get("database")=="GeneID":
                gid = x.get("id") or next((p.get("value") for p in x.get("properties",[])
                                           if str(p.get("key","")).lower().startswith("geneid")), None)
                if gid: break
        return sym, gid

    # --- PubChem concise table fetchers and utilities ---
    # -----------------------------------------------------------------------------
    # 3.7. Concise for target
    # -----------------------------------------------------------------------------
    def concise_for_target(acc: Any, sym: Any = None, gid: Any = None) -> Any:
        """
        Execute the concise for target routine.
        
        Args:
            acc (Any): Parameter accepted by this routine.
            sym (Any): Parameter accepted by this routine. Defaults to the configured value.
            gid (Any): Parameter accepted by this routine. Defaults to the configured value.
        
        Returns:
            Any: Value produced by the routine.
        """
        for tag, url in [
            (f"accession {acc}", URL_TGT_ACC_CSV.format(acc=quote(acc))),
            (f"genesymbol {sym}", URL_TGT_SYM_CSV.format(sym=quote(sym))) if sym else (None,None),
            (f"geneid {gid}", URL_TGT_GID_CSV.format(gid=quote(gid))) if gid else (None,None),
        ]:
            if not url: continue
            print(f"Trying {tag}: {url}")
            update_library_preparation_status(f"   Trying to fetch concise table via {tag}:\n      {url}", state)
            try:
                txt = tget(url)
                lines = txt.splitlines()
                if len(lines)>=2 and ("," in lines[0] or "\t" in lines[0]):
                    print(f"  -> {len(lines)-1} righe")
                    return txt
                print("  -> empty/unexpected")
            except Exception as e:
                print("  -> error:", e)
                update_library_preparation_status(f"      Error: {e}", state)
        return ""

    # -----------------------------------------------------------------------------
    # 3.8. Detect delim
    # -----------------------------------------------------------------------------
    def detect_delim(first_line: Any) -> Any:
        """
        Detect delim.
        
        Args:
            first_line (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        return "\t" if ("\t" in first_line and first_line.count("\t")>=first_line.count(",")) else ","

    # -----------------------------------------------------------------------------
    # 3.9. Find col
    # -----------------------------------------------------------------------------
    def find_col(headers: Any, candidates: Any) -> Any:
        """
        Find col.
        
        Args:
            headers (Any): Parameter accepted by this routine.
            candidates (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        for c in candidates:
            if c in headers: return c
        norm = {h: re.sub(r"[_\s]+"," ",h).strip().lower() for h in headers}
        for c in candidates:
            cn = re.sub(r"[_\s]+"," ",c).strip().lower()
            for h,hn in norm.items():
                if hn == cn: return h
        return None

    # ---- PUG-View parsing (SMILES/IUPAC/Title) ----
    # -----------------------------------------------------------------------------
    # 3.10. Strings from value
    # -----------------------------------------------------------------------------
    def _strings_from_value(valobj: Any) -> Any:
        """
        Execute the strings from value routine.
        
        Args:
            valobj (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        if not isinstance(valobj, dict): return []
        out = []
        if "StringWithMarkup" in valobj:
            for itm in valobj.get("StringWithMarkup") or []:
                s = itm.get("String")
                if s: out.append(str(s))
        if "String" in valobj:
            if isinstance(valobj["String"], list):
                out.extend([str(x) for x in valobj["String"] if x is not None])
            elif valobj["String"] is not None:
                out.append(str(valobj["String"]))
        return out

    # -----------------------------------------------------------------------------
    # 3.11. Walk sections
    # -----------------------------------------------------------------------------
    def _walk_sections(sections: Any, want: Any) -> Any:
        """
        Execute the walk sections routine.
        
        Args:
            sections (Any): Parameter accepted by this routine.
            want (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        found = {k: [] for k in want}
        if not isinstance(sections, list): return found
        for sec in sections:
            head = sec.get("TOCHeading")
            if head in want:
                for info in sec.get("Information") or []:
                    vs = _strings_from_value(info.get("Value") or {})
                    if vs: found[head].extend(vs)
            child = sec.get("Section")
            if child:
                sub = _walk_sections(child, want)
                for k in want:
                    if sub[k]: found[k].extend(sub[k])
        return found

    # -----------------------------------------------------------------------------
    # 3.12. Fetch compound fields via pugview
    # -----------------------------------------------------------------------------
    def fetch_compound_fields_via_pugview(cid: Any) -> Any:
        """
        Execute the fetch compound fields via pugview routine.
        
        Args:
            cid (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        d = jget(URL_PUGVIEW_COMPOUND.format(cid=quote(str(cid))))
        rec = d.get("Record") or {}
        title = rec.get("RecordTitle","") or ""
        sections = rec.get("Section") or []
        got = _walk_sections(sections, want={"IUPAC Name","SMILES"})
        iupac = got.get("IUPAC Name")[0] if got.get("IUPAC Name") else ""
        smiles = got.get("SMILES")[0] if got.get("SMILES") else ""
        return {"Title": title, "IUPACName": iupac, "SMILES": smiles}

    _PROP_CACHE = {}
    # -----------------------------------------------------------------------------
    # 3.13. Get props for cid
    # -----------------------------------------------------------------------------
    def get_props_for_cid(cid: Any) -> Any:
        """
        Return props for cid.
        
        Args:
            cid (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        key = str(cid)
        if key in _PROP_CACHE:
            return _PROP_CACHE[key]
        try:
            info = fetch_compound_fields_via_pugview(cid)
        except Exception:
            info = {"Title":"", "IUPACName":"", "SMILES":""}
        _PROP_CACHE[key] = info
        return info

    # -----------------------------------------------------------------------------
    # 3.14. Pick
    # -----------------------------------------------------------------------------
    def pick(row: Any, names: str) -> Any:
        """
        Execute the pick routine.
        
        Args:
            row (Any): Parameter accepted by this routine.
            names (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        for n in names:
            if n in row and row[n] not in (None,""):
                return str(row[n])
        return ""

    # -----------------------------------------------------------------------------
    # 3.15. Extract units from header generic
    # -----------------------------------------------------------------------------
    def extract_units_from_header_generic(header: Any) -> Any:
        """
        Execute the extract units from header generic routine.
        
        Args:
            header (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        if not header: return ""
        m = re.search(r"\[([^\]]+)\]", header)
        if m: return m.group(1).strip()
        m = re.search(r"\(([^)]+)\)", header)
        if m: return m.group(1).strip()
        return ""

    # -----------------------------------------------------------------------------
    # 3.16. Find activity value col
    # -----------------------------------------------------------------------------
    def find_activity_value_col(headers: Any) -> Any:
        """
        Find activity value col.
        
        Args:
            headers (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        for cand in ["Activity_Value","Activity Value","Result Value","Value","Readout Value"]:
            c = find_col(headers, [cand])
            if c: return c, extract_units_from_header_generic(c)
        for h in headers:
            if re.match(r"activity[\s_]*value", h, flags=re.I):
                return h, extract_units_from_header_generic(h)
        return None, ""

    # -----------------------------------------------------------------------------
    # 3.17. Choose primary activity
    # -----------------------------------------------------------------------------
    def choose_primary_activity(row: Any, headers: Any) -> Any:
        # Pick the primary activity fields from the concise table (fallback to specific measures).
        """
        Execute the choose primary activity routine.
        
        Args:
            row (Any): Parameter accepted by this routine.
            headers (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        type_col = find_col(headers, ["Activity Name","Activity_Type","Activity Type","Result Type","Measurement Type"])
        rel_col  = find_col(headers, ["Activity Relation","Activity_Qualifier","Activity Qualifier","Qualifier","Relation","Value Relation","Result Qualifier"])
        value_col, units_from_header = find_activity_value_col(headers)
        units_col = find_col(headers, ["Activity Units","Activity_Units","Activity Unit","Result Unit","Unit","Units"])

        if value_col and row.get(value_col) not in (None, ""):
            units = ""
            if units_col and row.get(units_col) not in (None, ""):
                units = str(row.get(units_col,""))
            elif units_from_header:
                units = units_from_header
            else:
                units = "uM"
            relation = str(row.get(rel_col,"")).strip() if rel_col else "="
            if not relation: relation = "="
            return {
                "type":     str(row.get(type_col,"")) if type_col else "",
                "relation": relation,
                "value":    str(row.get(value_col,"")),
                "units":    units,
            }

        # Specific measures fallback (e.g., AC50, IC50, etc.)
        candidates = state["activity_types"]
        for meas in candidates:
            val_col = find_col(headers, [meas])
            if val_col and row.get(val_col) not in (None,""):
                u_col = find_col(headers, [f"{meas} Unit", f"{meas} Units"])
                units = str(row.get(u_col,"")) if u_col else extract_units_from_header_generic(val_col) or "uM"
                q_col = find_col(headers, [f"{meas} Qualifier","Qualifier","Activity Qualifier","Activity_Qualifier","Relation"])
                relation = str(row.get(q_col,"")).strip() if q_col else "="
                return {"type": meas, "relation": relation, "value": str(row.get(val_col,"")), "units": units}

        return {"type":"","relation":"","value":"","units":""}

    _AID_CACHE = {}
    # -----------------------------------------------------------------------------
    # 3.18. Fetch activity from aid
    # -----------------------------------------------------------------------------
    def fetch_activity_from_aid(aid: Any, cid: Any) -> Any:
        # Retrieve per-AID activity details for a compound (CID) and parse primary fields.
        """
        Execute the fetch activity from aid routine.
        
        Args:
            aid (Any): Parameter accepted by this routine.
            cid (Any): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        try:
            if aid in _AID_CACHE:
                rdr = _AID_CACHE[aid]
            else:
                txt = tget(URL_AID_CONCISE_CSV.format(aid=quote(str(aid))))
                delim = detect_delim(txt.splitlines()[0]) if txt else ","
                rdr = list(csv.DictReader(io.StringIO(txt), delimiter=delim)) if txt else []
                _AID_CACHE[aid] = rdr
            if not rdr: return {"type":"","relation":"","value":"","units":""}
            hdrs = list(rdr[0].keys())
            cid_key_local = find_col(hdrs, ["Compound_CID","Compound CID","PubChem CID","Compound PubChem CID","CID"])
            if not cid_key_local: return {"type":"","relation":"","value":"","units":""}
            for row in rdr:
                if str(row.get(cid_key_local,"")).strip() == str(cid):
                    act = choose_primary_activity(row, hdrs)
                    if not act.get("relation"): act["relation"] = "="
                    if not act.get("units"):
                        _, u_from_hdr = find_activity_value_col(hdrs)
                        act["units"] = u_from_hdr or "uM"
                    return act
        except Exception:
            pass
        return {"type":"","relation":"","value":"","units":""}

    # Resolve a ChEMBL target to UniProt; otherwise treat provided string as a UniProt accession.
    try:
        if target_chembl_id.upper().startswith("CHEMBL"):
            acc, org, pref = chembl_to_uniprot(target_chembl_id)
            sym, gid = uniprot_gene(acc)
        else:
            acc = target_chembl_id.strip()
            d = jget(URL_UNIPROT_JSON.format(acc=quote(acc)))
            pref = d.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
            org  = (d.get("organism", {}).get("scientificName", "") or d.get("organism", {}).get("commonName", "") or "N/A")
            sym, gid = uniprot_gene(acc)
    except Exception as e:
        # ChEMBL mismatch / UniProt error / HTTP errors fall here
        reason = _explain_exc(e, "Target resolution (ChEMBL/UniProt)")
        append_to_log(state, f"Abort: {reason}")
        update_library_preparation_status(f"   {reason}\n   Analysis aborted", state, separator=True)
        confirm_cancellation(state)
        return

    print(f"\n    Protein Name:           {pref or '(no title)'}")
    print(f"    Gene Symbol (UniProt):  {sym or 'N/A'}")
    print(f"    Organism:               {org or 'N/A'}")
    print(f"    UniProt Accession ID:   {acc}")
    print(f"    PubChem URL:            {f'https://pubchem.ncbi.nlm.nih.gov/protein/{acc}'}\n")

    update_library_preparation_status(f"      Protein: {pref or '(no title)'}", state)
    update_library_preparation_status(f"      Gene Symbol: {sym or 'N/A'}", state)
    update_library_preparation_status(f"      Organism: {org or 'N/A'}", state)
    update_library_preparation_status(f"      UniProt Accession ID: {acc}", state)

    try:
        csv_text = concise_for_target(acc, sym, gid)
    except Exception as e:
        reason = _explain_exc(e, "PubChem concise fetch")
        append_to_log(state, f"Abort: {reason}")
        update_library_preparation_status(f"   {reason}\n   Analysis aborted", state, separator=True)
        confirm_cancellation(state)
        return

    if not csv_text:
        # Distinzione: target valido ma nessuna riga utile dai concise
        msg = "   No concise results for this target. Either no assays " \
              "   are indexed or the server returned empty data." \
              "   Analysis aborted"
        append_to_log(state, msg)
        update_library_preparation_status(msg, state, separator=True)
        confirm_cancellation(state)
        return

    delim = detect_delim(csv_text.splitlines()[0])
    rdr = csv.DictReader(io.StringIO(csv_text), delimiter=delim)
    rows = list(rdr)
    hdrs = rdr.fieldnames or []

    cid_key = find_col(hdrs, ["Compound_CID","Compound CID","PubChem CID","Compound PubChem CID","CID"])
    aid_key = find_col(hdrs, ["BioAssay_AID","BioAssay AID","AID"])
    name_key = find_col(hdrs, ["Target_Name","Target Name"])
    assay_name_key = find_col(hdrs, ["BioAssay_Name","Assay Name"])
    assay_type_key = find_col(hdrs, ["Aid_Type","AID Type","Assay Type"])  # if present

    append_to_log(state, f"{len(rows)} molecules with bioactivities found for {sym}")
    update_library_preparation_status(f"   {len(rows)} molecules found for {sym}", state)

    # For each concise row: fetch structural strings (PUG-View), choose a primary activity, curate it,
    # and assemble a minimal PubChem-compliant dictionary of properties.
    molecules = []

    for idx, r in enumerate(rows, 1):
        if state.get("abort_analysis", False):
            confirm_cancellation(state)
            return

        append_to_log(state, f"Processing Compound {idx}/{len(rows)}")
        update_library_preparation_status(f"    Processing Compound {idx}/{len(rows)}", state, temp=True)

        cid = str(r.get(cid_key,"") or "").strip()
        aid = str(r.get(aid_key,"") or "").strip() if aid_key else ""

        # Structural strings via PUG-View (cached by CID)
        props  = get_props_for_cid(cid) if cid else {"Title":"","IUPACName":"","SMILES":""}
        iupac  = (props.get("IUPACName","") or "").strip()
        title  = (props.get("Title","") or "").strip()
        smiles = (props.get("SMILES","") or "").strip()

        # Choose a readable record name: empty if Title equals IUPAC, otherwise Title.
        def _norm(s: Any) -> Any:
            """
            Execute the norm routine.
            
            Args:
                s (Any): Input accepted by this routine.
            
            Returns:
                Any: Value returned by the routine.
            """
            return re.sub(r"[^a-z0-9]+","", (s or "").lower())
        molecule_name = "" if (_norm(iupac) and _norm(iupac)==_norm(title)) else title

        # Primary activity: pick from concise row; fallback to AID table for same CID if needed.
        act = choose_primary_activity(r, hdrs)
        if not act["value"] and aid and cid:
            act2 = fetch_activity_from_aid(aid, cid)
            if act2["value"]:
                act = act2

        bio_type   = act.get("type") or ""
        bio_rel    = (act.get("relation") or "=")
        bio_value  = act.get("value") or ""
        bio_units  = act.get("units") or "uM"

        # Curation + internal pActivity computation (pActivity NOT written to SDF).
        bio_type_cur, bio_value_cur, bio_units_cur = bio_type, bio_value, bio_units
        if all(x not in [None, ""] for x in [bio_type, bio_rel, bio_value, bio_units]):
            if (state["checkbox_states"].get("Enable ambiguous activities") == False) and bio_rel in ["<", "<=", ">=", ">"]:
                bio_type_cur = ""
                bio_value_cur = ""
                bio_units_cur = ""
            else:
                try:
                    bio_type_cur, bio_value_cur, bio_units_cur = activity_curation(bio_type, bio_value, bio_units)
                except Exception:
                    pass
                if bio_type_cur in state.get("nM_activity_types", set()) and bio_units_cur == "nM":
                    try:
                        _ = calculate_pvalue(bio_type_cur, bio_value_cur, bio_units_cur)
                    except Exception:
                        pass

        # Target and assay (if present in concise headers)
        assay_desc   = r.get(assay_name_key, "") if assay_name_key else ""
        target_name  = (r.get(name_key) or pref or "")
        target_org   = org or ""
        target_unip  = acc

        # Minimal PubChem headers to be written into the SDF.
        molecule_entry = {
            "IUPAC_Name":            iupac or pick(r, ["Compound_Name","IUPAC Name","Compound Name"]),
            "PubChem_CID":           cid,
            "_Name":                 molecule_name,
            "Smiles":                smiles,
            "Activity":              f"{bio_type_cur} {bio_rel or '='} {bio_value_cur} {bio_units_cur or 'uM'}" if bio_type_cur and bio_value_cur and bio_units_cur else "",
            "Assay_PubChem_AID":     aid,
            "Assay_Description":     assay_desc,
            "Target":                target_name,
            "Target_ChEMBL_ID":      target_chembl_id if target_chembl_id.upper().startswith("CHEMBL") else "",
            "Target_Uniprot_ID":     target_unip,
            "Organism":              target_org,
        }
        molecules.append(molecule_entry)

    append_to_log(state, f"Got biological activity data for {len(molecules)} compound entries")

    # Stream curated molecules into an SDF file and provide progress updates.
    append_to_log(state, "Writing molecules in SDF...")
    update_library_preparation_status("   Writing molecules in SDF...", state)

    sdf_filename = f"{job_name}_PubChem.sdf"
    output_path = os.path.join(input_dir, sdf_filename)

    # --- Around SDWriter loop, add a defensive try if desired ---
    try:
        with SDWriter(output_path) as writer:
            for idx, mol_data in enumerate(molecules, start=1):
                if state.get("abort_analysis", False):
                    confirm_cancellation(state); return
                smiles = mol_data.get("Smiles")
                if not smiles: continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: continue
                # set props...
                writer.write(mol)
                if idx % 100 == 0 or idx == len(molecules):
                    update_library_preparation_status(f"   Writing molecule {idx}/{len(molecules)}", state, temp=True)
    except Exception as e:
        reason = _explain_exc(e, "SDF writing")
        append_to_log(state, f"Abort: {reason}")
        update_library_preparation_status(f"   {reason}\n   Analysis aborted", state, separator=True)
        confirm_cancellation(state)
        return

    append_to_log(state, f"SDF file {sdf_filename} created with {len(molecules)} molecules and saved in: {subset_dir} (one copy of the SDF file was saved in {input_dir})")
    update_library_preparation_status(f"   SDF saved as: {sdf_filename}", state, separator=True)

    # Copy to subset_dir and verify the SDF contains at least one valid molecule.
    sdf_path = os.path.join(input_dir, sdf_filename)
    backup_path = os.path.join(subset_dir, sdf_filename)
    shutil.copyfile(sdf_path, backup_path)

    state["selected_file_path"] = sdf_path
    state["output_sdf"] = backup_path
    add_recent_file = state.get("add_recent_file")
    if callable(add_recent_file):
        add_recent_file(sdf_path)
        add_recent_file(backup_path)

    suppl = Chem.SDMolSupplier(sdf_path)
    valid_mols = [mol for mol in suppl if mol is not None]
    if len(valid_mols) == 0:
        msg = f"The SDF '{sdf_filename}' is empty or contains no valid molecules"
        append_to_log(state, f"❌ {msg}")
        update_library_preparation_status(f"   {msg}\n   Likely upstream fetch returned malformed or empty SMILES.\n   Analysis aborted.", state, separator=True)
        confirm_cancellation(state)
        return


# -----------------------------------------------------------------------------
# 4. Merge fetched sdfs
# -----------------------------------------------------------------------------
def merge_fetched_sdfs(state: dict[str, Any]) -> Any:
    """
    Merge SDF files from ChEMBL and PubChem when both exist and are valid.
    
    Args:
        state (dict[str, Any]): Parameter accepted by this routine.
    
    Returns:
        Any: Value produced by the routine.
    """

    update_library_preparation_status("[MERGE] MERGING ChEMBL + PubChem SDF", state, step_id=True)
    input_dir  = state["input_dir"]
    subset_dir = state["subset_dir"]
    job_name   = state["checkbox_states"]["Job name"]

    chembl_fname   = f"{job_name}_ChEMBL.sdf"
    pubchem_fname  = f"{job_name}_PubChem.sdf"   # unified spelling
    merged_fname   = f"{job_name}_ChEMBL_PubChem.sdf"

    chembl_path  = os.path.join(input_dir, chembl_fname)
    pubchem_path = os.path.join(input_dir, pubchem_fname)
    out_path     = os.path.join(input_dir, merged_fname)

    # Load only if file exists; count valid molecules.
    # -----------------------------------------------------------------------------
    # 4.1. Load valid
    # -----------------------------------------------------------------------------
    def _load_valid(path: str) -> Any:
        """
        Load valid.
        
        Args:
            path (str): Parameter accepted by this routine.
        
        Returns:
            Any: Value produced by the routine.
        """
        if not os.path.isfile(path):
            return [], 0
        suppl = Chem.SDMolSupplier(path)
        mols = [m for m in suppl if m is not None]
        return mols, len(mols)

    if state.get("abort_analysis", False):
        confirm_cancellation(state); return

    chembl_mols, n_chembl   = _load_valid(chembl_path)
    pubchem_mols, n_pubchem = _load_valid(pubchem_path)

    append_to_log(state, f"Loaded {n_chembl} (ChEMBL) + {n_pubchem} (PubChem)")

    # Case A: both empty or missing -> abort.
    if n_chembl == 0 and n_pubchem == 0:
        append_to_log(state, "❌ No usable SDFs (both missing or empty). Aborting.")
        update_library_preparation_status("   Missing/empty SDFs\n   Analysis aborted", state, separator=True)
        confirm_cancellation(state)
        return

    # Case B: only ChEMBL usable -> select it and finish.
    if n_chembl > 0 and n_pubchem == 0:
        append_to_log(state, f"PubChem SDF missing/empty. Using only {chembl_fname}.")
        try:
            backup_path = os.path.join(subset_dir, chembl_fname)
            shutil.copyfile(chembl_path, backup_path)
        except Exception as e:
            append_to_log(state, f"Warning: backup copy failed -> {e}")
        state["selected_file_path"] = chembl_path
        state["output_sdf"] = backup_path
        add_recent_file = state.get("add_recent_file")
        if callable(add_recent_file):
            add_recent_file(chembl_path)
            add_recent_file(backup_path)
        update_library_preparation_status(f"   SDF selected: {chembl_fname}", state, separator=True)
        return

    # Case C: only PubChem usable -> select it and finish.
    if n_pubchem > 0 and n_chembl == 0:
        append_to_log(state, f"ChEMBL SDF missing/empty. Using only {pubchem_fname}.")
        try:
            backup_path = os.path.join(subset_dir, pubchem_fname)
            shutil.copyfile(pubchem_path, backup_path)
        except Exception as e:
            append_to_log(state, f"Warning: backup copy failed -> {e}")
        state["selected_file_path"] = pubchem_path
        state["output_sdf"] = backup_path
        add_recent_file = state.get("add_recent_file")
        if callable(add_recent_file):
            add_recent_file(pubchem_path)
            add_recent_file(backup_path)
        update_library_preparation_status(f"   SDF selected: {pubchem_fname}", state, separator=True)
        return

    # Case D: both usable -> true merge.
    append_to_log(state, f"Merging SDF files '{chembl_path}' and '{pubchem_path}'...")
    merged_mols = chembl_mols + pubchem_mols
    total = len(merged_mols)
    append_to_log(state, f"Total merged molecules: {total}")

    if state.get("abort_analysis", False):
        confirm_cancellation(state); return

    with SDWriter(out_path) as writer:
        for idx, mol in enumerate(merged_mols, 1):
            if state.get("abort_analysis", False):
                confirm_cancellation(state); return
            writer.write(mol)
            if idx % 200 == 0 or idx == total:
                update_library_preparation_status(f"   Writing molecule {idx}/{total}", state, temp=True)

    backup_path = os.path.join(subset_dir, merged_fname)
    try:
        shutil.copyfile(out_path, backup_path)
    except Exception as e:
        append_to_log(state, f"Warning: backup copy failed -> {e}")

    state["selected_file_path"] = out_path
    state["output_sdf"] = backup_path
    add_recent_file = state.get("add_recent_file")
    if callable(add_recent_file):
        add_recent_file(out_path)
        add_recent_file(backup_path)

    suppl_out = Chem.SDMolSupplier(out_path)
    valid_out = [m for m in suppl_out if m is not None]
    if len(valid_out) == 0:
        append_to_log(state, f"❌ The merged SDF '{merged_fname}' is empty or invalid")
        update_library_preparation_status("   SERVER ERROR\n   ANALYSIS ABORTED", state, separator=True)
        confirm_cancellation(state)
        return

    append_to_log(state, f"SDF file {merged_fname} created with {len(valid_out)} molecules and saved in: {subset_dir} (one copy was saved in {input_dir})")
    update_library_preparation_status(f"   SDF saved as: {merged_fname}", state, separator=True)
