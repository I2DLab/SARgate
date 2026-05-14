"""
========================
lmm_activity_curation.py
========================

Bioactivity data curation and standardisation.

Processes raw biological activity data extracted from public or local sources.
Normalises units, converts inequality expressions, and ensures consistent 
numerical representation across all activity columns before analysis.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Activity curation
# 3. Calculate pvalue
# 4. Classify assay

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import math
from typing import Any


# -----------------------------------------------------------------------------
# 2. Activity curation
# -----------------------------------------------------------------------------
def activity_curation(type: Any, value: Any, units: Any) -> Any:
    """
    Normalise activity units to a standard form and convert values to canonical units (nM or ug/mL).

    Args:
        type (str): Activity type (e.g. 'IC50', 'Inhibition', etc.).
        value (float or str): Numeric value of the activity (string is accepted and cast).
        units (str): Measurement units (e.g., 'nM', 'uM', 'M').

    Returns:
        tuple: (type, normalized_value, normalized_units)
    """
    
    # Cast the given value to float after trimming whitespace; raises if not convertible.
    value = float(str(value).strip())
    # Define equivalence lists for common molarity notations.
    pM_units = ["pM", "10^-12M", "10^-12mol/L", "pmol/L"]
    nM_units = ["nM", "10^-9M", "10^-9mol/L", "nmol/L"]
    uM_units = ["uM", "10^-6M", "10^-6mol/L", "umol/L", "microM", "micromol/L", "nmol/ml"]
    mM_units = ["mM", "10^-3M", "10^-3mol/L", "mmol/L", "microM/ml", "micromol/ml"]
    M_units  = ["M", "mol/L"]
    
    # Convert any recognised molarity into nM while preserving magnitude.
    if units in pM_units:
        units = "nM"
        value = value / 1_000
    elif units in nM_units:
        units = "nM"
    elif units in uM_units:
        value = value * 1_000
        units = "nM"
    elif units in mM_units:
        value = value * 1_000_000
        units = "nM"
    elif units in M_units:
        value = value * 1_000_000_000
        units = "nM"

    # Group unit spellings by scaling factor relative to µg/mL.
    units_1emin3 = [
        "microg/L", "microg.L-1", "microg L-1", "microg/l", "microg.l-1", "microg l-1",
        "μg/L", "μg.L-1", "μg L-1", "μg/l", "μg.l-1", "μg l-1",
        "pg/mL", "pg.mL-1", "pg mL-1", "pg/ml", "pg.ml-1", "pg ml-1", 
        "ng/mL", "ng.mL-1", "ng mL-1", "ng/ml", "ng.ml-1", "ng ml-1",
    ]

    units_1e0 = [
        "ug/mL", "ug.mL-1", "ug mL-1", "ug/ml", "ug.ml-1", "ug ml-1",
        "μg/mL", "μg.mL-1", "μg mL-1", "μg/ml", "μg.ml-1", "μg ml-1",
        "microg/mL", "microg.mL-1", "microg mL-1", "microg/ml", "microg.ml-1", "microg ml-1",
        "mcg/mL", "mcg.mL-1", "mcg mL-1", "mcg/ml", "mcg.ml-1", "mcg ml-1",
    ]

    units_1e3 = [
        "mg/L", "mg.L-1", "mg L-1", "mg/l", "mg.l-1", "mg l-1",
        "ng/uL", "ng.uL-1", "ng uL-1", "ng/ul", "ng.ul-1", "ng ul-1",
        "ng/μL", "ng.μL-1", "ng μL-1", "ng/μl", "ng.μl-1", "ng μl-1",
        "mg/mL", "mg.mL-1", "mg mL-1", "mg/ml", "mg.ml-1", "mg ml-1",
    ]

    # Convert any recognised mass concentration into ug/mL with appropriate scaling.
    if units in units_1emin3:
        value = value / 1_000
        units = "ug/mL"
    elif units in units_1e0:
        units = "ug/mL"
    elif units in units_1e3:
        value = value * 1_000
        units = "ug/mL"

    return type, value, units


# -----------------------------------------------------------------------------
# 3. Calculate pvalue
# -----------------------------------------------------------------------------
def calculate_pvalue(act_type: Any, value: Any, units: Any) -> Any:
    """
    Calculate the -log10(pValue) of an activity if the type and units are supported.

    Notes:
        - Supports typical dose/affinity metrics in molar units: IC50, EC50, GI50, DC50, CC50, LD50,
          ED50, ID50, Ka, Kd, Km, Ki.
        - Returns an empty string if inputs are not compatible or value is non-positive.

    Args:
        act_type (str): Activity type (e.g., 'IC50', 'Inhibition', etc.).
        value (float): Activity value.
        units (str): Measurement units (e.g., 'nM', 'uM', 'M').

    Returns:
        float or str: Calculated p-value (e.g., pIC50) rounded to 4 decimals, or "" if invalid.
    """
    
    # Types where a pValue (−log10 of molar quantity) is meaningful.
    nM_activity_types = ["IC50", "EC50", "GI50", "DC50", "CC50", "LD50", "ED50", "ID50", "Ka", "Kd", "Km", "Ki"]
    log_activity_types = ["pIC50", "pEC50", "pGI50", "pDC50", "pCC50", "pLD50", "pED50", "pID50", "pKa", "pKd", "pKm", "pKi",
                          "-Log IC50", "-Log EC50", "-Log GI50", "-Log DC50", "-Log CC50", "-Log LD50", 
                          "-Log ED50", "-Log ID50", "-log IC50", "-log EC50", "-log GI50", "-log DC50", 
                          "-log CC50", "-log LD50", "-log ED50", "-log ID50"]

    # Only proceed for supported types; unknown types return empty string.
    if act_type in nM_activity_types:
        try:
            value = float(value)
            if units == "pM":
                value = value / 1_000_000_000_000
            elif units == "nM":
                value = value / 1_000_000_000
            elif units == "uM":
                value = value / 1_000_000
            elif units == "mM":
                value = value / 1_000
            elif units == "M":
                value = value
            else:
                return ""

            if value > 0:
                return round(-math.log10(value), 4)
            else:
                return ""
        except Exception as e:
            print(f"[ERROR] Error in pValue calculation: {e}")
            return ""
    else:
        return ""


# -----------------------------------------------------------------------------
# 4. Classify assay
# -----------------------------------------------------------------------------
def classify_assay(bao_label: str, assay_type: Any) -> Any:
    """
    Classify assay formats into standard categories based on BAO label and assay type.

    Args:
        bao_label (str): BAO format label (e.g., 'single protein format', 'cell-based format').
        assay_type (str): Assay type defined by a single letter ('B', 'F', 'A', 'T', 'P', 'U').

    Returns:
        str: Standardised assay classification (e.g., 'In vitro biochemical', 'In vivo (functional)', etc.).
    """

    bl = (bao_label or "").lower()
    at = (assay_type or "").upper()

    # Single protein formats: prioritise binding vs functional enzyme activities.
    if bl in ["single protein format"]:
        return {
            "B": "In vitro biochemical (binding)",
            "F": "In vitro biochemical (functional)",
        }.get(at, "In vitro biochemical (other)")

    # Other biochemical-like formats (cell-free, membrane, complexes).
    if bl in [
        "protein complex format", "protein-substrate format", "membrane format",
        "cell-free format", "biochemical format"
    ]:
        return "In vitro biochemical (other)"

    # Cell-based formats with subtype mapping.
    if bl in ["cell-based format"]:
        return {
            "B": "In vitro cellular (binding)",
            "F": "In vitro cellular (functional)",
            "A": "In vitro cellular (ADME)",
            "T": "In vitro cellular (toxicity)",
        }.get(at, "In vitro cellular (other)")

    # Phenotypic screens in vitro.
    if bl in ["phenotypic format"]:
        return "In vitro cellular (phenotypic)"

    # Subcellular contexts (organelles, lysates).
    if bl in ["subcellular format"]:
        return "In vitro (subcellular)"

    # Tissue-level assays.
    if bl in ["tissue format", "tissue homogenate format"]:
        return "In vitro (tissue)"

    # Whole-organism / organism-based formats, with subtype mapping.
    if bl in ["organism-based format", "whole organism format"]:
        return {
            "B": "In vivo (binding)",
            "F": "In vivo (functional)",
            "A": "In vivo (ADME)",
            "T": "In vivo (toxicity)",
        }.get(at, "In vivo (other)")

    # Non-mammalian whole-organism assays.
    if bl in ["non-mammalian organism format"]:
        return "In vivo (non-mammalian)"

    # Biophysical measurements (e.g., SPR, ITC).
    if bl in ["biophysical format"]:
        return "Biophysical"

    # Generic/unspecified “assay format” with a light subtype mapping.
    if bl in ["assay format"]:
        return {
            "B": "Binding (unspecified)",
            "F": "Functional (unspecified)",
            "A": "ADME (unspecified)",
            "T": "Toxicity (unspecified)",
            "P": "Physicochemical (unspecified)",
            "U": "Unknown"
        }.get(at, "Unknown")

    # Fallback class when mapping is not possible.
    return "N/A"
