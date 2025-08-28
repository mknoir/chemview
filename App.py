import pubchempy as pcp
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcNumAromaticRings
from rdkit.Chem import SDWriter
from stmol import showmol
import py3Dmol
from io import StringIO

# Set page config for better layout
st.set_page_config(
    page_title="ChemView - Molecular Visualization",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def calculate_properties(mol):
    """Calculate molecular properties using RDKit."""
    return {
        "Molecular Formula": CalcMolFormula(mol),
        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
        "Exact Mass": f"{Descriptors.ExactMolWt(mol):.6f}",
        "Formal Charge": int(Chem.GetFormalCharge(mol)),
        "LogP": f"{Descriptors.MolLogP(mol):.2f}",
        "H-Bond Donors": int(Descriptors.NumHDonors(mol)),
        "H-Bond Acceptors": int(Descriptors.NumHAcceptors(mol)),
        "Rotatable Bonds": int(Descriptors.NumRotatableBonds(mol)),
        "Aromatic Rings": int(CalcNumAromaticRings(mol)),
        "TPSA": f"{Descriptors.TPSA(mol):.2f} Å²"
    }

def create_info_card(title, content):
    """Create a styled info card."""
    return f"""
    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin:10px 0; background-color:#fafafa;">
        <h4 style="margin:0 0 10px 0; color:#0575E6;">{title}</h4>
        <div>{content}</div>
    </div>
    """

def makeblock(smi: str):
    smi = (smi or "").strip()
    if not smi:
        raise ValueError("Empty SMILES.")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("3D embedding failed.")
    AllChem.UFFOptimizeMolecule(mol)
    return Chem.MolToMolBlock(mol)

def render_mol(mblock: str, style="stick", width=500, height=500):
    """Render molecule with customizable style."""
    v = py3Dmol.view(width=width, height=height)
    v.addModel(mblock, 'mol')
    
    style_options = {
        "stick": {'stick': {}},
        "ball_stick": {'stick': {}, 'sphere': {'scale': 0.3}},
        "sphere": {'sphere': {}},
        "line": {'line': {}},
        "surface": {'surface': {'opacity': 0.7}}
    }
    
    v.setStyle(style_options.get(style, {'stick': {}}))
    v.setBackgroundColor('white')
    v.zoomTo()
    showmol(v, height=height, width=width)

tab1, tab2 = st.tabs(["Chemical Search", "3D Visualization"])
st.session_state.setdefault("picked_smiles", "")

# Common example molecules
EXAMPLE_MOLECULES = {
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ethanol": "CCO",
    "Benzene": "C1=CC=CC=C1"
}

@st.cache_data(show_spinner=False)
def fetch_props(keyword: str):
    """Cached PubChem property lookup."""
    try:
        return pcp.get_properties(
            properties=['CanonicalSMILES','IsomericSMILES','IUPACName'],
            identifier=keyword,
            namespace='name',
            as_dataframe=True
        )
    except pcp.BadRequestError:
        return None

with tab1:
    st.title("Chemical Search")
    st.subheader("Search PubChem database by compound name")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        keyword = st.text_input("Enter compound name:", "Aspirin", help="Try: aspirin, caffeine, ibuprofen, etc.")
        cid_quick = st.text_input("Optional: PubChem CID", value="", help="Paste a CID to bypass name ambiguity")
    
    with col2:
        st.write("**Quick Examples:**")
        if st.button("Caffeine", use_container_width=True):
            st.session_state.picked_smiles = EXAMPLE_MOLECULES["Caffeine"]
        if st.button("Ibuprofen", use_container_width=True):
            st.session_state.picked_smiles = EXAMPLE_MOLECULES["Ibuprofen"]

    # Handle direct CID input
    if cid_quick.strip().isdigit():
        with st.spinner("Fetching compound by CID..."):
            try:
                c = pcp.Compound.from_cid(int(cid_quick.strip()))
                props = pd.DataFrame([{
                    'CID': c.cid,
                    'IUPACName': getattr(c, 'iupac_name', None),
                    'SMILES': getattr(c, 'canonical_smiles', None) or getattr(c, 'isomeric_smiles', None)
                }])
            except Exception as e:
                st.error(f"Failed to fetch CID {cid_quick}: {e}")
                props = pd.DataFrame()
    else:
        # Normal name-based search
        with st.spinner("Searching PubChem..."):
            props = fetch_props(keyword)
            if props is None:
                # Fallback: use get_compounds if the property query fails for a weird name
                hits = pcp.get_compounds(keyword, 'name')
                rows = []
                for c in hits:
                    rows.append({
                        'CID': getattr(c, 'cid', None),
                        'IUPACName': getattr(c, 'iupac_name', None),
                        'SMILES': getattr(c, 'canonical_smiles', None) or getattr(c, 'isomeric_smiles', None)
                    })
                props = pd.DataFrame(rows)

    if props is None or props.empty:
        st.warning("No compounds found for that keyword.")
        st.session_state.picked_smiles = ""
    else:
        # Build SMILES column
        if 'SMILES' not in props.columns:
            can = props['CanonicalSMILES'] if 'CanonicalSMILES' in props.columns else None
            iso = props['IsomericSMILES'] if 'IsomericSMILES' in props.columns else None
            if can is not None and iso is not None:
                props['SMILES'] = can.fillna(iso)
            elif can is not None:
                props['SMILES'] = can
            elif iso is not None:
                props['SMILES'] = iso
            else:
                props['SMILES'] = None

        props = props.dropna(subset=['SMILES'])
        if props.empty:
            st.warning("Found hits, but none expose SMILES. Try a more specific keyword or a CID.")
            st.session_state.picked_smiles = ""
        else:
            # Find CID-like column if present
            cid_col = next((c for c in props.columns if c.lower() == 'cid'), None)
            
            # De-duplicate safely (prefer CID+SMILES; otherwise just SMILES)
            subset_cols = ['SMILES'] + ([cid_col] if cid_col else [])
            props = props.drop_duplicates(subset=subset_cols)

            # Let user pick one
            def label_row(row):
                name = row.get('IUPACName') or (f"CID {int(row[cid_col])}" if cid_col and cid_col in row else "Unnamed")
                if cid_col and cid_col in row:
                    return f"{name} | CID {int(row[cid_col])}"
                return name

            options = list(props.itertuples(index=False))
            idx = st.selectbox(
                "Select a compound",
                options=range(len(options)),
                format_func=lambda i: label_row(dict(zip(props.columns, options[i]))),
                index=0
            )
            chosen = dict(zip(props.columns, options[idx]))
            
            # Create info card for selected compound
            card_content = f"<strong>SMILES:</strong> <code>{chosen['SMILES']}</code><br>"
            if cid_col and cid_col in chosen and pd.notna(chosen[cid_col]):
                cid_int = int(chosen[cid_col])
                card_content += f'<strong>PubChem:</strong> <a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid_int}" target="_blank">CID {cid_int}</a><br>'
            card_content += f"<strong>Name:</strong> {chosen.get('IUPACName') or (f'CID {int(chosen[cid_col])}' if cid_col and cid_col in chosen else 'Unnamed')}"
            
            st.markdown(create_info_card("Selected Compound", card_content), unsafe_allow_html=True)
            st.caption(f"{len(props)} results found; showing details for selection {idx+1}.")
            
            if st.button("Use this SMILES in Viewer"):
                st.session_state.picked_smiles = chosen['SMILES']
                st.success("Loaded into 3D Visualization tab")
            else:
                st.session_state.picked_smiles = chosen['SMILES']

        if st.button("Generate Table") and not props.empty:
            available_cols = [c for c in ['CID','IUPACName','SMILES','CanonicalSMILES','IsomericSMILES'] if c in props.columns]
            if cid_col and cid_col not in available_cols:
                available_cols.insert(0, cid_col)
            st.dataframe(props[available_cols] if available_cols else props, use_container_width=True)

with tab2:
    st.title("3D Molecular Visualization")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Molecule Viewer")
        
        # Viewer size controls
        size_col1, size_col2 = st.columns(2)
        with size_col1:
            viewer_w = st.slider("Viewer width", 400, 900, 500)
        with size_col2:
            viewer_h = st.slider("Viewer height", 400, 900, 500)
        
        # Input options
        input_col1, input_col2 = st.columns([3, 1])
        with input_col1:
            default_smiles = st.session_state.get("picked_smiles", "") or ""
            user_smiles = st.text_input("SMILES Input:", value=default_smiles, placeholder="e.g., CCO (ethanol)")
        
        with input_col2:
            style = st.selectbox("Render Style", 
                               options=["stick", "ball_stick", "sphere", "line", "surface"],
                               format_func=lambda x: x.replace("_", " & ").title())

        if user_smiles.strip():
            try:
                mol = Chem.MolFromSmiles(user_smiles)
                if mol is None:
                    st.error("Invalid SMILES structure")
                else:
                    # Auto-select ball_stick for small molecules
                    if mol.GetNumAtoms() < 20 and style == "stick":
                        style = "ball_stick"
                    
                    blk = makeblock(user_smiles)
                    render_mol(blk, style, width=viewer_w, height=viewer_h)
            except ValueError as e:
                if "3D embedding failed" in str(e):
                    st.error("3D embedding failed. Try a simpler conformer or a different molecule form (parent vs salt).")
                else:
                    st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.info("Enter a SMILES string or search for a compound in the Chemical Search tab")
    
    with col2:
        st.subheader("Molecular Properties")
        
        if user_smiles.strip():
            try:
                mol = Chem.MolFromSmiles(user_smiles)
                if mol is not None:
                    props = calculate_properties(mol)
                    
                    # Display properties in a clean table
                    prop_df = pd.DataFrame(list(props.items()), columns=["Property", "Value"])
                    st.dataframe(prop_df, use_container_width=True, hide_index=True)
                    
                    # Download options
                    st.subheader("Export Options")
                    col_mol, col_sdf = st.columns(2)
                    with col_mol:
                        st.download_button(
                            "Download MOL",
                            Chem.MolToMolBlock(mol),
                            file_name=f"molecule_{user_smiles[:10]}.mol",
                            mime="chemical/x-mdl-molfile"
                        )
                    with col_sdf:
                        # Create proper SDF format
                        sio = StringIO()
                        w = SDWriter(sio)
                        w.write(mol)
                        w.close()
                        sdf_text = sio.getvalue()
                        
                        st.download_button(
                            "Download SDF",
                            sdf_text,
                            file_name=f"molecule_{user_smiles[:10]}.sdf",
                            mime="chemical/x-mdl-sdfile"
                        )
                    
                    # Copy SMILES functionality
                    st.markdown(f"""
                    <div style="margin-top: 10px;">
                        <label>Copy SMILES:</label>
                        <input type="text" id="smi" value="{user_smiles}" style="width:100%; margin:5px 0;" readonly />
                        <button onclick="navigator.clipboard.writeText(document.getElementById('smi').value)" 
                                style="padding:5px 10px; background:#0575E6; color:white; border:none; border-radius:5px;">
                            Copy to Clipboard
                        </button>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.warning("Cannot calculate properties for invalid SMILES")
        else:
            st.info("Properties will appear here once you enter a valid SMILES")
