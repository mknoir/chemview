import streamlit as st

from stmol import showmol

import py3Dmol

import pandas as pd

 

from rdkit import Chem

from rdkit.Chem import AllChem

import pubchempy as pcp

 

tab1, tab2 = st.tabs(["SMILES finder", "3D Molecule Viewer"])

 

def makeblock(smi):

    mol = Chem.MolFromSmiles(smi)

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)

    mblock = Chem.MolToMolBlock(mol)

    return mblock

 

def render_mol(xyz):

    xyzview = py3Dmol.view()#(width=400,height=400)

    xyzview.addModel(xyz,'mol')

    xyzview.setStyle({'stick':{}})

    xyzview.setBackgroundColor('black')

    xyzview.zoomTo()

    showmol(xyzview,height=500,width=500)

 

with tab1:

    # Ask the user for a keyword

    st.title("Chemical Search")

    st.subheader("Enter a keyword to search for a chemical.")

    keyword = st.text_input("Enter keyword:", "Aspirin")

 

    # Search for compounds matching the keyword

    compounds = pcp.get_compounds(keyword, 'name')

 

    # Check if any compounds were found

    if compounds:

        # Get the first compound

        compound = compounds[0]

        smile = compound.canonical_smiles

 

        # Print the name and SMILES

        st.write(f"Name: {compound.synonyms[0] if compound.synonyms else 'No name available'}")

        st.write(f"SMILES: {compound.canonical_smiles}")

        #show number of compounds found

        st.write(f"Number of compounds found: {len(compounds)}")

    else:

        st.warning("No compounds found for that keyword.")

    #Add a button to generate a table of the compounds

    st.subheader("Generate Table of all related compounds and smiles.")

    if st.button("Generate Table"):

        #get the names of all compounds returned

        names = [compound.synonyms if compound.synonyms else 'No name available' for compound in compounds]

        #get the smiles of all compounds returned

        smiles = [compound.canonical_smiles for compound in compounds]

        #create a dictionary of the names and smiles

        data = {'Name': names, 'SMILES': smiles}

        #create a dataframe from the dictionary

        compounds = pd.DataFrame(data)

        #display the dataframe

 

        st.dataframe(compounds)

 

with tab2:

    st.title("3D Molecule Viewer")

    st.subheader("Your search result is already visualized, however you may choose to input your own SMILES to visualize.")

 

    compound_smiles=st.text_input('Add Smiles here', f'{smile}')

    blk=makeblock(compound_smiles)

    render_mol(blk)