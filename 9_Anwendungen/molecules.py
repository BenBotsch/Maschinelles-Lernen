#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 08:58:30 2023

@author: ben
"""

import sys
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from rdkit import Chem 
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def paint_molecules(df):
    mols = df['mol'][:20]
    #MolsToGridImage allows to paint a number of molecules at a time
    Draw.MolsToGridImage(mols, molsPerRow=5, useSVG=True, legends=list(df['smiles'][:20].values))
    
def number_of_atoms(atom_list, df):
    """
    Computes the number of occurrences of each 
    atom in the list `atom_list` in each 
    molecule in the `df` DataFrame. 
    The number of occurrences of each atom is 
    stored in a new column named 
    'num_of_{atom}_atoms', where `{atom}`
    is the symbol of the corresponding atom.

    Parameters:
    -----------
    atom_list: list
        A list of atom symbols whose number of 
        occurrences in each molecule will be 
        computed.
    df: pandas.DataFrame
        A DataFrame containing the molecules and 
        their properties. It must have a column 
        named 'mol' that contains the molecular 
        structure in the form of a RDKit Mol object.

    Returns:
    --------
    None
    """
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))

def train(df):
    """
    Trains a Ridge regression model using the 
    data in the `df` DataFrame and evaluates 
    its performance.

    Parameters:
    -----------
    df: pandas.DataFrame
        A DataFrame containing the molecules and 
        their properties. It must have columns 
        named 'smiles', 'mol', and 'logP', where 
        'smiles' contains the SMILES string of 
        the molecule, 'mol' contains the molecular 
        structure in the form of a RDKit Mol 
        object, and 'logP' contains the logarithm 
        of the partition coefficient of the 
        molecule.

    Returns:
    --------
    None
    """
    train_df = df.drop(columns=['smiles', 'mol', 'logP'])
    y = df['logP'].values
    X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.1, random_state=1)
    ridge = RidgeCV(cv=5)
    ridge.fit(X_train, y_train)
    evaluation(ridge, X_test, y_test)
    
def evaluation(model, X_test, y_test):
    """
    Evaluates the performance of a regression 
    model by computing the mean absolute 
    error (MAE) and mean squared error (MSE)
    between the true and predicted values of 
    the logarithm of the partition coefficient 
    for a test set of molecules. It also plots 
    a graph of the first 100 true and predicted 
    values.

    Parameters:
    -----------
    model: sklearn.linear_model
        A trained regression model from the 
        scikit-learn package.
    X_test: pandas.DataFrame
        A DataFrame containing the features 
        of the test set of molecules.
    y_test: numpy.array
        An array containing the true values 
        of the logarithm of the partition 
        coefficient for the test set of 
        molecules.

    Returns:
    --------
    None
    """
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=((9*cm, 5*cm)),dpi=600)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(prediction[:50], "red", label="Vorhersage", linewidth=1.0)
    plt.plot(y_test[:50], 'green', label="wahrer Wert", linewidth=1.0)
    plt.legend(loc='lower right')
    plt.ylabel('logP')
    plt.xlabel('Samples')
    plt.savefig('image.pdf',bbox_inches='tight')
    plt.show()
    
    print('MAE score:', round(mae, 4))
    print('MSE score:', round(mse,4))

if __name__ == "__main__":
    #Let's load the data and look at them
    df= pd.read_csv('../Datasets/logP_dataset.csv', names=['smiles', 'logP'])
    df.head()

    #Method transforms smiles strings to mol rdkit object
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 
    df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))
    df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
    df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    df['tpsa'] = df['mol'].apply(lambda x: Descriptors.TPSA(x))
    df['mol_w'] = df['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    df['num_valence_electrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    df['num_heteroatoms'] = df['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    
    number_of_atoms(['C','O', 'N', 'Cl'], df)

    train(df)


