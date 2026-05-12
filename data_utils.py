import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Batch
from torch_geometric.data import Data

from chemutils import get_mol, get_clique_mol, tree_decomp, get_smiles, set_atommap, enum_assemble
import copy


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        try:
            mol_graph = MolGraph(smiles)
            mol_tree = MolTree(smiles)
            mol_tree.recover()
            mol_tree.assemble()
        except Exception as e:
            print("Error SMILES:", smiles)
            raise e
        return mol_graph, mol_tree


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_inring': [None, False, True]
}


class MolGraph(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        '''
        #Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)
        '''
        atom_features_list = []
        for atom in self.mol.GetAtoms():
            atom_feature = [allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [allowable_features[
                                             'possible_degree_list'].index(atom.GetDegree())]

            atom_features_list.append(atom_feature)
        self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(self.mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in self.mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                    bond.IsInRing())]

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            self.edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            self.edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                                  dtype=torch.long)
        else:
            self.edge_index_nosuper = torch.empty((2, 0), dtype=torch.long)
            self.edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

        # add super node
        num_atoms = self.x_nosuper.size(0)
        super_x = torch.tensor([[119, 0]]).to(self.x_nosuper.device)

        # add motif
        cliques, _ = motif_decomp(self.mol)
        num_motif = len(cliques)
        if num_motif > 0:
            motif_x = torch.tensor([[120, 0]]).repeat_interleave(num_motif, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, motif_x, super_x), dim=0)

            motif_edge_index = []
            for k, motif in enumerate(cliques):
                motif_edge_index = motif_edge_index + [[i, num_atoms + k] for i in motif]
            motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)

            super_edge_index = [[num_atoms + i, num_atoms + num_motif] for i in range(num_motif)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

            motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 2)
            motif_edge_attr[:, 0] = 6  # bond type for self-loop edge
            motif_edge_attr = motif_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)

            super_edge_attr = torch.zeros(num_motif, 2)
            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim=0)

            self.num_part = (num_atoms, num_motif, 1)

        else:
            self.x = torch.cat((self.x_nosuper, super_x), dim=0)

            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

            super_edge_attr = torch.zeros(num_atoms, 2)
            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim=0)

            self.num_part = (num_atoms, 0, 1)

    def size_node(self):
        return self.x.size()[0]

    def size_edge(self):
        return self.edge_attr.size()[0]

    def size_atom(self):
        return self.x_nosuper.size()[0]

    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]


def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


def moltree_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        graph_data_batch.append(mol_to_graph_data_obj_simple(mol.mol))
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # Select the atoms at the intersection points as the motif.
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges


class MolTree(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        cliques, edges = motif_decomp(self.mol)
        if len(edges) <= 1:
            cliques, edges = tree_decomp(self.mol)
        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0:
                root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()


class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique]  # copy
        self.neighbors = []

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                # allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble(self, neighbors)
        if len(cands) > 0:
            self.cands, self.cand_mols, _ = zip(*cands)
            self.cands = list(self.cands)
            self.cand_mols = list(self.cand_mols)
        else:
            self.cands = []
            self.cand_mols = []


def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)


if __name__ == "__main__":
    smiles = "O=C(NCC)c1ccccc1"
    mol = get_mol(smiles)
    cliques, edges = motif_decomp(mol)
    print(cliques)
