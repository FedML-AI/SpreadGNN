import numpy as np
import scipy.sparse as sp
import torch
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler
from rdkit import Chem

# Seeds are all the nodes in the graph given small size
def random_walks_with_fanout(adjacency_matrix: torch.Tensor, fanouts=[2, 2]):
    if len(fanouts) != 2:
        raise ValueError('Fanouts must be a 2 dimensional list for a 2-hop graphsage network')

    seed_nodes = np.arange(adjacency_matrix.shape[0])

    # "Seed" the forest with the seed nodes
    forest = [seed_nodes.reshape(adjacency_matrix.shape[0], 1)]
    current_depth_nodes = torch.from_numpy(seed_nodes)
    view_transform = 1
    for fanout in fanouts:
        view_transform *= fanout
        next_depth_nodes = Categorical(adjacency_matrix[current_depth_nodes]).sample((fanout,)).T
        forest.append(next_depth_nodes.view(-1, view_transform).numpy())
        current_depth_nodes = next_depth_nodes.flatten()

    return forest


def generate_scaffolds(path, include_chirality = False):

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, MurckoScaffoldSmilesFromSmiles
        from .data_loader import get_smiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")
    
    scaffolds = {}
    smiles = get_smiles(path)
    data_len = len(smiles)

    #logger.info("About to generate scaffolds")
    for  idx, smile in enumerate(smiles):
        # mol = Chem.MolFromSmiles(smile)
        # scaffold = MurckoScaffoldSmiles(mol= mol,includeChirality=include_chirality)
        # mol =Chem.MolFromSmiles(smile,sanitize=False) 
        # mol.UpdatePropertyCache(strict=False)
        # Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        # scaffold = MurckoScaffoldSmiles(mol= mol, includeChirality=include_chirality)
        # if smile == '':
        #     print('empty SMILES')
        # mol = Chem.MolFromSmiles(smile)
        

        # if mol is not None:
        #     print(smile)
        #     scaffold = MurckoScaffoldSmiles(mol= mol,
        #                                     includeChirality=include_chirality)
        # else:
        #     mol =Chem.MolFromSmiles(smile,sanitize=False) 
        #     print('here none mol')
        #     if mol.GetNumHeavyAtoms() == 0:
        #         print('invalid heavy')
        #     # if mol.NeedsUpdatePropertyCache():
        #     #     print('here update')
        #     #     mol.UpdatePropertyCache()
        #     Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        #     if mol.NeedsUpdatePropertyCache():
        #         print('here update 2')
        #         print()
        #         mol.UpdatePropertyCache(strict = False)
        #     else: 
        #         print("WARNING")
        #         mol.UpdatePropertyCache(strict = False)
        #     scaffold = MurckoScaffoldSmiles(mol= mol, includeChirality=include_chirality)
        # try:
        #    
        # except ValueError:
        #     mol = Chem.MolFromSmiles(smile,sanitize=False) 
        #     mol.UpdatePropertyCache(strict=False)
        #     Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        #     print(mol)
        #     scaffold = MurckoScaffoldSmiles(mol= mol, includeChirality=include_chirality)
        if smile != '':
            scaffold = MurckoScaffoldSmilesFromSmiles(smile,includeChirality=include_chirality)


        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

        # Sort from largest to smallest scaffold sets
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]
    return scaffold_sets

class WalkForestCollator(object):
    def __init__(self, normalize_features=False, normalize_adj=True, mask_type = 0, client_mask = None, total_cli = None, client_idx = None):
        self.normalize_features = normalize_features
        self.normalize_adj = normalize_adj
        self.mask_type = mask_type
        self.client_idx = client_idx
        self.total_cli = total_cli
        self.client_mask  = client_mask

    def __call__(self, molecule):
        adj_matrix, feature_matrix, label, fanouts = molecule[0]
        torch_adj_matrix = torch.from_numpy(np.array(adj_matrix.todense()))
        forest = random_walks_with_fanout(torch_adj_matrix, fanouts)
        torch_forest = [torch.from_numpy(forest[0]).flatten()]

        for i in range(len(forest) - 1):
            torch_forest.append(torch.from_numpy(forest[i + 1]).reshape(-1, fanouts[i]))

        
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)
        if self.mask_type == 1:
            cli_mask = torch.where(self.client_mask  == self.client_idx , 1.0, 0.0)
        if self.mask_type == 2:
            #The worst case
            task_assign = torch.randint(low=0, high = self.total_cli, size =(1, label.shape[0] ))
            while torch.unique(task_assign).shape[0] != self.total_cli:
                task_assign = torch.randint(low=0, high = self.total_cli, size =(1, label.shape[0] ))
            cli_mask = torch.where(task_assign  == self.client_idx , 1.0, 0.0)
            

        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        if self.mask_type != 0:
            return torch_forest, torch.as_tensor(normalized_feature_matrix, dtype=torch.float32), torch.as_tensor(label, dtype=torch.float32), \
               torch.as_tensor(mask, dtype=torch.float32) , cli_mask
        else:
            #No masking
            return torch_forest, torch.as_tensor(normalized_feature_matrix, dtype=torch.float32), torch.as_tensor(label, dtype=torch.float32), \
               torch.as_tensor(mask, dtype=torch.float32) , None


class DefaultCollator(object):
    def __init__(self, normalize_features=True, normalize_adj=True, mask_type = 0, client_mask = None, total_cli = None, client_idx = None):
        self.normalize_features = normalize_features
        self.normalize_adj = normalize_adj
        self.mask_type = mask_type
        self.client_idx = client_idx
        self.total_cli = total_cli
        self.client_mask  = client_mask

    def __call__(self, molecule):
        adj_matrix, feature_matrix, label, _ = molecule[0]
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)
        if self.mask_type == 1:
            cli_mask = torch.where(self.client_mask  == self.client_idx , 1.0, 0.0)
        if self.mask_type == 2:
            task_assign = torch.randint(low=0, high = self.total_cli, size =(1, label.shape[0] ))
            while torch.unique(task_assign).shape[0] != self.total_cli:
                task_assign = torch.randint(low=0, high = self.total_cli, size =(1, label.shape[0] ))
            cli_mask = torch.where(task_assign  == self.client_idx , 1.0, 0.0)
           
        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        if self.normalize_adj:
            rowsum = np.array(adj_matrix.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            normalized_adj_matrix = adj_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
        else:
            normalized_adj_matrix = adj_matrix

        if self.mask_type != 0:
            return torch.as_tensor(np.array(normalized_adj_matrix.todense()), dtype=torch.float32), \
                torch.as_tensor(normalized_feature_matrix, dtype=torch.float32), \
                torch.as_tensor(label, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.float32), cli_mask
        else:
            #No masking
            return torch.as_tensor(np.array(normalized_adj_matrix.todense()), dtype=torch.float32), \
                torch.as_tensor(normalized_feature_matrix, dtype=torch.float32), \
                torch.as_tensor(label, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.float32) , None