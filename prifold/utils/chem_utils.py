import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from scipy.spatial.distance import cosine as cos_distance
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
import scipy.sparse
from functools import partial

from prifold.dataset.molgen.tokenizer import tokenizer

@torch.no_grad()
def estimate_sampled_smiles(tok, model, device, sample_num = 100, n_jobs=100, train=None, top_k = None):
    '''

    '''

    x = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    infer_times = 10
    bs = sample_num // infer_times
    x = x.expand(bs, -1)

    max_new_tokens = 128
    temperature = 1.0
    smiles_list = []
    for _ in tqdm(range(infer_times)):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        str_list = tok.decode_batch(y.tolist())
        smiles_list.extend(str_list)

    smiles_list = [s.replace('Ġ','').replace(' ','') for s in smiles_list]

    # save the sampled smiles
    with open('sampled_smiles.txt', 'w') as f:
        for smile in smiles_list:
            f.write(smile + '\n')

    # get the metrics, moses benchmark
    return get_metrics(smiles_list, k = [1000], n_jobs=n_jobs, device='cpu', pool=None, train=train), smiles_list

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args,
                 **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)

    fps = mapper(n_jobs)(
        partial(fingerprint, *args, **kwargs), smiles_mols_array
    )

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps

def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint

def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')


def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def is_valid_smiles(smiles):
    """
    判断给定的 SMILES 符号是否合法。

    参数:
        smiles (str): 要检查的 SMILES 符号。

    返回:
        bool: 如果 SMILES 符号合法，则返回 True，否则返回 False。

        # # 示例
        # smiles = "CCO"  # 这是乙醇的 SMILES 符号
        # print(is_valid_smiles(smiles))
    """
    molecule = Chem.MolFromSmiles(smiles)
    return molecule is not None

def draw_smiles_to_png(smiles_list, filename):
    """
    将给定的 SMILES 符号列表绘制为分子结构并保存为 PNG 图像。

    参数:
        smiles_list (list): 要绘制的 SMILES 符号列表。
        filename (str): 保存图像的文件名。
    """
    # 将 SMILES 符号转换为分子对象列表
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    # 绘制分子结构
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(500, 500))

    # 保存为 PNG 图像
    img.save(filename)


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


def get_metrics(gen, k=None, n_jobs=1, device='cpu', pool=None,train=None):
    """
    Computes all available metrics between test (scaffold test)
    and generated sets of SMILES.
    Parameters:
        gen: list of generated SMILES
        k: int or list with values for unique@k. Will calculate number of
            unique molecules in the first k molecules. Default [1000, 10000]
        n_jobs: number of workers for parallel processing
        device: 'cpu' or 'cuda:n', where n is GPU device number
        batch_size: batch size for FCD metric
        pool: optional multiprocessing pool to use for parallelization

        test (None or list): test SMILES. If None, will load
            a default test set
        test_scaffolds (None or list): scaffold test SMILES. If None, will
            load a default scaffold test set
        ptest (None or dict): precalculated statistics of the test set. If
            None, will load default test statistics. If you specified a custom
            test set, default test statistics will be ignored
        ptest_scaffolds (None or dict): precalculated statistics of the
            scaffold test set If None, will load default scaffold test
            statistics. If you specified a custom test set, default test
            statistics will be ignored
        train (None or list): train SMILES. If None, will load a default
            train set
    Available metrics:
        * %valid
        * %unique@k
        * Internal diversity (IntDiv)
        * Internal diversity 2: using square root of mean squared
            Tanimoto similarity (IntDiv2)
        * Novelty (molecules not present in train)
    """


    if k is None:
        k = [1000, 10000]

    disable_rdkit_log()
    metrics = {}
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1

    metrics['valid'] = fraction_valid(gen, n_jobs=pool)
    gen = remove_invalid(gen, canonize=True)

    if len(gen) > 0:
        if not isinstance(k, (list, tuple)):
            k = [k]
        for _k in k:
            metrics['unique@{}'.format(_k)] = fraction_unique(gen, _k, pool)

        mols = mapper(pool)(get_mol, gen)
        metrics['IntDiv'] = internal_diversity(mols, pool, device=device)
        metrics['IntDiv2'] = internal_diversity(mols, pool, device=device, p=2)

        if train is not None:
            metrics['Novelty'] = novelty(mols, train, pool)

    enable_rdkit_log()
    if close_pool:
        pool.close()
        pool.join()

    return metrics


if __name__ == '__main__':
    import time
    # test the speed of the fingerprint function
    start = time.time()
    gen = ['CCCCCCC'] * 10000
    train = gen
    metrics = get_metrics(gen, k = [1000], n_jobs=100, device='cpu', pool=None, train=None)
    print(metrics)
    end = time.time()
    print(end - start)

