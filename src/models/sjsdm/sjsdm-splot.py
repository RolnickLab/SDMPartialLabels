import sjSDM_py as fa
import numpy as np
import torch
import argparse
import json
from sklearn import metrics
import os
from tqdm import tqdm
import time


def mae(target, pred):
    return(np.mean(np.abs(target-pred)))

def iterate_in_batches(Env, Occ, batch_size):
    """
    Yields batches (slices) of a NumPy array.
    
    :param arr: The input NumPy array (must be 2D).
    :param batch_size: The number of rows in each batch.
    :yield: A NumPy array slice for the current batch.
    """
    num_rows = Env.shape[0]
    
    # Iterate through the array in steps of batch_size
    for i in tqdm(range(0, num_rows, batch_size)):
        # Slice the array from index i up to i + batch_size
        if i + batch_size < num_rows:
            yield (Env[i: i + batch_size], Occ[i: i + batch_size])

def merge_arrays(array_list, save_path):
    """
    :param array_list: list of paths to arrays to be merged 
    :param save_path: .npy path
    """
    arrs = []
    for elem in array_lost:
        arr = np.load(elem)
        arrs.append(arr)
    array =np.concatenate(arrs)
    np.save(save_path, array)
    
    
def main(seed_value,
        lr=0.001,
        batch_size=24,
        epochs=50,
        num_env=27):

    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #torch.cuda.set_device(0)#torch.device("cuda")
        torch.cuda.manual_seed_all(seed_value)
        print("GPU is available and will be used.")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU.")
    
    train_indices = np.load("data/sPlotOpen/train_indices.npy")
    test_indices = np.load("data/sPlotOpen/test_indices.npy")
    
    Env = np.load("data/sPlotOpen/env.npy")
    Occ = np.load("data/sPlotOpen/merged_species_occurrences_v2.npy")
    
    #only keep species with more than 100 occurrences
    species_indices = np.where(
            Occ.sum(axis=0) >= 100
        )[0]
    
 
    subset = species_indices
    model = fa.Model_sjSDM(device=device, dtype=torch.float32)
    model.add_env(27, len(subset))
    model.build(len(subset), optimizer=fa.optimizer_adamax(lr),scheduler=False)
    print("fit model")
    Occ_train =  Occ[train_indices, :]
    model.fit(Env[train_indices, :], Occ_train[ :, subset], batch_size = batch_size, epochs = epochs)
    
    #subnontrees ,subtrees
    Env_test = Env[test_indices, :]
    Y = Occ[test_indices, :]
    Y = Y[:, subset]
    
    #get trees and non trees indices
    csv = pd.read_csv("data/sPlotOpen/species_merge_duplicates_v2.csv")
    csv = csv.iloc[subset]
    nontrees = np.where(~csv.isTree==True)[0]
    trees = np.where(csv.isTree==True)[0]
    
    print(len(trees), len(nontrees))
    os.makedirs(f"/save_dir/sjsdm/predictions_splotopenall/unc_{lr}_{seed_value}/", exist_ok = True)
    os.makedirs(f"/save_dir/sjsdm/predictions_splotopenall/condtrees_{lr}_{seed_value}/", exist_ok = True)
    os.makedirs(f"/save_dir/sjsdm/predictions_splotopenall/condnontrees_{lr}_{seed_value}/", exist_ok = True)
    uncond_time = []
    condtrees_time = []
    condnontrees_time = []
    
    #model.predict() already processes predictions in batches, however, we ran into out-of-memory errors when trying to generate predictions in a single array on the whole dataset
    with torch.no_grad():
        for i, (env_batch, occ_batch) in enumerate(iterate_in_batches(Env_test, Y, 24)):
            print(f"\nProcessing Batch #{i + 1} (Size: {len(env_batch)}):")
            start = time.time()
            preds = model.predict(env_batch)
            t = time.time()-start
            uncond_time.append(t)
            print("uncond", t)
            np.save(f"/save_dir/sjsdm/predictions_splotopenall/unc_{lr}_{seed_value}/batch_{i}.npy", preds)
            
        print("uncond_time", len(uncond_time), np.mean(uncond_time))
        for i, (env_batch, occ_batch) in enumerate(iterate_in_batches(Env_test, Y, 24)):
            print(f"\nProcessing Batch #{i + 1} (Size: {len(env_batch)}):")

            Y_masktrees = occ_batch.astype(np.float64).copy()
            Y_masktrees[:,trees] = np.nan
            Y_masknontrees = occ_batch.astype(np.float64).copy()
            Y_masknontrees[:,nontrees] = np.nan
    
            start = time.time()
            preds_trees= model.predict(env_batch, Y=Y_masktrees)
            t = time.time()-start
            print("cond", t)
            condtrees_time.append(t) 
            
            start = time.time()
            preds_nontrees = model.predict(env_batch, Y=Y_masknontrees)
            t = time.time()-start
            condnontrees_time.append(t) 
            np.save(f"/save_dir/sjsdm/predictions_splotopenall/condtrees_{lr}_{seed_value}/batch_{i}.npy", preds_trees)
            np.save(f"/save_dir/sjsdm/predictions_splotopenall/condnontrees_{lr}_{seed_value}/batch_{i}.npy", preds_nontrees)
        print("condtree_time", len(condtrees_time), np.mean(condtrees_time)) 
        print("condnontree_time", len(condnontrees_time), np.mean(condnontrees_time)) 

        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="A script to demonstrate argparse for common ML parameters.",
        formatter_class=argparse.RawTextHelpFormatter
    )
  
    # 2. Add the --lr (learning rate) argument
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='The learning rate for the optimization algorithm. (Default: 0.001)'
    )

    # 3. Add the --batchsize argument
    # Note the use of 'dest' to map to a standard variable name (batch_size)
    parser.add_argument(
        '--batchsize',
        type=int,
        default=12,
        help='The size of the data batches for training. (Default: 32)'
    )
    
    parser.add_argument(
        '--seedvalue',
        type=int,
        default=42
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50
    )


    # 4. Parse the arguments
    args = parser.parse_args()

    
    main(args.seedvalue, args.lr, args.batchsize, args.epochs)
