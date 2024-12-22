import argparse
import yaml
import glob
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from npy_append_array import NpyAppendArray

from utils.seq2seq.dataset import InferDataset
from utils.seq2seq.tools import to_device
from utils.seq2seq.model.modules import get_model
    
def synthesize(model, step, config, batchs, outsavepath):
    feat_path = os.path.join(outsavepath,"features")
    os.makedirs(feat_path, exist_ok=True)
    for batch in batchs:
        batch = to_device(batch, device)
        model.test()
        with torch.no_grad():
            basename = batch[0][0]
            output = model(*(batch[1:]))
            np.save(os.path.join(feat_path,basename), output.to('cpu').numpy())
    print(f"Extracted features at {feat_path}")
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to config.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (e.g., 'cuda', 'cpu', or GPU index like '1')",
    )
    
    args = parser.parse_args()

    # Read Config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    result_path = os.path.join(config["path"]["root_path"],"results")
    os.makedirs(result_path,exist_ok=True)
    outsavepath = os.path.join(result_path,str(args.restore_step))
    nam_valid_wav_files = os.path.join(config["path"]["nam_wav_files"])
    
    # Get validation file
    source = os.path.join(config["path"]["root_path"],'val.txt')
    print(f"Reading validation files from: {source}")
    
    # Set device
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(f"Using device: {device}")

    # Get model
    model = get_model(args, config, device, train=False)
    print(f"Got StethoSpeech Model")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of StethoSpeech Parameters:", count_parameters(model))

    # Get dataset
    dataset = InferDataset(source, config)
    batchs = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate_fn,
    )

    # Extract StethoSpeech features
    os.makedirs(outsavepath, exist_ok=True)
    synthesize(model, args.restore_step, config, batchs, outsavepath)
    
    print(f"Now creating files necessary for quantization")
    # Create preprocessing files for mapping features to units
    files = sorted(glob.glob(outsavepath + '/features/*.npy'))
    lenfile = open(outsavepath + '/test_0_1.len','w')
    featurefile = NpyAppendArray(outsavepath + '/test_0_1.npy')
    tsvfile = open(outsavepath + '/test.tsv','w')
    tsvfile.write(nam_valid_wav_files + '\n')

    for f in files:
        tsvfile.write(f.split('/')[-1].replace('.npy','.wav').replace('_nam','_headset')+'\t0\n')
        val = np.load(f)
        len_ = len(val)
        lenfile.write(f"{len_}\n")
        featurefile.append(val)