import argparse
from pathlib import Path
import os
from tqdm import tqdm
import yaml
import random

# python utils/parse_hubert_codes.py --restore_step 20

def parse_manifest(manifest):
    audio_files = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files

def save(outdir, samples):
    # save
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / 'all_samples.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in samples]))

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step")
    parser.add_argument("-c","--config",type=str,required=True,help="path to config.yaml",)
    parser.add_argument('--ref-train', type=Path)
    parser.add_argument('--ref-val', type=Path)
    parser.add_argument('--ref-test', type=Path)
    args = parser.parse_args()

    # Load configuration from the provided YAML file
    config = load_config(args.config)
    step = args.restore_step
    
    result_path = os.path.join(config["path"]["root_path"],"results")
    
    manifest = os.path.join(result_path, step, "test.tsv")
    with open(manifest) as f:
        fnames = [l.strip() for l in f.readlines()]
    wav_root = Path(fnames[0])
    fnames = fnames[1:]

    codes = os.path.join(result_path, step, "labels", "test_0_1.km")
    with open(codes) as f:
        codes = [l.strip() for l in f.readlines()]

    # parse
    samples = []
    for fname_dur, code in tqdm(zip(fnames, codes)):
        sample = {}
        fname, dur = fname_dur.split('\t')

        sample['audio'] = str(wav_root / f'{fname}')
        sample['hubert'] = ' '.join(code.split(' '))
        sample['duration'] = int(dur) / 16000

        samples += [sample]

    # Save all samples into one file
    outdir = Path(os.path.join(result_path, step, "parsed_hubert"))
    print(f"Created file at: {outdir}")
    save(outdir, samples)

if __name__ == '__main__':
    main()
