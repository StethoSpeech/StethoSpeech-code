# StethoSpeech-code

This is the official code for the paper titled:
**"StethoSpeech: Speech Generation Through Stethoscopic Microphone Attached To The Skin"**, accepted to the *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies Vol. 8, No. 3*.

Ground-truth speech simulation is straightforward. Below are the steps that require collaboration with several repositories but are omitted here for simplicity:

1. **Extract Duration Alignments**:  
   Use input non-audible murmurs (NAMs) and their corresponding text files to train an ASR engine using [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).

2. **Generate Aligned Speech**:  
   Use any state-of-the-art Text-to-Speech (TTS) model. Pass the text and extracted durations from the aligner to the pre-trained TTS to generate aligned speech with NAMs in a voice of interest. For the paper, we relied on the [FastSpeech2 TTS model](https://github.com/ming024/FastSpeech2).

---

### Step 1: Download vocoder checkpoint and k-means checkpoint:

```bash
python utils/download.py
```

### Step 2: Compute training and validation files, ASR tokens, and vocabulary dictionary:

To get started, update the following entries in `utils/seq2seq/config.yaml`:

- `root_path`: The root directory where all processes are executed.
- `nam_hubert_path`: The directory containing HuBERT features extracted from NAM audio.
- `nam_wav_files`: The directory containing NAM audio files.
- `simulated_speech_hubert_path`: The directory containing simulated speech, used as training targets for sequence-to-sequence tasks.

After updating the configuration, run the following script to preprocess the data:  

```bash
python utils/preprocess.py
```

The above code produces the following files and directories at `root_path`:

- `train.txt`
- `val.txt`
- `vocab_character.json`
- `ASR_tokens_character`

Modify `ctc_vocab_size` at `utils/seq2seq/config.yaml` with the vocabulary size obtained from Step 1.

### Step 3: Train a sequence-to-sequence StethoSpeech architecture:

Training script to train our sequence-to-sequence model. The script produces the `ckpt` and `logs` folder at `root_path`

```bash
export PYTHONPATH=/path/to/your/StethoSpeech-code
python utils/seq2seq/train.py -c utils/seq2seq/config.yaml --device 0
```

### Step 4: Perform inference on `val.txt` using a trained sequence-to-sequence model:

Inference script to predict speech HuBERT features from input NAM HuBERT features. The script produces at <root_path>/<results>
```bash
python utils/seq2seq/synthesize.py -c utils/seq2seq/config.yaml  --restore_step <Path-to-checkpoint> --device 0
```

### Step 5: Discretize the predicted speech HuBERT features using a K-Means model:

We relied on a unit-based vocoder to generate speech. Therefore, the following steps are needed to discretize the predicted features. 

```bash
python utils/dump_km_label.py --km_path pretrained_ckpt/km.bin --restore_step <Path-to-checkpoint>
python utils/parse_hubert_codes.py -c utils/seq2seq/config.yaml --restore_step <Path-to-checkpoint>
```

### Step 6: Infer using LJSpeech trained vocoder on discretized speech codes:

- Infer the vocoder on predictions from the seq2seq module. The converted speech is available at `generations` folder.
```bash
python utils/vocoder/inference.py --checkpoint_file pretrained_ckpt -n <Path-to-checkpoint> --vc --input_code_file /media/newhd/Neil/fairseq/AR_aug/stethospeech/results/<Path-to-checkpoint>/parsed_hubert/all_samples.txt --output_dir generations
```

## Acknowledgements

This repository is developed using insights from:
- [ParrotTTS](https://github.com/parrot-tts/Parrot-TTS)
- [Speech Resynthesis by Facebook Research](https://github.com/facebookresearch/speech-resynthesis)
- [FastSpeech2 by Ming024](https://github.com/ming024/FastSpeech2)