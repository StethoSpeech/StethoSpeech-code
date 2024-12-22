import os
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.seq2seq.dataset import TrainDataset
from utils.seq2seq.model.modules import get_model, get_param_num
from utils.seq2seq.model.loss import StethoSpeechLoss
from utils.seq2seq.tools import to_device, log
from utils.seq2seq.evaluate import evaluate

def main(args, config):

    # Set device
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    print(f"Using device: {device}")

    # Get dataset
    dataset = TrainDataset("train.txt", config, device, sort=True, drop_last=True)
    batch_size = config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    
    # Prepare model
    model, optimizer = get_model(args, config, device, train=True)    
    model.to(device) 
    model = nn.DataParallel(model, device_ids=[device])
    num_param = get_param_num(model)
    Loss = StethoSpeechLoss().to(device)
    ckpt_path = os.path.join(config["path"]["root_path"],"ckpt")
    os.makedirs(ckpt_path,exist_ok=True)
    print(f"StethoSpeech Model: {model}")
    print(f"Number of FastSpeech2 Parameters: {num_param}")
    print(f"Checkpoints will be stored at: {ckpt_path}")
    
    
    # Init logger
    for p in config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(config["path"]["root_path"], "logs/train")
    val_log_path = os.path.join(config["path"]["root_path"], "logs/val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    
    # Training
    step = args.restore_step
    epoch = 1
    grad_acc_step = config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = config["optimizer"]["grad_clip_thresh"]
    total_step = config["step"]["total_step"]
    log_step = config["step"]["log_step"]
    save_step = config["step"]["save_step"]
    synth_step = config["step"]["synth_step"]
    val_step = config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                output = model(*(batch[1:]))
                losses = Loss(step, batch, output)
                total_loss = losses[0]
                
                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                
                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.6f},MSE Loss: {:.4f},CTC loss: {:.4f}".format(losses[0],losses[1],losses[2])
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)
                
                if step % save_step == 0:
                    torch.save({"model": model.module.state_dict(),"optimizer": optimizer._optimizer.state_dict(),},
                        os.path.join(ckpt_path,"{}.pth.tar".format(step),),)
                                    
                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, config, device, val_logger)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)   
                    model.train()
                
                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
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
    config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader
    )

    main(args, config)
