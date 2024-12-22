import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss, _Loss, MSELoss, L1Loss, CosineEmbeddingLoss, KLDivLoss
import numpy as np

class CTCLoss():
    def __init__(self) -> None:
        super().__init__()  
    def forward(self, input: torch.Tensor, target: torch.Tensor,src_lens,target_lengths):
        _loss = nn.functional.ctc_loss(
            input,
            target,
            src_lens,
            target_lengths,
        )
        return _loss
    def __call__(self, input: torch.Tensor, target: torch.Tensor,src_lens,target_lengths) -> torch.Tensor:
        return self.forward(input, target,src_lens,target_lengths)

class TransformerLoss(MSELoss): 
    def __init__(self, ignore_index=-100, reduction='mean') -> None:
        self.reduction = reduction
        self.ignore_index = ignore_index
        super().__init__(reduction='none')  
    def forward(self, input: torch.Tensor, target: torch.Tensor,trg_mask) -> torch.Tensor:
        target = target 
        mask = trg_mask==False 
        mask = mask.unsqueeze(-1).repeat(1,1,768)  
        not_masked_length = mask.to(torch.int).sum()
        _loss = super().forward(input, target) 
        _loss *= mask.to(_loss.dtype) 
        _loss = _loss.sum() / not_masked_length 
        return _loss

    def __call__(self, input: torch.Tensor, target: torch.Tensor,trg_mask) -> torch.Tensor:
        return self.forward(input, target,trg_mask)

class StethoSpeechLoss(nn.Module):
    """ StethoSpeech Loss """

    def __init__(self,ignore_index=-100, reduction='mean',sigma=0.2):
        super(StethoSpeechLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.transformer_loss = TransformerLoss()
        self.ctc_loss = CTCLoss()

    def forward(self, steps, inputs, predictions,attn=None):

        (   input_lengths,
            max_input_len,
            target,
            target_lengths,
            max_target_len,
            ctc_labels,
            ctc_labels_lens,
            max_ctc_labels_lens
        ) = inputs[2:]

        (
            preds,
            src_masks,
            trg_mask,
            src_lens,
            target_lengths,
            flattened_targets,
            log_probs,
        ) = predictions

        mseloss = self.transformer_loss(preds,target,trg_mask)
        ctcloss = self.ctc_loss(log_probs.transpose(0,1),flattened_targets,src_lens,target_lengths)
        _loss = 1 * mseloss + 0.0001 * ctcloss
 
        return (_loss,mseloss,ctcloss)
