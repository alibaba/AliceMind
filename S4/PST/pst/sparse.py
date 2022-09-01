import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseBinarizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask_scores, sparsity):
        num_prune = int(mask_scores.numel() * sparsity)
        prune_indices = torch.argsort(mask_scores.reshape(-1))[:num_prune]
        mask = mask_scores.clone().fill_(1)
        mask.reshape(-1)[prune_indices] = 0.0
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

class SparseLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        pruning_method = "pst",
        weight_rank = 8,
        weight_beta = 1.0,
        mask_rank = 8,
        mask_alpha1 = 1.0,
        mask_alpha2 = 1.0
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.pruning_method = pruning_method
        
        self.weight_rank = weight_rank
        self.weight_beta = weight_beta
        self.mask_rank = mask_rank
        self.mask_alpha1 = mask_alpha1
        self.mask_alpha2 = mask_alpha2

        self.cur_sparsity = 0.0

        if self.pruning_method == "pst":
            # create trainable params
            self.weight_U = nn.Parameter(torch.randn(out_features, self.weight_rank))
            self.weight_V = nn.Parameter(torch.zeros(self.weight_rank, in_features))
            
            self.mask_scores_A = nn.Parameter(torch.randn(out_features, self.mask_rank))
            self.mask_scores_B = nn.Parameter(torch.zeros(self.mask_rank, in_features))
            self.mask_scores_R = nn.Parameter(torch.zeros(out_features))
            self.mask_scores_C = nn.Parameter(torch.zeros(in_features))

            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, inputs):
        if self.pruning_method == "pst":
            weight = self.weight + self.weight_beta * self.weight_U @ self.weight_V
            mask_scores = weight.abs() + self.mask_alpha1 * self.mask_scores_A @ self.mask_scores_B + \
             self.mask_alpha2 * (self.mask_scores_R.unsqueeze(1) + self.mask_scores_C.unsqueeze(0))

            mask = SparseBinarizer.apply(mask_scores, self.cur_sparsity)

            masked_weight = mask * weight

            return F.linear(inputs, masked_weight, self.bias)
        else:
            return F.linear(inputs, self.weight, self.bias)
    
    def convert(self):
        if self.pruning_method == "pst":
            weight = self.weight + self.weight_beta * self.weight_U @ self.weight_V
            mask_scores = weight.abs() + self.mask_alpha1 * self.mask_scores_A @ self.mask_scores_B + \
             self.mask_alpha2 * (self.mask_scores_R.unsqueeze(1) + self.mask_scores_C.unsqueeze(0))

            mask = SparseBinarizer.apply(mask_scores, self.cur_sparsity)

            masked_weight = mask * weight

            self.old_weight = self.weight.data.clone()
            self.weight.data = masked_weight.data
    
    def restore(self):
        if self.pruning_method == "pst":
            self.weight.data = self.old_weight
            del self.old_weight