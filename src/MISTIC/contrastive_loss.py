import torch
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Forward pass for Contrastive Loss.

        Args:
            output1 (torch.Tensor): Feature embedding of the first sample.
            output2 (torch.Tensor): Feature embedding of the second sample.
            label (torch.Tensor): Binary label indicating relationship.
                                  0 = Same class (Similar)
                                  1 = Different class (Dissimilar)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate Euclidean distance between vectors
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        # Calculate Contrastive Loss
        # If label=0 (Similar): minimize distance (euclidean_distance^2)
        # If label=1 (Dissimilar): maximize distance up to margin (max(0, margin - distance)^2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive