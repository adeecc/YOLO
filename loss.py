import torch
import torch.nn as nn

from utils import intersection_over_union

# TODO: Fix inconsistency in flattens


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()

        self.mse = nn.MSELoss(
            reduction="sum"
        )

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1,
                                          self.S, self.S, self.C + self.B * 5)

        # 0-19 are class probabilites, 20 is class score, 21-24 is bounding box coords
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25])

        ious = torch.cat([
            iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)
        ], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)

        # Iobj_i
        # alternatively use target[..., 20:21]
        exists_box = target[..., 20].unsqueeze(3)

        # For Box Coordinates
        box_predictions = exists_box * (
            (bestbox * predictions[..., 26:30] +
             (1 - bestbox) * predictions[..., 21:25])
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4] * torch.sqrt(
            # Epsilon added for numerical stability
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        ))

        box_targets[..., 2:4] = torch.sqrt(
            box_targets[..., 2:4])  # Why not add epsilon here?

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # For object Loss

        pred_box = (
            # Could use unqueeze as well
            bestbox * predictions[..., 25:26] + \
            (1 - bestbox) * predictions[..., 20:21]
        )

        # (N, S, S, 1) -> (N * S * S)
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # For no Object Loss

        # (N, S, S, 1) -> (N, S * S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) *
                          predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) *
                          predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # For class Loss

        # (N, S, S, 20) -> (N * S * S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss  # First Two rows of loss in paper
            + obj_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
