import torch 
import torch.nn.functional as F 


class YoloV1Loss(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.lambda_coord = 5 
        self.lambda_noobj = 0.5         
    
    def forward(self, outputs, targets):
        """
        Arguments:
        outputs: (batch_size, 7, 7, 5)
        targets: (batch_size, 7, 7, 5)

        Returns:
        loss: scalar
        """
        # Get masks for grid cells with and without objects
        obj_mask = (targets[:, :, :, 4] == 1).detach()
        noobj_mask = torch.logical_not(obj_mask).detach()

        loss1a = F.mse_loss(outputs[..., 0][obj_mask], targets[..., 0][obj_mask], reduction="sum")
        loss1b = F.mse_loss(outputs[..., 1][obj_mask], targets[..., 1][obj_mask], reduction="sum")
        loss2a = F.mse_loss(outputs[..., 2][obj_mask], targets[..., 2][obj_mask], reduction="sum")
        loss2b = F.mse_loss(outputs[..., 3][obj_mask], targets[..., 3][obj_mask], reduction="sum")
        loss3 = F.mse_loss(outputs[..., 4][obj_mask], targets[..., 4][obj_mask], reduction="sum")
        loss4 = F.mse_loss(outputs[..., 4][noobj_mask], targets[..., 4][noobj_mask], reduction="sum")

        # Total loss
        loss = self.lambda_coord * (loss1a + loss1b + loss2a + loss2b) + loss3 + self.lambda_noobj * loss4

        return loss
    

        
if __name__=="__main__":
    from data_utils import get_data_loaders
    from model import YoloV1Model
    
    mod = YoloV1Model()
    print(f"Model: {mod}")
    
    wider_train_loader, _ = get_data_loaders(batch_size=4)
    print(f"Number of train batches: {len(wider_train_loader):,.0f}")

    train_img_batch, train_label_batch = next(iter(wider_train_loader))
    print(f"Images shape: {train_img_batch.shape}, Labels shape: {train_label_batch.shape}")

    train_pred_batch = mod(train_img_batch)
    print(f"Pred labels shape: {train_pred_batch.shape}")

    fake_pred_batch = torch.rand(train_pred_batch.size())
    print(f"Fake pred labels size: {fake_pred_batch.shape}")

    loss_fn = YoloV1Loss()
    loss = loss_fn(train_pred_batch, train_label_batch)
    print(f"Loss: {loss:.4f}")

    loss = loss_fn(fake_pred_batch, train_label_batch)
    print(f"Loss: {loss:.4f}")