import torch 
import torch.nn.functional as F 


class YoloV1Loss(torch.nn.Module): 
    def __init__(self, **kwargs): 
        super().__init__() 
        self.lambda_coord = 5 
        self.lambda_noobj = 0.5         
        self.__dict__.update(kwargs)

    def forward(self, outputs, targets):
        # Get masks for grid cells with and without objects
        obj_mask = (targets[..., 4] == 1).detach()
        noobj_mask = torch.logical_not(obj_mask).detach()

        loss_xywh = self.lambda_coord * F.mse_loss(outputs[..., 0:4][obj_mask], targets[..., 0:4][obj_mask], reduction="sum")
        loss_b =  F.mse_loss(outputs[..., 4][obj_mask], targets[..., 4][obj_mask], reduction="sum") 
        loss_no_b = self.lambda_noobj * F.mse_loss(outputs[..., 4][noobj_mask], targets[..., 4][noobj_mask], reduction="sum")

        # Total loss
        loss = loss_xywh + loss_b + loss_no_b

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