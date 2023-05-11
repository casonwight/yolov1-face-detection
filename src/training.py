import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from model import YoloV1Model
from loss import YoloV1Loss
from data_utils import get_data_loaders

class Trainer:
    def __init__(self, **kwargs):
        self.__dict__ = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "model": None,
            "data_root": "data/",
            "batch_size": 32,
            "n_epochs": 16,
            "val_every": 32,
            "lr": 0.001,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "scheduler_step_size": 1,
            "scheduler_gamma": 0.1,
        }

        self.__dict__.update(kwargs)
        
        if self.model is None:
            self.model = YoloV1Model()

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        self.criterion = YoloV1Loss()

        self.train_loader, self.val_loader = get_data_loaders(self.data_root, batch_size=self.batch_size)

        self.results = pd.DataFrame(columns=["i", "epoch", "batch", "acc_box", "acc_no_box", "acc_overall", "loss", "source"])

    def train(self):
        for epoch in range(self.n_epochs):
            pbar = tqdm(enumerate(self.train_loader))
            pbar.set_description(f"Epoch {epoch}")

            for i, (images, labels) in pbar:
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy in places there are not actual boxes
                box_mask = labels[:, :, :, 4] == 1
                no_box_mask = torch.logical_not(box_mask)
                acc_no_box = 100*(torch.round(outputs[no_box_mask]) == labels[no_box_mask]).float().mean().item()
                acc_box = 100*(torch.round(outputs[box_mask]) == labels[box_mask]).float().mean().item()
                overall_acc = 100*(torch.round(outputs) == labels).float().mean().item()

                # Update progress bar (and results dataframe)
                pbar.set_description(f"Epoch {epoch} | Batch: {i} | Loss: {loss.item():.2f} | Acc (box): {acc_box:.0f}% | Acc (no box): {acc_no_box:.0f}% | Acc (overall): {overall_acc:.0f}%")
                self.results = pd.concat([self.results, pd.DataFrame({
                    "i": [i],
                    "epoch": [epoch],
                    "batch": [i],
                    "acc_box": [acc_box],
                    "acc_no_box": [acc_no_box],
                    "acc_overall": [overall_acc],
                    "loss": [loss.item()],
                    "source": ["train"],
                })], ignore_index=True)

                # Validate
                if i % self.val_every == 0:
                    images_val, labels_val = next(iter(self.val_loader))
                    images_val, labels_val = images_val.to(self.device), labels_val.to(self.device)
                    outputs_val = self.model(images_val)
                    loss_val = self.criterion(outputs_val, labels_val)
                   
                    box_mask_val = labels[:, :, :, 4] == 1
                    no_box_mask_val = torch.logical_not(box_mask_val)
                    acc_no_box_val = 100*(torch.round(outputs[no_box_mask_val]) == labels[no_box_mask_val]).float().mean().item()
                    acc_box_val = 100*(torch.round(outputs[box_mask_val]) == labels[box_mask_val]).float().mean().item()
                    overall_acc_val = 100*(torch.round(outputs) == labels).float().mean().item()
                    
                    self.results = pd.concat([self.results, pd.DataFrame({
                        "i": [i],
                        "epoch": [epoch],
                        "batch": [i],
                        "acc_box": [acc_box_val],
                        "acc_no_box": [acc_no_box_val],
                        "acc_overall": [overall_acc_val],
                        "loss": [loss_val.item()],
                        "source": ["val"],
                    })], ignore_index=True)


            self.scheduler.step()
            self.validate(epoch)

    def plot_results(self):
        # Vertically stacked loss and accuracy plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(self.query("source == 'train'")["i"], self.query("source == 'train'")["loss"], label="Train")
        axs[0].plot(self.query("source == 'val'")["i"], self.query("source == 'val'")["loss"], label="Validation")
        axs[1].plot(self.query("source == 'train'")["i"], self.query("source == 'train'")["acc"], label="Train")
        axs[1].plot(self.query("source == 'val'")["i"], self.query("source == 'val'")["acc"], label="Validation")

        axs[0].set_title("Loss")
        axs[1].set_title("Accuracy")

        axs[0].legend()
        axs[1].legend()

        axs[0].set_xlabel("Batch")
        axs[1].set_xlabel("Batch")

        axs[0].set_ylabel("Loss")
        axs[1].set_ylabel("Accuracy")

        axs[0].grid()
        axs[1].grid()

        plt.show()


if __name__ == "__main__":
    trainer = Trainer(n_epochs=1)
    trainer.train()
    trainer.plot_results()