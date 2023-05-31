from torchvision.datasets import WIDERFace
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


class WiderFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.img_size = 896
        self.wider_train_data = WIDERFace(root=root, split=split, download=True)
        self.transform = transforms.Compose([Resize((self.img_size, self.img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        img, target = self.wider_train_data[index]
        
        # Images
        w, h = torch.tensor(img.size)
        img = self.transform(img)

        # Bounding boxes
        # n x 4 tensor, where n is the number of bounding boxes in the image
        bboxes_list = target['bbox']

        # For each box, getting which grid cell it belongs to
        grid_x = torch.floor(bboxes_list[:, 0] * 7 / w).to(torch.int64)
        grid_y = torch.floor(bboxes_list[:, 1] * 7 / h).to(torch.int64)
        grid_x = torch.clamp(grid_x, min=0, max=6)
        grid_y = torch.clamp(grid_y, min=0, max=6)
        grid_cells = torch.stack((grid_x, grid_y), dim=1)

        # Removing bounding boxes if there is more than one bounding box in a grid cell
        unique_grid_cells, inverse_index = torch.unique(grid_cells, dim=0, return_inverse=True)
        bboxes_list_no_duplicates = torch.zeros((unique_grid_cells.shape[0], bboxes_list.shape[1])).to(torch.float32)
        bboxes_list_no_duplicates[inverse_index] = bboxes_list.to(torch.float32)

        # Actual boxes are in grid cell format
        bboxes = torch.zeros((7, 7, 5)).to(torch.float32)
        bboxes[unique_grid_cells[:, 0], unique_grid_cells[:, 1], 0] = bboxes_list_no_duplicates[:, 0] / w
        bboxes[unique_grid_cells[:, 0], unique_grid_cells[:, 1], 1] = bboxes_list_no_duplicates[:, 1] / h
        bboxes[unique_grid_cells[:, 0], unique_grid_cells[:, 1], 2] = torch.sqrt(bboxes_list_no_duplicates[:, 2] / w)
        bboxes[unique_grid_cells[:, 0], unique_grid_cells[:, 1], 3] = torch.sqrt(bboxes_list_no_duplicates[:, 3] / h)
        bboxes[unique_grid_cells[:, 0], unique_grid_cells[:, 1], 4] = torch.ones((unique_grid_cells.shape[0])).to(torch.float32)

        return img, bboxes

    def __len__(self):
        return len(self.wider_train_data)


def get_data_loaders(root="data/", batch_size=32, batches_per_epoch=None):
    wider_train_data = WiderFaceDataset(root=root, split="train")
    wider_val_data = WiderFaceDataset(root=root, split="val")

    if batches_per_epoch is not None:
        print(f"Limiting to {batches_per_epoch:,.0f} batch{'' if batches_per_epoch == 1 else 'es'} per epoch.")
        wider_train_data = torch.utils.data.Subset(wider_train_data, range(batches_per_epoch * batch_size))
        wider_val_data = torch.utils.data.Subset(wider_val_data, range(batches_per_epoch * batch_size))

    wider_train_loader = DataLoader(wider_train_data, batch_size=batch_size, shuffle=True)
    wider_val_loader = DataLoader(wider_val_data, batch_size=batch_size, shuffle=True)

    return wider_train_loader, wider_val_loader


def show_images(images, all_boxes, nms_threshold=0.5, nms=False,
                img_w=896, img_h=896):
    max_bgr_value = 255
    bbox_color = (0, max_bgr_value, 0)

    images = (images.cpu().permute(0, 2, 3, 1).numpy() * max_bgr_value).astype(np.uint8)

    for image, boxes in zip(images, all_boxes):

        if nms:
            # Non-maximum suppression
            boxes_reshaped = boxes.reshape(-1, 5)
            non_suppressed_boxes = torchvision.ops.nms(boxes_reshaped[:, :4], boxes_reshaped[:, 4], iou_threshold=nms_threshold)
            boxes_reshaped = boxes_reshaped[non_suppressed_boxes]
            boxes = boxes_reshaped.reshape(boxes.shape)

        for grid_row in range(boxes.shape[0]):
            for grid_col in range(boxes.shape[1]):
                if boxes[grid_row, grid_col, 4]:
                    x = int(boxes[grid_row, grid_col, 0] * img_w)
                    y = int(boxes[grid_row, grid_col, 1] * img_h)
                    w = int(boxes[grid_row, grid_col, 2] **2 * img_w)
                    h = int(boxes[grid_row, grid_col, 3] **2 * img_h)
                    image = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), bbox_color, 2)
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    import os 

    print("current dir")
    print(os.getcwd())

    # set seed
    torch.manual_seed(0)
    
    wider_train_loader, wider_val_loader = get_data_loaders(batch_size=4)
    print(len(wider_train_loader), len(wider_val_loader))

    # Extract single batch of training data using next(iter(...))
    images, labels = next(iter(wider_train_loader))
    print(images.shape, labels.shape)
    print(images[0])

    # if models/yolo_faces_model_2023-05-19.pt exists, load it
    if os.path.exists("models/yolov1_faces_model_2023-05-19.pt"):
        print("Loading existing model.")
        model = torch.jit.load("models/yolov1_faces_model_2023-05-19.pt")
        labels = model(images)

    # Show images
    show_images(images, labels)