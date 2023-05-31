import torch


class YoloV1Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.grid_size = 7

        pooling_layers = [True] * 7 + [False] * 3
        n_in = [3, 16, 32, 64, 128, 256, 512, 1024, 1024, 1024]
        n_out = n_in[1:] + [1024]

        self.layers = torch.nn.Sequential(*[
                self.downsize_layer(n_in, n_out, pooling)
                for n_in, n_out, pooling in zip(n_in, n_out, pooling_layers)])

        self.fc1 = torch.nn.Linear(self.grid_size * self.grid_size * 1024, 4096)
        self.fc2 = torch.nn.Linear(4096, self.grid_size * self.grid_size * 5) 

    def downsize_layer(self, n_in, n_out, pooling=True):
        out_layers = [
            torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(n_out),
            torch.nn.LeakyReLU(0.1)
        ]
        if pooling:
            out_layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
        return torch.nn.Sequential(*out_layers)

    def forward(self, x):
        x = self.layers(x) 
        x = x.view(x.shape[0], -1)
        x = self.fc1(x) 
        x = torch.relu(x)
        x = self.fc2(x) 
        x = x.view(x.shape[0], 7, 7, 5)
        x = torch.sigmoid(x)
        return x
        

if __name__=="__main__":
    from data_utils import get_data_loaders
    train_loader, val_loader = get_data_loaders()

    model = YoloV1Model()
    print(model)

    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)

    output = model(images)
    print(output.shape)