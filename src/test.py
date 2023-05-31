from data_utils import get_data_loaders
from tqdm import tqdm


train_loader, val_loader = get_data_loaders()

mean = 0.
std = 0.
nb_samples = 0.
for data, _ in tqdm(train_loader):
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean, std)