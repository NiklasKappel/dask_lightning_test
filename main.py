import os
import shutil
from contextlib import contextmanager

import lightning as L
from dask.distributed import Client
from dask_jobqueue.slurm import SLURMCluster
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, _):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


def sbatch_is_available() -> bool:
    return shutil.which("sbatch") is not None


@contextmanager
def dask_client(cluster):
    cluster.scale(jobs=1)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


def train_model():
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    autoencoder = LitAutoEncoder(encoder, decoder)

    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = DataLoader(dataset)

    trainer = L.Trainer(
        enable_progress_bar=False, limit_train_batches=100, max_epochs=1
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


def main():
    if sbatch_is_available():
        cluster = SLURMCluster(
            cores=8,
            memory="32GB",
            job_extra_directives=["--gres=gpu:1"],
            queue="a100",
        )
        with dask_client(cluster) as client:
            client.submit(train_model).result()
    else:
        train_model()


def main2():
    if sbatch_is_available():
        cluster = SLURMCluster(
            cores=8,
            memory="32GB",
            job_extra_directives=["--gres=gpu:1"],
            queue="a100",
        )
        with dask_client(cluster) as client:
            client.submit(lambda: print("Running on Slurm")).result()
    else:
        print("Running locally")


if __name__ == "__main__":
    main2()
