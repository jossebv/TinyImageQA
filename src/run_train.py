from argparse import ArgumentParser
import lightning as L
from models_definitions.TinyImageQA import TinyImageQA
from data_module import RLAIFDataModule
from lightning.pytorch.loggers import WandbLogger


def main(hparams):
    L.seed_everything(42, workers=True)

    datamodule = RLAIFDataModule(batch_size=4, num_workers=7)

    model = TinyImageQA()

    logger = WandbLogger(
        project="TinyImageQA", entity="josebravopacheco-team", group=hparams.group
    )

    trainer = L.Trainer(
        precision="bf16-mixed", max_epochs=20, logger=logger, deterministic=True
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--group", default="Unnamed run")

    args = parser.parse_args()
    main(args)
