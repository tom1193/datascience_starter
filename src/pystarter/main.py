import os, argparse, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pystarter.datasets.cohorts import NLST_CohortWrapper
from pystarter.datasets.imaging import SubjectDataset
from pystarter.models.encoders import MLP
import pystarter.definitions as D
from pystarter.utils import seed_everything, load_config

COHORTS = {
    'nlst.train_set': lambda: NLST_CohortWrapper().train_set,
    'nlst.train_oldage': lambda: NLST_CohortWrapper().train_oldage
}

MODEL_CLASS = {
    'mlp': MLP,
}

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("phase", choices=['train', 'val', 'cv', 'test', 'predict'])
    parser.add_argument("cohort", choices=COHORTS.keys())
    parser.add_argument("--img_dir", type=str, default="/home/local/nlst")
    args = parser.parse_args()

    seed_everything(D.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    cohort = COHORTS[args.cohort]()
    img_dir = args.img_dir

    log_dir = os.path.join(D.LOGS, config['id'])
    checkpoint_dir = os.path.join(D.CHECKPOINTS, config['id'])
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = MODEL_CLASS[config['model_class']](
        in_features=256,
        hidden_features=config['hidden_features'],
        out_features=config['num_classes'],
    ).to(device)

    if args.phase =='train':
        # training/validation split
        val_df = cohort.groupby('cancer', group_keys=False).apply(lambda x: x.sample(frac=0.2))
        train_df = cohort.drop(val_df.index)
        train_dataset = SubjectDataset(train_df, img_dir)
        val_dataset = SubjectDataset(val_df, img_dir)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

        # loss function
        criterion = nn.BCEWithLogitsLoss()
        # optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

        experiment = Experiment(config, model, train_loader, val_loader, device, log_dir, checkpoint_dir)
        experiment.train()
        
class Experiment():
    def __init__(self,
            config: dict,
            model,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str,
            log_dir: str,
            checkpoint_dir: str,
        ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0
        self.best_metric = 1e6

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95))

    def train(self):
        self.model.train()
        for epoch in range(self.config["epochs"]):
            for batch in self.train_loader:
                data, y = batch['data'].to(self.device), batch['label'].to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(*data)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
                self.logger.add_scalar("train_loss", loss.item(), self.global_step)

                if self.global_step % self.config["val_interval"] == 0:
                    val_loss = self.validate()
                    
                    if val_loss < self.best_metric:
                        self.best_metric = val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_metric': self.best_metric,
                        }, os.path.join(self.checkpoint_dir, f"epoch{epoch}.tar"))

                self.global_step += 1

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in self.val_loader:
                data, y = batch['data'].to(self.device), batch['label'].to(self.device)
                y_hat = self.model(*data)
                loss = self.criterion(y_hat, y)
                val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            self.logger.add_scalar("val_loss", val_loss, self.global_step)
            return val_loss

