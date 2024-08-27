from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    batch_size: int = 8
    num_epochs: int = 20

    lr: float = 10**-4

    seq_len: int = 350
    d_model: int = 512

    datasource: str = 'news_commentary'
    lang_src: str = "en"
    lang_tgt: str = "ar"

    model_folder: str = "models"
    preload: str = "latest"
    tokenizer_file: str = "tokenizer_{0}.json"
    
    experiment_name: str = "runs/tmodel"

    def get_weights_files_path(self, epoch: str):
        return str(Path(self.model_folder).glob(f'checkpoint_{epoch}.pth'))

    def latest_weights_file_path(self) -> Optional[str]:
        checkpoints = list(Path(self.model_folder).glob('checkpoint_*.pth'))
        if checkpoints:
            # Extract the epoch number from each checkpoint file name and find the maximum
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
            return str(latest_checkpoint)
        
        return None


if __name__=="__main__":
    config = Config()

    weights_file_path = config.get_weights_file_path("10")
    print("Specific weights file path:", weights_file_path)

    latest_weights = config.latest_weights_file_path()
    print("Latest weights file path:", latest_weights)
