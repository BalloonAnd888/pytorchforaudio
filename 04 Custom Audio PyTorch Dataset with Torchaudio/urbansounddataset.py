import os 
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
# import soundata

# dataset: https://goo.gl/8hY5ER

class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label 
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:/Users/alau2/Documents/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "C:/Users/alau2/Documents/datasets/UrbanSound8K/audio"
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    print(f"There are {len(usd)} samples in the dataset.")
    # signal, label = usd[0]


    # print("Initializing soundata...")
    # dataset = soundata.initialize('urbansound8k')
    # dataset.download()  # download the dataset
    # dataset.validate()  # validate that all the expected files are there

    # ANNOTATIONS_FILE = os.path.join(dataset.data_home, "metadata", "UrbanSound8K.csv")
    # AUDIO_DIR = os.path.join(dataset.data_home, "audio")

    # usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    # print(f"There are {len(usd)} samples in the dataset.")

    # signal, label = usd[0]
