import os, torch, numpy as np, pandas as pd, nibabel as nib
from torchvision.transforms import Compose, Resize, ToTensor
from typing import TypedDict
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

class Item(TypedDict):
    id: str
    scandate: str
    data: list
    label: torch.Tensor

class PandasDataset(Dataset):
    def __init__(self,
        cohort: pd.DataFrame,
        label: str="cancer",
    ):
        self.cohort = cohort.reset_index(drop=True)
        self.label = label

    def __getitem__(self, index) -> Item:
        return Item()
    
    def __len__(self) -> int:
        return len(self.cohort)
    
class SubjectDataset(PandasDataset):
    def __init__(self, 
                cohort,
                img_dir,
                transforms=Compose([
                    ToTensor(),
                    Resize((256,256)),
                ]),
                **kwargs,
                ):
        super().__init__(cohort, **kwargs)
        self.img_dir = img_dir
        self.transforms = transforms

    def __getitem__(self, index) -> Item:
        row = self.cohort.iloc[index]
        id, scandate = str(row.id), row.scandate
        fname = f"{id}_{scandate}.nii.gz"
        
        nii = nib.load(os.path.join(self.img_dir, fname))
        img = nii.get_fdata()
        img = self.transforms(img.astype(np.float32))

        age = torch.tensor(row.age, dtype=torch.float32)

        label = torch.tensor(row[self.label], dtype=torch.int64)

        return Item(
            id=id,
            scandate=scandate,
            data=[img, age],
            label=label
        )

class ScanDataset(PandasDataset):
    def __init__(self,
                cohort,
                img_dir,
                seqlen: int=3,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.ids = cohort.id.unique().tolist()
        self.seqlen = seqlen
        self.img_dir = img_dir
    

    def __getitem__(self, index) -> Item:
        id = self.ids[index]
        id_rows = self.cohort[self.cohort['id']==id].sort_values(by='scandate', ascending=False)
        id_rows = id_rows.iloc[:self.seqlen]

        # return stuff
    
    def __len__(self) -> int:
        return len(self.ids)