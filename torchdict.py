import os
from pathlib import Path
import torch
from tensordict.prototype import tensorclass

@tensorclass
class FolderData:
    files: torch.Tensor

    @classmethod
    def from_folder(cls, folder_path):
        file_list = sorted([str(file) for file in Path(folder_path).glob('*')])
        data = cls(
            files=torch.tensor(file_list, dtype=torch.object)
        )
        return data

if __name__ == "__main__":
    folder_path = "training-monolingual"

    folder_data = FolderData.from_folder(folder_path)

    # Now you have a tensorclass containing the list of files in the folder
    print(folder_data.files)















