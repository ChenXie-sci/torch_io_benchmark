import time
import os
from torch.utils.data import Dataset, DataLoader

class NLPDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        # You might want to sort the file paths if the order matters

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r', encoding='utf-8') as file:
            data = file.read()
        return data

def main():
    folder_path = 'training-monolingual'
    dataset = NLPDataset(folder_path)



    # Set batch_size and num_workers according to your needs
    batch_size = 1024
    num_workers = 8

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Measure total I/O time
    start_time = time.time()

    for batch in dataloader:
        # Your processing logic here
        pass

    end_time = time.time()
    total_io_time = end_time - start_time
    print(f"Total I/O time: {total_io_time} seconds")

if __name__ == "__main__":
    main()




