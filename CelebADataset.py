import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, df, root_dir, id_map, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with columns ['image_path', 'id']
            root_dir (string): Directory containing the 'img_align_celeba' folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.id_to_label = id_map
        
        # MAPPING: Create a map from the original CelebA ID to a training index (0 to N-1).
        # This ensures that if you select 500 random people, their labels become 0-499.
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {original_id: idx for idx, original_id in enumerate(self.unique_ids)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get row data
        row = self.df.iloc[idx]
        filename = row['image_path']
        original_id = row['id']

        # Construct path: root/img_align_celeba/000001.jpg
        img_path = os.path.join('img_align_celeba', filename)

        # Load Image
        # We convert to 'L' (Grayscale) immediately to match your paper's requirement.
        # If you want RGB, change 'L' to 'RGB'.
        image = Image.open(img_path).convert('L')

        # Apply Transforms
        if self.transform:
            image = self.transform(image)

        # Get the mapped label (0 to N-1)
        label = self.id_to_label[original_id]

        return image, label, original_id