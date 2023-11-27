import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import random
from torch.utils.data import SubsetRandomSampler

class dataProcess:
    def __init__(self, config) -> None:
        self.config = config
        # Define data transformations
        transform = transforms.Compose([
            transforms.Resize((config["resizewidth"], config["resizeheight"])), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])
        # Load dataset
        self.dataset = ImageFolder(config["data_root"], transform=transform) 
    
        # Generate samplers for train/test split
        self.train_sampler, self.test_sampler = self.generate_sampler()
        
    # Split dataset into train/test indices
    def generate_sampler(self):
        # Get class indices
        class_to_idx = self.dataset.class_to_idx
  
        sample_indices = []
        total_class_indices =[]
        for class_label, class_idx in class_to_idx.items():
           
        
            class_indices = [idx for idx, (_, label) in enumerate(self.dataset.samples) if label == class_idx]
           
            sampled_class_indices = random.sample(class_indices, self.config["num_samples_per_class"])
          
            sample_indices.extend(sampled_class_indices)
            total_class_indices.extend(class_indices)
                
        set1=set(sample_indices)
        set2=set(total_class_indices)
        test_indices = list(set2.difference(set1))
     

     
        train_sampler = SubsetRandomSampler(sample_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        return train_sampler, test_sampler

    # Return training and test dataloaders  
    def get_data(self):
        train_loader = DataLoader(self.dataset, batch_size=self.config["batch_size"], sampler=self.train_sampler,pin_memory=True)
        test_loader = DataLoader(self.dataset, batch_size=self.config["batch_size"], sampler=self.test_sampler,pin_memory=True)
        return train_loader, test_loader