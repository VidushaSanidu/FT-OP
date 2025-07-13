"""
Data management module for handling datasets and data loaders.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, ConcatDataset

from data.loader import data_loader
from utils import get_dset_path

logger = logging.getLogger(__name__)

class DataManager:
    """Manages data loading for different training scenarios."""
    
    def __init__(self, config):
        self.config = config
        self.available_datasets = config.data.available_datasets
        self.validation_dataset = config.validation_dataset
        self.train_datasets = config.train_datasets or self.available_datasets
        
    def get_single_dataset_loader(self, dataset_name: str, split: str) -> Tuple[any, DataLoader]:
        """Get data loader for a single dataset."""
        dataset_path = get_dset_path(dataset_name, split)
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return None, None
        
        # Create a simple config object that matches what data_loader expects
        class SimpleConfig:
            def __init__(self, config):
                self.obs_len = config.model.obs_len
                self.pred_len = config.model.pred_len
                self.skip = config.data.skip
                self.delim = config.data.delim
                self.batch_size = config.data.batch_size
                self.loader_num_workers = config.data.loader_num_workers
        
        simple_config = SimpleConfig(self.config)
        dset, loader = data_loader(simple_config, dataset_path)
        logger.info(f"Loaded {dataset_name} {split} dataset with {len(dset)} samples")
        return dset, loader
    
    def get_centralized_train_loader(self) -> Tuple[List[any], DataLoader]:
        """Get combined data loader for centralized training using all specified datasets."""
        all_datasets = []
        total_samples = 0
        
        logger.info(f"Preparing centralized training data from datasets: {self.train_datasets}")
        
        for dataset_name in self.train_datasets:
            dset, _ = self.get_single_dataset_loader(dataset_name, 'train')
            if dset is not None:
                all_datasets.append(dset)
                total_samples += len(dset)
                logger.info(f"Added {dataset_name}: {len(dset)} samples")
        
        if not all_datasets:
            raise ValueError("No valid training datasets found")
        
        # Combine all datasets
        combined_dataset = ConcatDataset(all_datasets)
        
        # Create combined data loader with proper collate function
        from data.trajectories import seq_collate
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.loader_num_workers,
            collate_fn=seq_collate
        )
        
        logger.info(f"Created centralized training loader with {total_samples} total samples")
        return all_datasets, combined_loader
    
    def get_validation_loader(self) -> Tuple[any, DataLoader]:
        """Get validation data loader."""
        logger.info(f"Preparing validation data from dataset: {self.validation_dataset}")
        return self.get_single_dataset_loader(self.validation_dataset, 'val')
    
    def get_test_loader(self, dataset_name: Optional[str] = None) -> Tuple[any, DataLoader]:
        """Get test data loader."""
        test_dataset = dataset_name or self.validation_dataset
        logger.info(f"Preparing test data from dataset: {test_dataset}")
        return self.get_single_dataset_loader(test_dataset, 'test')
    
    def get_federated_client_loaders(self) -> Dict[str, Tuple[any, DataLoader]]:
        """Get data loaders for federated learning clients."""
        client_loaders = {}
        
        logger.info(f"Preparing federated client data from datasets: {self.train_datasets}")
        
        for dataset_name in self.train_datasets:
            dset, loader = self.get_single_dataset_loader(dataset_name, 'train')
            if dset is not None and loader is not None:
                client_loaders[dataset_name] = (dset, loader)
        
        if not client_loaders:
            raise ValueError("No valid client datasets found for federated learning")
        
        logger.info(f"Created {len(client_loaders)} federated clients")
        return client_loaders
    
    def get_data_statistics(self) -> Dict[str, any]:
        """Get statistics about the available data."""
        stats = {
            'available_datasets': self.available_datasets,
            'train_datasets': self.train_datasets,
            'validation_dataset': self.validation_dataset,
            'dataset_sizes': {}
        }
        
        for dataset_name in self.available_datasets:
            dataset_stats = {}
            for split in ['train', 'val', 'test']:
                dset, _ = self.get_single_dataset_loader(dataset_name, split)
                dataset_stats[split] = len(dset) if dset is not None else 0
            stats['dataset_sizes'][dataset_name] = dataset_stats
        
        return stats
    
    def validate_datasets(self) -> bool:
        """Validate that all required datasets exist and are accessible."""
        missing_datasets = []
        
        # Check training datasets
        for dataset_name in self.train_datasets:
            train_path = get_dset_path(dataset_name, 'train')
            if not os.path.exists(train_path):
                missing_datasets.append(f"{dataset_name}/train")
        
        # Check validation dataset
        val_path = get_dset_path(self.validation_dataset, 'val')
        if not os.path.exists(val_path):
            missing_datasets.append(f"{self.validation_dataset}/val")
        
        if missing_datasets:
            logger.error(f"Missing dataset paths: {missing_datasets}")
            return False
        
        logger.info("All required datasets are available")
        return True

def create_data_manager(config) -> DataManager:
    """Factory function to create and validate data manager."""
    data_manager = DataManager(config)
    
    if not data_manager.validate_datasets():
        raise ValueError("Dataset validation failed. Please check dataset paths.")
    
    return data_manager
