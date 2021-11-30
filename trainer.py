from typing import Tuple
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from numpy import random
import matplotlib.pyplot as plt
from pathlib import Path

class NNTrainer:
    
    def __init__(self,
                 model: 'Module',
                 optim: 'Optimizer',
                 loss,
                 metric,
                 random_seed: 'int') -> None:
        
        assert optim.param_groups[0]['params'] == list(model.parameters())
        
        self.model = model
        self.optim = optim
        self.loss = loss
        self.metric = metric
        if random_seed is not None:
            random.seed(random_seed)
        
        
    @staticmethod    
    def _generate_batches(X: 'Tensor',
                          y: 'Tensor',
                          batch_size: 'int'=8) -> Tuple[Tensor]:
        
        number_of_instances = X.shape[0]
        
        for i in range(0, number_of_instances, batch_size):
            X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
            
            yield X_batch, y_batch
            
            
    @staticmethod
    def _shuffle_data(X: Tensor, 
                      y: Tensor):
        
        permuted_mask = random.permutation(X.shape[0])
        
        return X[permuted_mask], y[permuted_mask]
    
    
    def _print_last_epoch_stats(self):
        print(f'Epoch {len(self.train_losses) - 1}: train loss {self.train_losses[-1]:.4f} | validation loss {self.val_losses[-1]:.4f}, metric {self.val_metrics[-1]*100:.3f}%')
    
    
    def fit(self,
            X_train: Tensor,
            y_train: Tensor,
            X_test: Tensor,
            y_test: Tensor,
            epochs: int=10,
            eval_every: int=2,
            batch_size: int=8,
            verbose: 'bool'=True):
        
        # Save initial losses and metric
        train_out = self.model(X_train, inference=True)
        val_out = self.model(X_test, inference=True)
        
        self.train_losses = [self.loss(train_out, y_train).item()]
        self.val_losses = [self.loss(val_out, y_test).item()]
        self.val_metrics = [self.metric(y_test, val_out > .5)]
        
        if verbose:
            self._print_last_epoch_stats()
        
        
        # Run through the epochs
        for epoch in range(1, epochs + 1):
            # Shuffle data
            X_train, y_train = self._shuffle_data(X_train, y_train)
            
            # Get batch generator
            batch_generator = self._generate_batches(X_train, y_train, batch_size)
            
            # Run through the batches
            for i, (X_batch, y_batch) in enumerate(batch_generator):
                # Zero the optimizer gradients
                self.optim.zero_grad()
                
                # Forward step
                train_out = self.model(X_batch)
                train_loss = self.loss(train_out, y_batch)
                
                # Backward step
                train_loss.backward()
                
                # Update parameters
                self.optim.step()
            

            # Save initial losses and metric
            train_out = self.model(X_train, inference=True)
            val_out = self.model(X_test, inference=True)
            
            self.train_losses.append(self.loss(train_out, y_train).item())
            self.val_losses.append(self.loss(val_out, y_test).item())
            self.val_metrics.append(self.metric(y_test, val_out > .5))
            
            if verbose and epoch % eval_every == 0:
                self._print_last_epoch_stats()
                
                
                
    def _save_plot_fit_history(self, destination_path):
        model_name = str(self.model.__class__).split('.')[-1][:-2]
        plt.plot(self.train_losses, label='Train loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.plot(self.val_metrics, label='Validation accuracy')
        plt.legend()
        plt.title(f"{model_name}")
        plt.xlabel('epoch')
        plt.savefig(Path(destination_path) / f"{model_name.lower()}.jpeg")
    
            
        
        