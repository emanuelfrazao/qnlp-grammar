from torch import Tensor, nn

class SimpleBilinearModel(nn.Module):
    
    def __init__(self,
                 space_dimension: 'int',
                 dropout_prob: 'float') -> None:
        super().__init__()
        
        # Set bilinear layer with space dimension
        self.bi = nn.Bilinear(space_dimension, space_dimension, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Sigmoid()
        
        
        
    def forward(self,
                X: Tensor,
                inference: 'bool'=False) -> Tensor:
        
        if inference:
            self.eval()
        else:
            self.train()
        
        out = self.bi(X[:,0,:], X[:,1,:])
        out = self.dropout(out)
        out = self.activation(out)

        return out
    
    

class Perceptron(nn.Module):
    
    def __init__(self,
                 space_dimension: 'int',
                 hidden_dim: 'int',
                 dropout_prob: 'float') -> None:
        super().__init__()
        
        # Set bilinear layer with space dimension
        self.layers = nn.Sequential(
            nn.Linear(space_dimension, hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(2*hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                X: Tensor,
                inference: 'bool'=False) -> Tensor:
        
        if inference:
            self.eval()
        else:
            self.train()
        
        out = self.layers(X)

        return out
    




class LinearBilinearModel(nn.Module):
    
    def __init__(self,
                 space_dimension: 'int',
                 hidden_dim: 'int',
                 dropout_prob: 'float') -> None:
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(space_dimension, hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Sigmoid(),
        )
        
        # Set bilinear layer with hidden dimension
        self.bi = nn.Bilinear(hidden_dim, hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Sigmoid()
        
        
    def forward(self,
                X: Tensor,
                inference: 'bool'=False) -> Tensor:
        if inference:
            self.eval()
        else:
            self.train()
        
        out = self.linear(X)
        out = self.bi(out[:,0,:], out[:,1,:])
        out = self.dropout(out)
        out = self.activation(out)
        
        return out
    
    
    

