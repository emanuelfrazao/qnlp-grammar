import numpy as np

class BaseModel:
    """Class for the base model computing the inner product of the 2 sentences and finding a threshold for separating the 2 classes.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def inner_product(X: np.ndarray) -> np.ndarray:
        """Computes inner product of every pair of sentences.

        Args:
            X (np.ndarray): The features array with shape (n_instances, n_sentences, space_dimension).

        Returns:
            np.ndarray: A numpy ndarray with shape (n_instances,) with the inner product of the instance' sentences.
        """
        return np.sum(X[:,0,:] * X[:,1,:], axis=1)
        
        
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray) -> None:
        """Fits the model.

        Args:
            X_train (np.ndarray): The features array to train on, with shape (n_instances, n_sentences, space_dimension).
            y_train (np.ndarray): The labels to train on.
        """        
        assert X_train.shape[0] == y_train.shape[0], "Shapes of features and labels must match on the first axis."
        assert X_train.shape[1] == 2, "The sentences must be in the second axis of the training batch."
        
        # Compute inner product of every pair of sentences
        inner = self.inner_product(X_train)
    
        # Get mean inner product of each class
        y_pos = y_train.reshape(-1) == 1
        
        mean_pos = inner[y_pos].mean()
        mean_neg = inner[~y_pos].mean()
        
        # Save threshold for prediction
        self.threshold = (mean_pos + mean_neg) / 2
        self.higher_is_positive = mean_pos > mean_neg
        
    
    def predict(self,
                X_test: np.ndarray) -> np.ndarray:
        """Predicts the labels by making use of the threshold found.

        Args:
            X_test (np.ndarray): The features array to predict, with shape (n_instances, n_sentences, space_dimension).

        Returns:
            np.ndarray: The prediction.
        """        
        assert hasattr(self, "threshold") and hasattr(self, "higher_is_positive"), "The model is not fitted yet."
        
        inner = self.inner_product(X_test)
        
        return (inner > self.threshold if self.higher_is_positive else inner < self.threshold).astype(int)