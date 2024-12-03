from numba import jit
import numpy as np

class KDE2D:
    def __init__(self, x, y):
        self.dataset = np.vstack([x, y])
        self.d, self.n = self.dataset.shape
        
        # Compute scipy's exact bandwidth (Scott's rule)
        self.factor = self.n ** (-1./(self.d+4))
        self.covariance = np.cov(self.dataset, rowvar=True, bias=False)
        self.inv_cov = np.linalg.inv(self.covariance)
        self.norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance))

    @staticmethod
    @jit(nopython=True)
    def _evaluate_kernel(dataset, points, inv_cov, norm_factor, factor):
        d, n = dataset.shape
        m = points.shape[1]
        result = np.zeros(m)
        
        for i in range(m):
            sum_val = 0.0
            for j in range(n):
                diff = points[:, i] - dataset[:, j]
                diff = diff / factor
                
                # Compute Mahalanobis distance
                mahal = 0.0
                for k in range(d):
                    for l in range(d):
                        mahal += diff[k] * inv_cov[k, l] * diff[l]
                
                sum_val += np.exp(-0.5 * mahal)
            
            result[i] = sum_val / (n * factor**d * norm_factor)
            
        return result

    def evaluate(self, points):
        """Raw evaluate method expecting (2, n_points) shaped array"""
        points = np.asarray(points)
        return self._evaluate_kernel(
            self.dataset, points,
            self.inv_cov, self.norm_factor,
            self.factor
        )
    
    def evaluate_points(self, x, y):
        """
        Convenient method to evaluate density at arbitrary points
        
        Parameters:
        -----------
        x, y : array-like or float
            Points at which to evaluate the density
        
        Returns:
        --------
        density : array or float
            Density values at the input points
        """
        # Convert to arrays if needed
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        
        # Reshape for evaluation
        points = np.vstack([x.ravel(), y.ravel()])
        
        # Evaluate
        density = self.evaluate(points)
        
        # Return scalar if input was scalar
        if len(x) == 1:
            return density[0]
        
        # Otherwise return array with original shape
        return density.reshape(x.shape)