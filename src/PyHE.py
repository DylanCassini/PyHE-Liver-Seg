import numpy as np
import cv2
from scipy import ndimage
from typing import Union, Tuple, Optional
import warnings

class PyHE:
    """
    PyHE: Python Histogram Equalization Library
    
    A comprehensive library for histogram equalization techniques
    specifically designed for medical imaging applications including
    CT and MRI grayscale images.
    
    Supported methods:
    - Standard Histogram Equalization (HE)
    - Brightness Preserving Bi-Histogram Equalization (BBHE)
    - Recursive Mean Separated Histogram Equalization (RMSHE)
    - Dynamic Histogram Equalization (DHE)
    - Contrast Limited Adaptive Histogram Equalization (CLAHE)
    """
    
    def __init__(self):
        self.version = "1.0.0"
        
    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess input image.
        
        Args:
            image: Input image array
            
        Returns:
            Validated and preprocessed image
            
        Raises:
            ValueError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D grayscale or 3D volume")
            
        # For 3D volumes, don't convert to grayscale - keep as 3D
        if len(image.shape) == 3 and image.shape[2] in [3, 4]:  # RGB/RGBA
            # Convert to grayscale
            if image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                
        # Ensure uint8 format for histogram operations
        if image.dtype != np.uint8:
            if image.max() <= 1.0:  # Normalized image
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
                
        return image
    
    @staticmethod
    def _validate_volume(volume: np.ndarray, preserve_range: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Validate and preprocess 3D volume for enhancement.
        
        Args:
            volume: Input 3D volume array
            preserve_range: Whether to preserve original intensity range
            
        Returns:
            Tuple of (normalized_volume, original_min, original_max)
        """
        if not isinstance(volume, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(volume.shape) != 3:
            raise ValueError("Volume must be 3D")
        
        # Store original range
        original_min = float(volume.min())
        original_max = float(volume.max())
        
        if preserve_range:
            # Normalize to 0-255 range for processing
            if original_max == original_min:
                normalized = np.zeros_like(volume, dtype=np.uint8)
            else:
                normalized = ((volume - original_min) / (original_max - original_min) * 255).astype(np.uint8)
        else:
            # Ensure uint8 format
            if volume.dtype != np.uint8:
                if volume.max() <= 1.0:  # Normalized volume
                    normalized = (volume * 255).astype(np.uint8)
                else:
                    normalized = np.clip(volume, 0, 255).astype(np.uint8)
            else:
                normalized = volume
                
        return normalized, original_min, original_max
    
    @staticmethod
    def _restore_range(enhanced_volume: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
        """
        Restore original intensity range to enhanced volume.
        
        Args:
            enhanced_volume: Enhanced volume in 0-255 range
            original_min: Original minimum intensity
            original_max: Original maximum intensity
            
        Returns:
            Volume with restored original range
        """
        if original_max == original_min:
            return np.full_like(enhanced_volume, original_min, dtype=np.float64)
        
        # Convert back to original range
        normalized = enhanced_volume.astype(np.float64) / 255.0
        restored = normalized * (original_max - original_min) + original_min
        
        return restored
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Standard Histogram Equalization (HE).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Histogram equalized image
        """
        image = PyHE._validate_image(image)
        return cv2.equalizeHist(image)
    
    @staticmethod
    def bbhe(image: np.ndarray) -> np.ndarray:
        """
        Brightness Preserving Bi-Histogram Equalization (BBHE).
        
        Separates the histogram at the mean intensity and equalizes
        each sub-histogram independently to preserve brightness.
        
        Args:
            image: Input grayscale image
            
        Returns:
            BBHE processed image
        """
        image = PyHE._validate_image(image)
        
        # Calculate mean intensity
        mean_intensity = np.mean(image)
        
        # Create masks for lower and upper parts
        lower_mask = image <= mean_intensity
        upper_mask = image > mean_intensity
        
        # Create result image
        result = np.zeros_like(image)
        
        # Process lower part (0 to mean)
        if np.any(lower_mask):
            lower_part = image[lower_mask]
            hist_lower, bins_lower = np.histogram(lower_part, bins=int(mean_intensity) + 1, 
                                                range=(0, mean_intensity))
            cdf_lower = hist_lower.cumsum()
            cdf_lower = cdf_lower / cdf_lower[-1] * mean_intensity
            
            # Map lower intensities
            for i in range(len(bins_lower) - 1):
                mask = (image >= bins_lower[i]) & (image < bins_lower[i + 1]) & lower_mask
                result[mask] = cdf_lower[i]
        
        # Process upper part (mean to 255)
        if np.any(upper_mask):
            upper_part = image[upper_mask]
            hist_upper, bins_upper = np.histogram(upper_part, 
                                                bins=255 - int(mean_intensity),
                                                range=(mean_intensity, 255))
            cdf_upper = hist_upper.cumsum()
            cdf_upper = mean_intensity + (cdf_upper / cdf_upper[-1] * (255 - mean_intensity))
            
            # Map upper intensities
            for i in range(len(bins_upper) - 1):
                mask = (image >= bins_upper[i]) & (image < bins_upper[i + 1]) & upper_mask
                result[mask] = cdf_upper[i]
        
        return result.astype(np.uint8)
    
    @staticmethod
    def bbhe_3d(volume: np.ndarray) -> np.ndarray:
        """
        3D Brightness Preserving Bi-Histogram Equalization (BBHE).
        
        Applies BBHE to the entire 3D volume using global statistics
        to avoid slice-by-slice artifacts.
        
        Args:
            volume: Input 3D volume
            
        Returns:
            BBHE processed 3D volume
        """
        # Validate and normalize volume
        normalized_volume, original_min, original_max = PyHE._validate_volume(volume)
        
        # Calculate global mean intensity across entire volume
        mean_intensity = np.mean(normalized_volume)
        
        # Create masks for lower and upper parts
        lower_mask = normalized_volume <= mean_intensity
        upper_mask = normalized_volume > mean_intensity
        
        # Create result volume
        result = np.zeros_like(normalized_volume)
        
        # Process lower part (0 to mean)
        if np.any(lower_mask):
            lower_part = normalized_volume[lower_mask]
            hist_lower, bins_lower = np.histogram(lower_part, bins=int(mean_intensity) + 1, 
                                                range=(0, mean_intensity))
            cdf_lower = hist_lower.cumsum()
            if cdf_lower[-1] > 0:
                cdf_lower = cdf_lower / cdf_lower[-1] * mean_intensity
                
                # Map lower intensities using vectorized operations
                for i in range(len(bins_lower) - 1):
                    mask = (normalized_volume >= bins_lower[i]) & (normalized_volume < bins_lower[i + 1]) & lower_mask
                    result[mask] = cdf_lower[i]
        
        # Process upper part (mean to 255)
        if np.any(upper_mask):
            upper_part = normalized_volume[upper_mask]
            hist_upper, bins_upper = np.histogram(upper_part, 
                                                bins=255 - int(mean_intensity),
                                                range=(mean_intensity, 255))
            cdf_upper = hist_upper.cumsum()
            if cdf_upper[-1] > 0:
                cdf_upper = mean_intensity + (cdf_upper / cdf_upper[-1] * (255 - mean_intensity))
                
                # Map upper intensities using vectorized operations
                for i in range(len(bins_upper) - 1):
                    mask = (normalized_volume >= bins_upper[i]) & (normalized_volume < bins_upper[i + 1]) & upper_mask
                    result[mask] = cdf_upper[i]
        
        # Restore original range
        result_restored = PyHE._restore_range(result, original_min, original_max)
        
        return result_restored
    
    @staticmethod
    def histogram_equalization_3d(volume: np.ndarray) -> np.ndarray:
        """
        3D Histogram Equalization.
        
        Applies histogram equalization to the entire 3D volume using global statistics.
        
        Args:
            volume: Input 3D volume
            
        Returns:
            Histogram equalized 3D volume
        """
        # Validate and normalize volume
        normalized_volume, original_min, original_max = PyHE._validate_volume(volume)
        
        # Calculate global histogram
        flat_volume = normalized_volume.flatten()
        hist, bins = np.histogram(flat_volume, bins=256, range=(0, 256))
        
        # Calculate CDF
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1] * 255
        
        # Apply transformation
        result_flat = np.interp(flat_volume, bins[:-1], cdf)
        result = result_flat.reshape(normalized_volume.shape)
        
        # Restore original range
        result_restored = PyHE._restore_range(result, original_min, original_max)
        
        return result_restored
    
    @staticmethod
    def rmshe(image: np.ndarray, recursion_level: int = 2) -> np.ndarray:
        """
        Recursive Mean Separated Histogram Equalization (RMSHE).
        
        Recursively applies mean separation and histogram equalization.
        
        Args:
            image: Input grayscale image
            recursion_level: Number of recursive levels (default: 2)
            
        Returns:
            RMSHE processed image
        """
        image = PyHE._validate_image(image)
        
        def _recursive_equalize(img_segment: np.ndarray, level: int, original_shape: tuple = None) -> np.ndarray:
            if original_shape is None:
                original_shape = img_segment.shape
                
            if level == 0 or len(np.unique(img_segment)) <= 2:
                if len(img_segment.shape) == 1:
                    # For 1D array, create a temporary 2D array for equalizeHist
                    temp_img = np.zeros((1, len(img_segment)), dtype=np.uint8)
                    temp_img[0, :] = img_segment
                    equalized = cv2.equalizeHist(temp_img)
                    return equalized[0, :]
                else:
                    return cv2.equalizeHist(img_segment)
            
            mean_val = np.mean(img_segment)
            
            # For 2D images
            if len(img_segment.shape) == 2:
                # Split into lower and upper parts
                lower_mask = img_segment <= mean_val
                upper_mask = img_segment > mean_val
                
                result = np.zeros_like(img_segment)
                
                if np.any(lower_mask):
                    lower_indices = np.where(lower_mask)
                    lower_values = img_segment[lower_indices]
                    if len(lower_values) > 0:
                        lower_enhanced = _recursive_equalize(lower_values, level - 1)
                        result[lower_indices] = lower_enhanced
                
                if np.any(upper_mask):
                    upper_indices = np.where(upper_mask)
                    upper_values = img_segment[upper_indices]
                    if len(upper_values) > 0:
                        upper_enhanced = _recursive_equalize(upper_values, level - 1)
                        result[upper_indices] = upper_enhanced
                
                return result
            
            # For 1D arrays (flattened segments)
            else:
                lower_mask = img_segment <= mean_val
                upper_mask = img_segment > mean_val
                
                result = np.zeros_like(img_segment)
                
                if np.any(lower_mask):
                    lower_values = img_segment[lower_mask]
                    if len(lower_values) > 0:
                        lower_enhanced = _recursive_equalize(lower_values, level - 1)
                        result[lower_mask] = lower_enhanced
                        
                if np.any(upper_mask):
                    upper_values = img_segment[upper_mask]
                    if len(upper_values) > 0:
                        upper_enhanced = _recursive_equalize(upper_values, level - 1)
                        result[upper_mask] = upper_enhanced
                
                return result
        
        return _recursive_equalize(image, recursion_level).astype(np.uint8)
    
    @staticmethod
    def rmshe_3d(volume: np.ndarray, recursion_level: int = 2) -> np.ndarray:
        """
        3D Recursive Mean Separated Histogram Equalization (RMSHE).
        
        Applies RMSHE to the entire 3D volume using global statistics.
        
        Args:
            volume: Input 3D volume
            recursion_level: Number of recursive levels (default: 2)
            
        Returns:
            RMSHE processed 3D volume
        """
        # Validate and normalize volume
        normalized_volume, original_min, original_max = PyHE._validate_volume(volume)
        
        def _recursive_equalize_3d(vol_segment: np.ndarray, level: int) -> np.ndarray:
            if level == 0 or len(np.unique(vol_segment)) <= 2:
                # Apply histogram equalization to the entire volume segment
                flat_segment = vol_segment.flatten()
                hist, bins = np.histogram(flat_segment, bins=256, range=(0, 255))
                cdf = hist.cumsum()
                if cdf[-1] > 0:
                    cdf = cdf / cdf[-1] * 255
                    # Map intensities
                    result = np.zeros_like(vol_segment)
                    for i in range(len(bins) - 1):
                        mask = (vol_segment >= bins[i]) & (vol_segment < bins[i + 1])
                        result[mask] = cdf[i]
                    return result.astype(np.uint8)
                else:
                    return vol_segment
            
            mean_val = np.mean(vol_segment)
            
            # Split into lower and upper parts
            lower_mask = vol_segment <= mean_val
            upper_mask = vol_segment > mean_val
            
            result = np.zeros_like(vol_segment)
            
            if np.any(lower_mask):
                lower_segment = vol_segment.copy()
                lower_segment[~lower_mask] = 0  # Zero out upper part
                lower_enhanced = _recursive_equalize_3d(lower_segment[lower_mask].reshape(-1), level - 1)
                if len(lower_enhanced.shape) == 1:
                    # Reshape back to original positions
                    result[lower_mask] = lower_enhanced
            
            if np.any(upper_mask):
                upper_segment = vol_segment.copy()
                upper_segment[~upper_mask] = 0  # Zero out lower part
                upper_enhanced = _recursive_equalize_3d(upper_segment[upper_mask].reshape(-1), level - 1)
                if len(upper_enhanced.shape) == 1:
                    # Reshape back to original positions
                    result[upper_mask] = upper_enhanced
            
            return result
        
        # Apply recursive enhancement
        enhanced_volume = _recursive_equalize_3d(normalized_volume, recursion_level)
        
        # Restore original range
        result_restored = PyHE._restore_range(enhanced_volume, original_min, original_max)
        
        return result_restored
    
    @staticmethod
    def dhe_3d(volume: np.ndarray, x: float = 0.5) -> np.ndarray:
        """
        3D Dynamic Histogram Equalization (DHE).
        
        Applies DHE to the entire 3D volume using global statistics.
        
        Args:
            volume: Input 3D volume
            x: Plateau limit parameter (0 < x < 1, default: 0.5)
            
        Returns:
            DHE processed 3D volume
        """
        # Validate and normalize volume
        normalized_volume, original_min, original_max = PyHE._validate_volume(volume)
        
        if not 0 < x < 1:
            raise ValueError("Parameter x must be between 0 and 1")
        
        # Calculate histogram
        hist, bins = np.histogram(normalized_volume.flatten(), bins=256, range=(0, 256))
        
        # Calculate plateau limit
        total_voxels = normalized_volume.size
        plateau_limit = x * total_voxels / 256
        
        # Clip histogram
        clipped_hist = np.minimum(hist, plateau_limit)
        
        # Redistribute clipped pixels
        excess = np.sum(hist - clipped_hist)
        redistribution = excess / 256
        clipped_hist += redistribution
        
        # Calculate CDF
        cdf = clipped_hist.cumsum()
        cdf = cdf / cdf[-1] * 255
        
        # Apply transformation
        result_flat = np.interp(normalized_volume.flatten(), bins[:-1], cdf)
        result = result_flat.reshape(normalized_volume.shape)
        
        # Restore original range
        result_restored = PyHE._restore_range(result, original_min, original_max)
        
        return result_restored
    
    @staticmethod
    def clahe_3d(volume: np.ndarray, 
                 clip_limit: float = 2.0, 
                 tile_grid_size: Tuple[int, int, int] = (4, 4, 4)) -> np.ndarray:
        """
        3D Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Applies CLAHE slice by slice to maintain 3D coherence.
        
        Args:
            volume: Input 3D volume
            clip_limit: Threshold for contrast limiting (default: 2.0)
            tile_grid_size: Size of the grid for tiles in 3D (default: (4, 4, 4))
            
        Returns:
            CLAHE processed 3D volume
        """
        # Validate and normalize volume
        normalized_volume, original_min, original_max = PyHE._validate_volume(volume)
        
        # Create CLAHE object for 2D slices
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size[:2])
        
        # Apply CLAHE slice by slice
        result = np.zeros_like(normalized_volume)
        for i in range(normalized_volume.shape[2]):
            result[:, :, i] = clahe_obj.apply(normalized_volume[:, :, i])
        
        # Restore original range
        result_restored = PyHE._restore_range(result, original_min, original_max)
        
        return result_restored.astype(np.uint8)
    
    @staticmethod
    def dhe(image: np.ndarray, x: float = 0.5) -> np.ndarray:
        """
        Dynamic Histogram Equalization (DHE).
        
        Assigns specific gray level ranges based on local characteristics.
        
        Args:
            image: Input grayscale image
            x: Plateau limit parameter (0 < x < 1, default: 0.5)
            
        Returns:
            DHE processed image
        """
        image = PyHE._validate_image(image)
        
        if not 0 < x < 1:
            raise ValueError("Parameter x must be between 0 and 1")
        
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
        
        # Calculate plateau limit
        total_pixels = image.size
        plateau_limit = x * total_pixels / 256
        
        # Clip histogram
        clipped_hist = np.minimum(hist, plateau_limit)
        
        # Redistribute clipped pixels
        excess = np.sum(hist - clipped_hist)
        redistribution = excess / 256
        clipped_hist += redistribution
        
        # Calculate CDF
        cdf = clipped_hist.cumsum()
        cdf = cdf / cdf[-1] * 255
        
        # Apply transformation
        result = np.interp(image.flatten(), bins[:-1], cdf)
        return result.reshape(image.shape).astype(np.uint8)
    
    @staticmethod
    def clahe(image: np.ndarray, 
              clip_limit: float = 2.0, 
              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Applies histogram equalization to small regions (tiles) with
        contrast limiting to prevent over-amplification of noise.
        
        Args:
            image: Input grayscale image
            clip_limit: Threshold for contrast limiting (default: 2.0)
            tile_grid_size: Size of the grid for tiles (default: (8, 8))
            
        Returns:
            CLAHE processed image
        """
        image = PyHE._validate_image(image)
        
        # Create CLAHE object
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        return clahe_obj.apply(image)
    
    @staticmethod
    def adaptive_clahe(image: np.ndarray, 
                      window_size: int = 64,
                      clip_limit: float = 2.0) -> np.ndarray:
        """
        Adaptive CLAHE with automatic parameter adjustment for medical images.
        
        Args:
            image: Input grayscale image
            window_size: Size of the processing window (default: 64)
            clip_limit: Base clip limit (default: 2.0)
            
        Returns:
            Adaptive CLAHE processed image
        """
        image = PyHE._validate_image(image)
        
        # Calculate local statistics
        local_mean = ndimage.uniform_filter(image.astype(np.float32), size=window_size)
        local_std = ndimage.uniform_filter(image.astype(np.float32)**2, size=window_size)
        local_std = np.sqrt(local_std - local_mean**2)
        
        # Adaptive tile size based on image characteristics
        mean_std = np.mean(local_std)
        if mean_std < 20:  # Low contrast regions
            tile_size = (16, 16)
            adaptive_clip = clip_limit * 1.5
        elif mean_std > 50:  # High contrast regions
            tile_size = (4, 4)
            adaptive_clip = clip_limit * 0.7
        else:
            tile_size = (8, 8)
            adaptive_clip = clip_limit
        
        # Apply adaptive CLAHE
        clahe_obj = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=tile_size)
        return clahe_obj.apply(image)
    
    @staticmethod
    def medical_enhancement(image: np.ndarray, 
                          method: str = 'clahe',
                          preserve_range: bool = True) -> np.ndarray:
        """
        Medical image specific enhancement combining multiple techniques.
        
        Args:
            image: Input medical image (CT/MRI)
            method: Enhancement method ('clahe', 'bbhe', 'dhe', 'rmshe')
            preserve_range: Whether to preserve original intensity range
            
        Returns:
            Enhanced medical image
        """
        image = PyHE._validate_image(image)
        original_range = (image.min(), image.max())
        
        # Apply selected enhancement method
        if method.lower() == 'clahe':
            enhanced = PyHE.adaptive_clahe(image, window_size=32, clip_limit=1.5)
        elif method.lower() == 'bbhe':
            enhanced = PyHE.bbhe(image)
        elif method.lower() == 'dhe':
            enhanced = PyHE.dhe(image, x=0.3)
        elif method.lower() == 'rmshe':
            enhanced = PyHE.rmshe(image, recursion_level=3)
        else:
            enhanced = PyHE.histogram_equalization(image)
        
        # Preserve original intensity range if requested
        if preserve_range:
            enhanced = np.interp(enhanced, 
                               (enhanced.min(), enhanced.max()),
                               original_range)
            enhanced = enhanced.astype(image.dtype)
        
        return enhanced
    
    @staticmethod
    def compare_methods(image: np.ndarray) -> dict:
        """
        Compare different histogram equalization methods on the same image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary containing results from different methods
        """
        image = PyHE._validate_image(image)
        
        results = {
            'original': image,
            'he': PyHE.histogram_equalization(image),
            'bbhe': PyHE.bbhe(image),
            'rmshe': PyHE.rmshe(image),
            'dhe': PyHE.dhe(image),
            'clahe': PyHE.clahe(image),
            'adaptive_clahe': PyHE.adaptive_clahe(image),
            'medical_enhancement': PyHE.medical_enhancement(image)
        }
        
        return results
    
    @staticmethod
    def compare_methods_3d(volume: np.ndarray) -> dict:
        """
        Compare different 3D histogram equalization methods on the same volume.
        
        Args:
            volume: Input 3D volume
            
        Returns:
            Dictionary containing results from different 3D methods
        """
        results = {
            'original': volume,
            'he_3d': PyHE.histogram_equalization_3d(volume),
            'bbhe_3d': PyHE.bbhe_3d(volume),
            'rmshe_3d': PyHE.rmshe_3d(volume),
            'dhe_3d': PyHE.dhe_3d(volume),
            'clahe_3d': PyHE.clahe_3d(volume)
        }
        
        return results
    
    @staticmethod
    def calculate_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
        """
        Calculate quality metrics for enhanced images.
        
        Args:
            original: Original image
            enhanced: Enhanced image
            
        Returns:
            Dictionary containing quality metrics
        """
        original = PyHE._validate_image(original)
        enhanced = PyHE._validate_image(enhanced)
        
        # Mean Squared Error
        mse = np.mean((original.astype(np.float32) - enhanced.astype(np.float32))**2)
        
        # Peak Signal-to-Noise Ratio
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Structural Similarity Index (simplified)
        mu1, mu2 = np.mean(original), np.mean(enhanced)
        sigma1, sigma2 = np.std(original), np.std(enhanced)
        sigma12 = np.mean((original - mu1) * (enhanced - mu2))
        
        c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        
        # Contrast Enhancement Ratio
        contrast_original = np.std(original)
        contrast_enhanced = np.std(enhanced)
        cer = contrast_enhanced / contrast_original if contrast_original > 0 else 1
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'contrast_enhancement_ratio': cer,
            'mean_brightness_change': np.mean(enhanced) - np.mean(original)
        }

# Example usage and testing functions
def demo_pyhe():
    """
    Demonstration function showing PyHE capabilities.
    """
    print(f"PyHE Library v{PyHE().version}")
    print("Medical Image Histogram Equalization Library")
    print("\nSupported methods:")
    print("- Standard Histogram Equalization (HE)")
    print("- Brightness Preserving Bi-Histogram Equalization (BBHE)")
    print("- Recursive Mean Separated Histogram Equalization (RMSHE)")
    print("- Dynamic Histogram Equalization (DHE)")
    print("- Contrast Limited Adaptive Histogram Equalization (CLAHE)")
    print("- Adaptive CLAHE for medical images")
    print("- Medical-specific enhancement")
    
    # Create a sample medical-like image for demonstration
    sample_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # Add some structure to simulate medical image characteristics
    center = (128, 128)
    y, x = np.ogrid[:256, :256]
    mask = (x - center[0])**2 + (y - center[1])**2 <= 80**2
    sample_image[mask] = sample_image[mask] * 0.3 + 180  # Bright region
    
    print("\nProcessing sample image...")
    results = PyHE.compare_methods(sample_image)
    
    for method, result in results.items():
        if method != 'original':
            metrics = PyHE.calculate_metrics(sample_image, result)
            print(f"{method.upper()}: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}")

if __name__ == "__main__":
    demo_pyhe()