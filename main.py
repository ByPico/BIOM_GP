import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import cv2
import dlib
import glob
from tqdm import tqdm

class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.
        d' measures how well separated the genuine and impostor distributions are.
        Higher d' means better separation and better system performance.

        Returns:
        - float: The calculated d' value.
        """
        #calculate mean of genuine and impostor scores
        genuine_mean = np.mean(self.genuine_scores)
        impostor_mean = np.mean(self.impostor_scores)
        
        #calculate standard deviation of genuine and impostor scores
        genuine_std = np.std(self.genuine_scores)
        impostor_std = np.std(self.impostor_scores)
        
        #d' formula is |mean1 - mean2| / sqrt((std1^2 + std2^2)/2)
        x = np.abs(genuine_mean - impostor_mean)
        y = np.sqrt((genuine_std**2 + impostor_std**2) / 2)
        
        #Return d' avoiding division by zero with epsilon value 
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores with logarithmic y-axis scale.
        """
        plt.figure(figsize=(10, 6))
        
        # Define the number of bins and their edges
        bins = np.linspace(0, 1, 21)  # 20 bins between 0 and 1
        
        # Create histograms with hatched patterns
        plt.hist(
            self.genuine_scores,
            bins=bins,
            color='green',
            lw=2,
            histtype='step',  # Use step type for the histogram outline
            hatch='/////',     #Add hatching pattern for genuine scores
            label='Genuine',
            log=True          #Set logarithmic scale for y-axis
        )
        
        plt.hist(
            self.impostor_scores,
            bins=bins,
            color='red',
            lw=2,
            histtype='step',  # Use step type for the histogram outline
            hatch='\\\\\\',    #add hatching pattern for impostor scores
            label='Impostor',
            log=True          #set logarithmic scale for y-axis
        )
        
        # Set y-axis to log scale
        plt.yscale('log')
        
        # Set y-axis ticks to be powers of 10
        plt.yticks([1, 10, 100, 1000], ['10⁰', '10¹', '10²', '10³'])
        
        # Set axis limits
        plt.xlim([-0.05, 1.05])
        plt.ylim([0.9, 2000])  #Adjust as needed  
        
        # Add grid
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        if np.mean(self.genuine_scores) > np.mean(self.impostor_scores):
            legend_loc = "upper right"
        else:
            legend_loc = "upper left"
        
        plt.legend(loc=legend_loc, fontsize=12)
        
        #Labels and title
        plt.xlabel('Matching Score', fontsize=12, weight='bold')
        plt.ylabel('Frequency', fontsize=12, weight='bold')
        
        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set font size for ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Title with d-prime and system info
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                (self.get_dprime(), self.plot_title),
                fontsize=15, weight='bold')
        
        # Save and show the plot
        plt.savefig(f'score_distribution_plot_{self.plot_title}.png', dpi=300, bbox_inches="tight")
       # plt.show()
        plt.close()
        
        return
        
    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
        EER is the point where FPR = FNR (false positive rate equals false negative rate).
        Lower EER means better biometric system performance.
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """

        # Convert to numpy arrays for calculations
        FPR = np.array(FPR)
        FNR = np.array(FNR)
        
        #absolute difference between FPR and FNR
        diff = np.abs(FPR - FNR)
        # index where the difference is smallest(this is where FPR ≈ FNR)
        index = np.argmin(diff)
        
        #EER is the average of FPR and FNR at this point
        EER = (FPR[index] + FNR[index]) / 2
        
        return EER, self.thresholds[index]  # Return EER and corresponding threshold

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Shows the trade-off between false positives and false negatives.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER, _ = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            FPR,   #FPR on x-axis
            FNR,  #FNR on y-axis
            lw=2, # width of 2 pixels
            color='grey' #grey curve
        )
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve
        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
    
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            color='gray', #gray color for grid lines
            linestyle='--', #dashed line style
            linewidth=0.5 #width of 0.5 pixels
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x-axis label with specified font size and weight
        plt.xlabel(
            'False Positive Rate', #label for x-axis
            fontsize=12, #12 points
            weight='bold' #font weight
        )
        
        # Set y-axis label with specified font size and weight
        plt.ylabel(
            'False Negative Rate',#abel for y-axis
            fontsize=12, #12 points
            weight="bold" #bold font weight
        )
        
        # Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s' %
            (EER, self.plot_title),
            fontsize=15,
            weight='bold'
        )
        
        # Set font size for x and y-axis ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Save the plot as an image file
        plt.savefig(f'det_curve_plot_{self.plot_title}.png', dpi=300, bbox_inches="tight")
        
        # Display the plot
       # plt.show()
        
        # Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Shows the trade-off between true positives and false positives.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for the ROC curve
        plt.figure()
        
        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(
            FPR, #FPR values on the x-axis
            TPR, #TPR values on the y-axis
            lw=2, #line width of 2 pixels
            color='grey' #grey curve
        )
        
        # Calculate AUC (Area Under Curve)
        auc = metrics.auc(FPR, TPR)
        
        # Set x and y axis limits
        plt.xlim([-0.05, 1.05]) 
        plt.ylim([-0.05, 1.05])

        # Add grid
        plt.grid(color='gray', linestyle='--', linewidth=0.5) 

        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False) 
        plt.gca().spines['right'].set_visible(False)
        
        # Set labels for x and y axes, and add a title
        plt.xlabel(
            'False Positive Rate',
            fontsize=12,
            weight='bold'
        )
        
        plt.ylabel(
            'True Positive Rate',
            fontsize=12,
            weight='bold'
        )
        
        plt.title(
            f'Receiver Operating Characteristic Curve\nAUC = {auc:.4f}\nSystem {self.plot_title}',
            fontsize=15,
            weight='bold'
        )

        # Set font sizes for ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Save and display the plot
        plt.savefig(f'roc_curve_plot_{self.plot_title}.png', dpi=300, bbox_inches="tight")
      #  plt.show()
        plt.close()
 
        return auc  # Return Area Under Curve

    def compute_rates(self):
        """
        Compute performance metrics (FPR, FNR, TPR) across all thresholds.
        
        Returns:
        - FPR: List of False Positive Rates for each threshold
        - FNR: List of False Negative Rates for each threshold
        - TPR: List of True Positive Rates for each threshold
        """
        # Initialize lists for False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR)
        FPR = []
        FNR = []
        TPR = []

        # Iterate through threshold values and calculate TP, FP, TN, and FN for each threshold
        for threshold in self.thresholds:
            # True Positives are genuine scores correctly classified as genuine (above threshold)
            TP = np.sum(self.genuine_scores >= threshold)
            
            # False positives are impostor scores incorrectly classified as genuine (above threshold)
            FP = np.sum(self.impostor_scores >= threshold)
            
            # True negatives are impostor scores correctly classified as impostor (below threshold)
            TN = np.sum(self.impostor_scores < threshold)
            
            # False negatives are genuine scores incorrectly classified as impostor (below threshold)
            FN = np.sum(self.genuine_scores < threshold)
            
            # Calculate FPR, FNR, and TPR
            fpr = FP / (FP + TN + self.epsilon)
            fnr = FN / (TP + FN + self.epsilon)
            tpr = TP / (TP + FN + self.epsilon)
            
            # Append calculated rates to their respective lists
            FPR.append(fpr)
            FNR.append(fnr)
            TPR.append(tpr)
            
        # Return the lists of FPR, FNR, and TPR
        return FPR, FNR, TPR


class FacialLandmarkExtractor:
    """
    A class for extracting facial landmarks from images.
    """
    
    def __init__(self, predictor_path):
        """
        Initialize the face detector and landmark predictor.
        
        Parameters:
        - predictor_path (str): Path to the facial landmark predictor model
        """
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Check if the file exists
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"The shape predictor file doesn't exist at: {predictor_path}")
            
        self.predictor = dlib.shape_predictor(predictor_path)
    
    def extract_landmarks(self, image_path, landmark_indices=None):
        """
        Extract facial landmarks from an image.
        
        Parameters:
        - image_path (str): Path to the image file.
        - landmark_indices (list, optional): Indices of landmarks to extract. If None, all 68 landmarks are extracted.
        
        Returns:
        - np.array: Array of facial landmark coordinates (each with x,y coordinates)
                   or None if no face is detected.
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return None
        
        # Get the first face (assuming one face per image)
        face = faces[0]
        
        # Get facial landmarks
        shape = self.predictor(gray, face)
        
        # If no specific landmarks are requested, get all 68
        if landmark_indices is None:
            landmark_indices = range(68)
            
        # Convert specified landmarks to numpy array
        num_landmarks = len(landmark_indices)
        landmarks = np.zeros((num_landmarks, 2), dtype=np.float32)
        
        for i, idx in enumerate(landmark_indices):
            landmarks[i] = (shape.part(idx).x, shape.part(idx).y)
        
        return landmarks
    
    def extract_landmarks_from_directory(self, directory_path, landmark_indices=None):
        """
        Extract facial landmarks from all images in a directory.
        
        Parameters:
        - directory_path (str): Path to the directory containing images.
        - landmark_indices (list, optional): Indices of landmarks to extract. If None, all 68 landmarks are extracted.
        
        Returns:
        - X (np.array): Array of facial landmarks for all images where faces were detected.
        - y (np.array): Array of identity labels (based on subdirectory names).
        - file_paths (list): List of file paths corresponding to the extracted features.
        """
        X = []
        y = []
        file_paths = []
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Get all subdirectories (assuming each subdirectory represents a person)
        person_dirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        
        print(f"Found {len(person_dirs)} person directories")
        
        # Iterate through each person's directory
        for person_id, person_dir in enumerate(tqdm(person_dirs, desc="Processing directories")):
            person_path = os.path.join(directory_path, person_dir)
            
            # Get all image files in the person's directory
            image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                          glob.glob(os.path.join(person_path, "*.jpeg")) + \
                          glob.glob(os.path.join(person_path, "*.png"))
            
            print(f"Processing {len(image_files)} images for person {person_dir}")
            
            # Process each image
            for image_file in tqdm(image_files, desc=f"Person {person_dir}"):
                landmarks = self.extract_landmarks(image_file, landmark_indices)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(person_id)
                    file_paths.append(image_file)
        
        # Make sure we have at least some data
        if len(X) == 0:
            raise ValueError("No faces were detected in any of the images!")
            
        return np.array(X), np.array(y), file_paths


class AgeFeatureExtractor:
    """
    A class for extracting age-related features from facial images.
    This is a simplified implementation for demonstration purposes.
    """
    
    def __init__(self, age_model_path=None):
        """
        Initialize the age feature extractor.
        
        Parameters:
        - age_model_path (str, optional): Path to a pre-trained age estimation model.
                                        If None, a simple heuristic approach is used.
        """
        self.model = None
        if age_model_path and os.path.exists(age_model_path):
            # Load pre-trained model (implementation depends on model type)
            pass
        
    def extract_age_features(self, image_path):
        """
        Extract age-related features from a facial image.
        This implementation uses simple image processing techniques as a proxy for age features.
        
        Parameters:
        - image_path (str): Path to the facial image.
        
        Returns:
        - np.array: Age-related features or None if face detection fails.
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use edge detection as a proxy for wrinkles (more edges may indicate more wrinkles)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate the percentage of edge pixels as a feature
        edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate texture features using Local Binary Patterns (LBP) or Gabor filters
        # This is simplified here - just using basic histogram of image intensity
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist = hist.flatten() / np.sum(hist)  # Normalize
        
        # Combine features
        features = np.append(edge_percentage, hist)
        
        return features
    
    def extract_age_features_from_directory(self, directory_path):
        """
        Extract age-related features from all images in a directory.
        
        Parameters:
        - directory_path (str): Path to the directory containing images.
        
        Returns:
        - X (np.array): Array of age-related features for all images.
        - y (np.array): Array of identity labels (based on subdirectory names).
        - file_paths (list): List of file paths corresponding to the extracted features.
        """
        X = []
        y = []
        file_paths = []
        
        # Get all subdirectories (assuming each subdirectory represents a person)
        person_dirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        
        print(f"Found {len(person_dirs)} person directories for age feature extraction")
        
        # Iterate through each person's directory
        for person_id, person_dir in enumerate(tqdm(person_dirs, desc="Processing directories for age features")):
            person_path = os.path.join(directory_path, person_dir)
            
            # Get all image files in the person's directory
            image_files = glob.glob(os.path.join(person_path, "*.jpg")) + \
                          glob.glob(os.path.join(person_path, "*.jpeg")) + \
                          glob.glob(os.path.join(person_path, "*.png"))
            
            # Process each image
            for image_file in tqdm(image_files, desc=f"Age features for {person_dir}"):
                features = self.extract_age_features(image_file)
                if features is not None:
                    X.append(features)
                    y.append(person_id)
                    file_paths.append(image_file)
        
        if len(X) == 0:
            raise ValueError("No features could be extracted from the images!")
            
        return np.array(X), np.array(y), file_paths


def extract_landmark_distance_features(raw_data):
    """
    Extract features from facial landmarks using Manhattan distances
    between every pair of points
    
    Parameters:
    raw_data: Array of facial landmarks (shape: num_samples x num_landmarks x 2)
    
    Returns:
    features: Array of features (shape: num_samples x num_features)
    """
    num_samples = raw_data.shape[0]
    features = []
    
    for k in range(num_samples):
        person_k = raw_data[k]
        features_k = []
        for i in range(person_k.shape[0]):
            for j in range(i+1, person_k.shape[0]):  # Only compute unique pairs
                p1 = person_k[i,:]
                p2 = person_k[j,:]
                # Calculate Manhattan distance between landmarks
                features_k.append(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
        features.append(features_k)
    
    return np.array(features)


def extract_landmark_ratio_features(raw_data):
    """
    Extract ratio-based features from facial landmarks to make them more
    scale-invariant.
    
    Parameters:
    raw_data: Array of facial landmarks (shape: num_samples x num_landmarks x 2)
    
    Returns:
    features: Array of features (shape: num_samples x num_features)
    """
    num_samples = raw_data.shape[0]
    features = []
    
    for k in range(num_samples):
        person_k = raw_data[k]
        
        # Find face dimensions (width and height)
        min_x = np.min(person_k[:, 0])
        max_x = np.max(person_k[:, 0])
        min_y = np.min(person_k[:, 1])
        max_y = np.max(person_k[:, 1])
        face_width = max_x - min_x
        face_height = max_y - min_y
        face_diagonal = np.sqrt(face_width**2 + face_height**2)
        
        # Center of the face
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Calculate features
        features_k = []
        
        # Distance from each point to the center, normalized by face diagonal
        for i in range(person_k.shape[0]):
            p = person_k[i, :]
            dist_to_center = np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) / face_diagonal
            features_k.append(dist_to_center)
        
        # Ratios of selected distances between landmark pairs
        for i in range(person_k.shape[0]):
            for j in range(i+1, person_k.shape[0], 3):  # Skip some pairs to reduce dimensionality
                p1 = person_k[i, :]
                p2 = person_k[j, :]
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) / face_diagonal
                features_k.append(dist)
        
        features.append(features_k)
    
    return np.array(features)


def normalize_features(features):
    """
    Normalize features using Min-Max scaling to [0,1] range.
    
    Parameters:
    features: Array of features (shape: num_samples x num_features)
    
    Returns:
    normalized_features: Normalized array of features
    """
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(features)


class BiometricFusion:
    """
    Class for implementing different fusion strategies in biometric systems.
    """
    
    @staticmethod
    def feature_level_fusion(feature_sets):
        """
        Implement feature-level fusion by concatenating features from different sources.
        
        Parameters:
        feature_sets: List of feature arrays to fuse
        
        Returns:
        fused_features: Concatenated and normalized feature array
        """
        # Ensure all feature sets have the same number of samples
        num_samples = feature_sets[0].shape[0]
        for features in feature_sets:
            if features.shape[0] != num_samples:
                raise ValueError("All feature sets must have the same number of samples")
        
        # Normalize each feature set separately
        normalized_features = [normalize_features(features) for features in feature_sets]
        
        # Concatenate the features
        fused_features = np.hstack(normalized_features)
        
        return fused_features
    
    @staticmethod
    def score_level_fusion(score_sets, weights=None):
        """
        Implement score-level fusion by combining matching scores from different modalities.
        
        Parameters:
        score_sets: List of score arrays to fuse
        weights: List of weights for each score set (default: equal weights)
        
        Returns:
        fused_scores: Combined scores
        """
        num_sets = len(score_sets)
        
        # If weights are not provided, use equal weights
        if weights is None:
            weights = np.ones(num_sets) / num_sets
        else:
            # Normalize weights to sum to 1
            weights = np.array(weights) / np.sum(weights)
        
        # Normalize each score set to [0,1] range
        normalized_scores = []
        for scores in score_sets:
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score > min_score:
                norm_scores = (scores - min_score) / (max_score - min_score)
            else:
                norm_scores = np.zeros_like(scores)
            normalized_scores.append(norm_scores)
        
        # Weighted sum fusion
        fused_scores = np.zeros_like(normalized_scores[0])
        for i, scores in enumerate(normalized_scores):
            fused_scores += weights[i] * scores
        
        return fused_scores
    
    @staticmethod
    def decision_level_fusion(decisions, fusion_method='majority_vote'):
        """
        Implement decision-level fusion by combining binary decisions from different modalities.
        
        Parameters:
        decisions: List of binary decision arrays (1 for accept, 0 for reject)
        fusion_method: Method to use for fusion ('majority_vote', 'AND', 'OR')
        
        Returns:
        fused_decisions: Combined binary decisions
        """
        decisions = np.array(decisions)
        
        if fusion_method == 'majority_vote':
            # Sum decisions across modalities and apply threshold
            decision_sum = np.sum(decisions, axis=0)
            fused_decisions = (decision_sum >= np.ceil(len(decisions) / 2)).astype(int)
        
        elif fusion_method == 'AND':
            # Accept only if all modalities accept
            fused_decisions = np.all(decisions, axis=0).astype(int)
        
        elif fusion_method == 'OR':
            # Accept if any modality accepts
            fused_decisions = np.any(decisions, axis=0).astype(int)
        
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        return fused_decisions


def run_landmark_experiment(image_folder, predictor_path, landmark_config, config_name, age_based=False):
    """
    Run facial recognition experiment with a specific landmark configuration.
    
    Parameters:
    - image_folder (str): Path to the folder containing face images
    - predictor_path (str): Path to the dlib shape predictor model
    - landmark_config (list): List of landmark indices to use
    - config_name (str): Name of the configuration for plotting
    - age_based (bool): Whether to include age-based features in fusion
    
    Returns:
    - dict: Dictionary containing experiment results
    """
    # Initialize the facial landmark extractor
    print(f"Initializing facial landmark extractor for {config_name}...")
    extractor = FacialLandmarkExtractor(predictor_path)
    
    # Extract landmarks using the specified configuration
    print(f"Extracting landmarks from {image_folder}...")
    X_raw, y, file_paths = extractor.extract_landmarks_from_directory(image_folder, landmark_config)
    
    # Extract distance-based features from landmarks
    print("Extracting distance features...")
    X_distances = extract_landmark_distance_features(X_raw)
    
    # Extract ratio-based features from landmarks
    print("Extracting ratio features...")
    X_ratios = extract_landmark_ratio_features(X_raw)
    
    # Initialize feature sets for fusion
    feature_sets = [X_distances, X_ratios]
    
    # If age-based features are requested, extract them
    if age_based:
        print("Extracting age-based features...")
        age_extractor = AgeFeatureExtractor()
        X_age, y_age, _ = age_extractor.extract_age_features_from_directory(image_folder)
        
        # Ensure we have age features for the same images as landmark features
        # This assumes file_paths are in the same order for both feature extraction methods
        if len(X_age) == len(X_distances):
            feature_sets.append(X_age)
        else:
            print("Warning: Age features could not be matched with landmark features. Skipping fusion.")
    
    # Perform feature-level fusion
    print("Performing feature-level fusion...")
    X_fused = BiometricFusion.feature_level_fusion(feature_sets)
    
    # Normalize individual feature sets for comparison
    X_distances_norm = normalize_features(X_distances)
    X_ratios_norm = normalize_features(X_ratios)
    
    # Split data into training and testing sets
    test_size = 0.3
    random_state = 42
    
    # Split for distance features
    X_dist_train, X_dist_test, y_dist_train, y_dist_test = train_test_split(
        X_distances_norm, y, test_size=test_size, random_state=random_state)
    
    # Split for ratio features
    X_ratio_train, X_ratio_test, y_ratio_train, y_ratio_test = train_test_split(
        X_ratios_norm, y, test_size=test_size, random_state=random_state)
    
    # Split for fused features
    X_fused_train, X_fused_test, y_fused_train, y_fused_test = train_test_split(
        X_fused, y, test_size=test_size, random_state=random_state)
    
    # Initialize classifiers
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = ORC(SVC(kernel='rbf', probability=True))
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Train and evaluate using distance features
    print("Training and evaluating with distance features...")
    knn.fit(X_dist_train, y_dist_train)
    svm.fit(X_dist_train, y_dist_train)
    rf.fit(X_dist_train, y_dist_train)
    
    dist_scores_knn = knn.predict_proba(X_dist_test)
    dist_scores_svm = svm.predict_proba(X_dist_test)
    dist_scores_rf = rf.predict_proba(X_dist_test)
    
    # Train and evaluate using ratio features
    print("Training and evaluating with ratio features...")
    knn.fit(X_ratio_train, y_ratio_train)
    svm.fit(X_ratio_train, y_ratio_train)
    rf.fit(X_ratio_train, y_ratio_train)
    
    ratio_scores_knn = knn.predict_proba(X_ratio_test)
    ratio_scores_svm = svm.predict_proba(X_ratio_test)
    ratio_scores_rf = rf.predict_proba(X_ratio_test)
    
    # Train and evaluate using fused features
    print("Training and evaluating with fused features...")
    knn.fit(X_fused_train, y_fused_train)
    svm.fit(X_fused_train, y_fused_train)
    rf.fit(X_fused_train, y_fused_train)
    
    fused_scores_knn = knn.predict_proba(X_fused_test)
    fused_scores_svm = svm.predict_proba(X_fused_test)
    fused_scores_rf = rf.predict_proba(X_fused_test)
    
    # Prepare data for evaluating biometric performance
    genuine_scores = []
    impostor_scores = []
    
    # Use SVM with fused features for biometric evaluation
    # For each test sample, get the confidence score for its true class
    for i, sample in enumerate(X_fused_test):
        true_class = y_fused_test[i]
        sample_reshaped = sample.reshape(1, -1)
        probabilities = svm.predict_proba(sample_reshaped)[0]
        
        # Find the probability for the true class
        class_indices = svm.classes_
        true_class_idx = np.where(class_indices == true_class)[0]
        
        if len(true_class_idx) > 0:
            true_class_prob = probabilities[true_class_idx[0]]
            genuine_scores.append(true_class_prob)
            
            # Get highest probability for any other class (impostor)
            impostor_probs = np.delete(probabilities, true_class_idx)
            if len(impostor_probs) > 0:
                impostor_scores.append(np.max(impostor_probs))
    
    # Evaluate biometric performance
    print("Evaluating biometric performance...")
    evaluator = Evaluator(
        num_thresholds=100,
        genuine_scores=np.array(genuine_scores),
        impostor_scores=np.array(impostor_scores),
        plot_title=config_name
    )
    
    # Compute performance metrics
    FPR, FNR, TPR = evaluator.compute_rates()
    
    # Calculate EER
    EER, threshold = evaluator.get_EER(FPR, FNR)
    
    # Calculate AUC
    AUC = evaluator.plot_roc_curve(FPR, TPR)
    
    # Plot DET curve
    evaluator.plot_det_curve(FPR, FNR)
    
    # Plot score distribution
    evaluator.plot_score_distribution()
    
    # Calculate d-prime
    dprime = evaluator.get_dprime()
    
    # Return results
    results = {
        'config_name': config_name,
        'EER': EER,
        'AUC': AUC,
        'dprime': dprime,
        'threshold': threshold,
        'genuine_scores': genuine_scores,
        'impostor_scores': impostor_scores,
        'feature_dimension': X_fused.shape[1]
    }
    
    print(f"Experiment with {config_name} completed. Results:")
    print(f"- EER: {EER:.4f}")
    print(f"- AUC: {AUC:.4f}")
    print(f"- d-prime: {dprime:.4f}")
    print(f"- Feature dimension: {X_fused.shape[1]}")
    
    return results


def main():
    """
    Main function to run facial recognition experiments with different landmark configurations.
    """
    # Paths
    image_folder = r"C:\Users\clutc\Desktop\py\caltech 1999"
    predictor_path = r"C:\Users\clutc\Desktop\py\caltech 1999\shape_predictor_68_face_landmarks.dat"

    
    # Define landmark configurations to test
    configurations = {
        "full_face": list(range(68)),              # All 68 landmarks
        "eyes_only": list(range(36, 48)),          # Eye landmarks (36-47)
        "mouth_only": list(range(48, 68)),         # Mouth landmarks (48-67)
        "jawline": list(range(0, 17)),             # Jawline landmarks (0-16)
        "nose": list(range(27, 36)),               # Nose landmarks (27-35)
        "eyes_nose": list(range(27, 48)),          # Eyes and nose (27-47)
        "eyes_mouth": list(range(36, 48)) + list(range(48, 68))  # Eyes and mouth
    }
    
    # Store results for comparison
    all_results = {}
    
    # Run experiments for each configuration
    for config_name, landmark_indices in configurations.items():
        print(f"\n{'='*50}\nRunning experiment for {config_name}\n{'='*50}")
        results = run_landmark_experiment(
            image_folder=image_folder,
            predictor_path=predictor_path,
            landmark_config=landmark_indices,
            config_name=config_name,
            age_based=(config_name == "full_face")  # Only use age features for full face
        )
        all_results[config_name] = results
    
    # Compare results across configurations
    print("\n\nComparison of all configurations:")
    print("-" * 80)
    print(f"{'Configuration':<15} {'EER':<10} {'AUC':<10} {'d-prime':<10} {'Feature Dim':<10}")
    print("-" * 80)
    
    for config_name, results in all_results.items():
        print(f"{config_name:<15} {results['EER']:<10.4f} {results['AUC']:<10.4f} {results['dprime']:<10.4f} {results['feature_dimension']:<10}")
    
    # Create comparison plots
    plt.figure(figsize=(12, 8))
    
    # EER comparison
    plt.subplot(2, 2, 1)
    configs = list(all_results.keys())
    eers = [all_results[c]['EER'] for c in configs]
    plt.bar(configs, eers)
    plt.title('Equal Error Rate (EER) Comparison')
    plt.ylabel('EER (lower is better)')
    plt.xticks(rotation=45, ha='right')
    
    # AUC comparison
    plt.subplot(2, 2, 2)
    aucs = [all_results[c]['AUC'] for c in configs]
    plt.bar(configs, aucs)
    plt.title('Area Under Curve (AUC) Comparison')
    plt.ylabel('AUC (higher is better)')
    plt.xticks(rotation=45, ha='right')
    
    # d-prime comparison
    plt.subplot(2, 2, 3)
    dprimes = [all_results[c]['dprime'] for c in configs]
    plt.bar(configs, dprimes)
    plt.title('d-prime Comparison')
    plt.ylabel('d-prime (higher is better)')
    plt.xticks(rotation=45, ha='right')
    
    # Feature dimension comparison
    plt.subplot(2, 2, 4)
    dims = [all_results[c]['feature_dimension'] for c in configs]
    plt.bar(configs, dims)
    plt.title('Feature Dimension Comparison')
    plt.ylabel('Number of features')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('landmark_configuration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExperiment completed successfully! Results saved to landmark_configuration_comparison.png")


if __name__ == "__main__":
    main()
