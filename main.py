'''
Improved Face Recognition Biometric Authentication System for IMDB Dataset

This system creates user templates from 10 training faces per identity and tests against
a mix of genuine and impostor faces (up to 100 total) to evaluate biometric performance.
Genuine matches are based on celebrity names from the metadata.

Fixed to address uniform prediction scores issue.
'''

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Import necessary libraries
import numpy as np
import os
import cv2
import scipy.io as sio
import h5py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import random
import time

'''
    Configuration
'''
# Path to your IMDB dataset
IMDB_PATH = r"C:\Users\dylan\Documents\VS CODE\Python\BIOM_GP\imdb_crop"
IMDB_MAT_PATH = r"C:\Users\dylan\Documents\VS CODE\Python\BIOM_GP\imdb_crop\imdb.mat"

# Decision threshold for authentication
MATCH_THRESHOLD = 0.5

# Maximum images to process in total
MAX_TOTAL_IMAGES = 10000

# Training/testing configuration
TRAINING_FACES_PER_IDENTITY = 10  # Number of faces to use for template training
MIN_TESTING_GENUINE_FACES = 1     # Minimum number of genuine faces to test against
MAX_TESTING_FACES = 100           # Maximum number of total test faces per identity

# Maximum number of identities to evaluate
MAX_TEST_IDENTITIES = 50

# Classifier choice - options: 'svm', 'rf', 'knn'
CLASSIFIER_TYPE = 'svm'

'''
    Feature Extraction Improvements
'''
def extract_hog_features(image):
    """Extract HOG features for better face recognition"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to standard size
    resized = cv2.resize(gray, (64, 64))
    
    # Calculate HOG features (simplified)
    # In a real implementation, use proper HOG with cv2.HOGDescriptor
    # For simplicity, we'll use basic gradient calculation
    
    # Calculate gradients
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # Bin angles into 8 directions
    num_bins = 8
    bin_size = 360 // num_bins
    angles = ang * 180 / np.pi
    bins = ((angles % 360) // bin_size).astype(int)
    
    # Create histogram for each cell (8x8 cells)
    cell_size = 8
    num_cells_x = resized.shape[1] // cell_size
    num_cells_y = resized.shape[0] // cell_size
    hist_features = []
    
    for y in range(num_cells_y):
        for x in range(num_cells_x):
            # Cell coordinates
            x_start = x * cell_size
            y_start = y * cell_size
            x_end = min(x_start + cell_size, resized.shape[1])
            y_end = min(y_start + cell_size, resized.shape[0])
            
            # Get magnitudes and bins for this cell
            cell_mags = mag[y_start:y_end, x_start:x_end]
            cell_bins = bins[y_start:y_end, x_start:x_end]
            
            # Create histogram
            hist = np.zeros(num_bins)
            for b in range(num_bins):
                hist[b] = np.sum(cell_mags[cell_bins == b])
            
            # Normalize histogram
            if np.sum(hist) > 0:
                hist = hist / np.sum(hist)
            
            hist_features.extend(hist)
    
    # Add LBP-like features for texture
    lbp_features = []
    step = 8
    for y in range(0, resized.shape[0] - step, step):
        for x in range(0, resized.shape[1] - step, step):
            # Get center pixel and 8 neighbors
            center = resized[y + step//2, x + step//2]
            neighbors = [
                resized[y, x], resized[y, x + step], 
                resized[y + step, x + step], resized[y + step, x]
            ]
            
            # Calculate simple texture pattern
            pattern = 0
            for i, neighbor in enumerate(neighbors):
                if neighbor >= center:
                    pattern |= (1 << i)
            
            lbp_features.append(pattern)
    
    # Combine HOG and LBP features
    combined_features = np.array(hist_features + lbp_features)
    
    return combined_features

'''
    Face Processing Functions
'''
def detect_face(image):
    """Simple face detection function that returns the original image"""
    # For simplicity, assume the images are already face crops
    return image

def extract_face_features(image):
    """Extract features from a face image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to a standard size
    resized = cv2.resize(gray, (64, 64))
    
    # Extract HOG-like features
    return extract_hog_features(resized)

'''
    Load IMDB Dataset with Metadata
'''
def load_imdb_dataset():
    """
    Load the IMDB dataset using the metadata from the .mat file
    Returns:
        X: feature vectors for each face
        y: identity labels (based on celeb_name)
        identity_to_idx: mapping from identity to index
        idx_to_identity: mapping from index to identity
        image_metadata: additional metadata for each image
    """
    print("Loading IMDB dataset...")
    
    try:
        # Try to load the matlab file
        mat_data = sio.loadmat(IMDB_MAT_PATH)
        print("Successfully loaded .mat file")
        
        # Extract metadata
        if 'imdb' in mat_data:
            imdb_data = mat_data['imdb'][0, 0]
            fields = imdb_data.dtype.names
            print(f"Available fields: {fields}")
            
            # Check if celeb_names is available
            if 'celeb_names' in fields and 'celeb_id' in fields:
                # Extract celebrity names and IDs
                try:
                    celeb_names = []
                    for name in imdb_data['celeb_names'][0]:
                        try:
                            celeb_names.append(str(name[0]))
                        except:
                            celeb_names.append(f"unknown_{len(celeb_names)}")
                    
                    celeb_ids = [int(id_val) for id_val in imdb_data['celeb_id'][0]]
                    
                    # Create mapping from celeb_id to celeb_name
                    celeb_id_to_name = {}
                    for i, name in enumerate(celeb_names):
                        if i < len(celeb_ids):
                            celeb_id_to_name[i] = name
                    
                    # Extract paths, face scores, and other metadata
                    paths = []
                    face_scores = []
                    identities = []  # Based on celeb_names
                    
                    # Process in batches to handle large datasets
                    batch_size = 10000
                    total_images = len(imdb_data['full_path'][0])
                    num_batches = (total_images + batch_size - 1) // batch_size
                    
                    for batch in range(num_batches):
                        start_idx = batch * batch_size
                        end_idx = min(start_idx + batch_size, total_images)
                        
                        for i in range(start_idx, end_idx):
                            try:
                                paths.append(str(imdb_data['full_path'][0][i][0]))
                                face_scores.append(float(imdb_data['face_score'][0][i]))
                                
                                # Get identity from celeb_id
                                celeb_id = int(imdb_data['celeb_id'][0][i]) if i < len(imdb_data['celeb_id'][0]) else -1
                                if celeb_id in celeb_id_to_name:
                                    identities.append(celeb_id_to_name[celeb_id])
                                else:
                                    identities.append(f"unknown_{celeb_id}")
                            except Exception as e:
                                # Skip this image if there's an error
                                continue
                    
                    print(f"Extracted metadata for {len(identities)} images")
                    return process_metadata(identities, paths, face_scores)
                except Exception as e:
                    print(f"Error extracting metadata: {e}")
            
            # Use names field directly if celeb_names not available
            return process_metadata_without_celeb(imdb_data)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
    
    # If we reach here, use directory approach
    print("Falling back to directory-based processing...")
    return process_directories()

def process_metadata(identities, paths, face_scores):
    """Process metadata to create dataset with identity labels"""
    print("Processing metadata...")
    
    # Ensure arrays have the same length
    min_len = min(len(identities), len(paths), len(face_scores))
    identities = identities[:min_len]
    paths = paths[:min_len]
    face_scores = face_scores[:min_len]
    
    # Count images per identity
    identity_counts = {}
    for identity in identities:
        if identity in identity_counts:
            identity_counts[identity] += 1
        else:
            identity_counts[identity] = 1
    
    # Filter identities with enough images for our testing protocol
    min_images_needed = TRAINING_FACES_PER_IDENTITY + MIN_TESTING_GENUINE_FACES
    valid_identities = [name for name, count in identity_counts.items() 
                       if count >= min_images_needed]
    
    print(f"Found {len(valid_identities)} identities with at least {min_images_needed} images")
    
    # Create identity mapping
    identity_to_idx = {name: idx for idx, name in enumerate(valid_identities)}
    idx_to_identity = {idx: name for name, idx in identity_to_idx.items()}
    
    # Create dataset
    dataset = {}  # Map from identity index to list of tuples (image_path, face_score)
    
    for i in range(len(identities)):
        identity = identities[i]
        path = paths[i]
        face_score = face_scores[i]
        
        # Only use valid identities with good face scores
        if identity in identity_to_idx and face_score > 0.5:
            identity_idx = identity_to_idx[identity]
            
            if identity_idx not in dataset:
                dataset[identity_idx] = []
            
            # Extract filename from path
            filename = os.path.basename(path)
            dataset[identity_idx].append((filename, face_score))
    
    # Process the images to create X and y
    return process_images(dataset, identity_to_idx, idx_to_identity)

def process_metadata_without_celeb(imdb_data):
    """Process metadata when celeb_names is not available"""
    print("Celebrity names not available, using regular names instead...")
    
    # Extract names, paths, and scores
    names = []
    paths = []
    face_scores = []
    
    try:
        if 'name' in imdb_data.dtype.names:
            for name in imdb_data['name'][0]:
                names.append(str(name[0]) if isinstance(name[0], str) else name[0].item())
    except Exception as e:
        print(f"Error extracting names: {e}")
        names = [f"person_{i}" for i in range(len(imdb_data['full_path'][0]))]
    
    try:
        if 'full_path' in imdb_data.dtype.names:
            for path in imdb_data['full_path'][0]:
                paths.append(str(path[0]) if isinstance(path[0], str) else path[0].item())
    except Exception as e:
        print(f"Error extracting paths: {e}")
        return process_directories()
    
    try:
        if 'face_score' in imdb_data.dtype.names:
            for score in imdb_data['face_score'][0]:
                face_scores.append(float(score) if not np.isnan(score) else 0.0)
    except Exception as e:
        print(f"Error extracting face scores: {e}")
        face_scores = [1.0] * len(paths)
    
    return process_metadata(names, paths, face_scores)

def process_directories():
    """Process images using directory structure as identity"""
    print("Creating synthetic identities from directory structure...")
    
    # Create dataset mapping
    dataset = {}
    identity_to_idx = {}
    idx_to_identity = {}
    
    # Scan directories
    for subfolder in range(1, 100):
        folder_path = os.path.join(IMDB_PATH, str(subfolder))
        if os.path.exists(folder_path):
            # Create synthetic identity from folder
            identity = f"folder_{subfolder}"
            identity_idx = len(identity_to_idx)
            identity_to_idx[identity] = identity_idx
            idx_to_identity[identity_idx] = identity
            
            # Find all images in this folder
            images = []
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append((file, 1.0))  # Assume maximum face score
            
            # Only use folders with enough images
            min_images_needed = TRAINING_FACES_PER_IDENTITY + MIN_TESTING_GENUINE_FACES
            if len(images) >= min_images_needed:
                dataset[identity_idx] = images
    
    print(f"Found {len(dataset)} synthetic identities with enough images")
    
    # Process the images to create X and y
    return process_images(dataset, identity_to_idx, idx_to_identity)

def process_images(dataset, identity_to_idx, idx_to_identity):
    """Process images to create feature vectors and labels"""
    print("Processing images...")
    
    # Limit the number of identities to test
    identity_indices = list(dataset.keys())
    if MAX_TEST_IDENTITIES and len(identity_indices) > MAX_TEST_IDENTITIES:
        identity_indices = random.sample(identity_indices, MAX_TEST_IDENTITIES)
    
    # Track total number of images processed
    total_processed = 0
    
    # Create X and y arrays
    X = []  # Feature vectors
    y = []  # Identity labels
    metadata = []  # Additional metadata for each image
    
    for identity_idx in identity_indices:
        if total_processed >= MAX_TOTAL_IMAGES:
            print(f"Reached maximum number of images ({MAX_TOTAL_IMAGES})")
            break
            
        image_files = dataset[identity_idx]
        
        # Limit to reasonable number of images per identity
        max_images = TRAINING_FACES_PER_IDENTITY + MAX_TESTING_FACES
        if len(image_files) > max_images:
            image_files = random.sample(image_files, max_images)
        
        identity_processed = 0
        identity_name = idx_to_identity[identity_idx]
        
        for filename, face_score in image_files:
            if total_processed >= MAX_TOTAL_IMAGES:
                break
                
            # Find the image in the filesystem
            found = False
            image_path = None
            
            # Search in all subfolders
            for subfolder in range(1, 100):
                folder_path = os.path.join(IMDB_PATH, str(subfolder))
                if os.path.exists(folder_path):
                    potential_path = os.path.join(folder_path, filename)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        found = True
                        break
            
            if found:
                try:
                    # Load and process image
                    image = cv2.imread(image_path)
                    if image is not None:
                        face = detect_face(image)
                        features = extract_face_features(face)
                        
                        X.append(features)
                        y.append(identity_idx)
                        metadata.append({
                            'filename': filename,
                            'identity': identity_name,
                            'face_score': face_score,
                            'path': image_path
                        })
                        
                        identity_processed += 1
                        total_processed += 1
                        
                        # Print progress occasionally
                        if total_processed % 500 == 0:
                            print(f"Processed {total_processed} images...")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    print(f"Processed {total_processed} images for {len(set(y))} identities")
    
    # Filter identities with enough images after processing
    identity_counts = {}
    for idx in y:
        if idx in identity_counts:
            identity_counts[idx] += 1
        else:
            identity_counts[idx] = 1
    
    # Only keep identities with enough images for our testing protocol
    min_images_needed = TRAINING_FACES_PER_IDENTITY + MIN_TESTING_GENUINE_FACES
    valid_indices = []
    
    for i in range(len(y)):
        if identity_counts[y[i]] >= min_images_needed:
            valid_indices.append(i)
    
    # Filter X and y
    X_filtered = [X[i] for i in valid_indices]
    y_filtered = [y[i] for i in valid_indices]
    metadata_filtered = [metadata[i] for i in valid_indices]
    
    print(f"After filtering: {len(X_filtered)} images for {len(set(y_filtered))} identities")
    
    return np.array(X_filtered), np.array(y_filtered), identity_to_idx, idx_to_identity, metadata_filtered

'''
    Biometric Authentication Evaluation with improved classifiers
'''
def get_classifier():
    """Return the appropriate classifier based on configuration"""
    if CLASSIFIER_TYPE == 'svm':
        # SVM with probability calibration
        base_clf = SVC(kernel='rbf', C=10, gamma='scale')
        return CalibratedClassifierCV(base_clf, cv=5)
    elif CLASSIFIER_TYPE == 'rf':
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    elif CLASSIFIER_TYPE == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
    else:
        # Default to SVM
        base_clf = SVC(kernel='rbf', C=10, gamma='scale')
        return CalibratedClassifierCV(base_clf, cv=5)

def evaluate_biometric_system(features, identities, metadata):
    """
    Evaluate biometric authentication by creating templates from 10 training faces
    and testing against up to 100 faces with at least 1 genuine face
    """
    print("\nStarting biometric authentication evaluation...")
    
    # Storage for results
    genuine_scores = []
    imposter_scores = []
    confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    
    # Group samples by identity
    identity_samples = {}
    for i, identity in enumerate(identities):
        if identity not in identity_samples:
            identity_samples[identity] = []
        identity_samples[identity].append(i)
    
    # Filter identities with enough samples
    min_samples = TRAINING_FACES_PER_IDENTITY + MIN_TESTING_GENUINE_FACES
    valid_identities = [identity for identity, samples in identity_samples.items() 
                      if len(samples) >= min_samples]
    
    print(f"Testing {len(valid_identities)} identities with at least {min_samples} samples each")
    
    # Start testing
    for identity in tqdm(valid_identities, desc="Evaluating identities"):
        # Get all samples for this identity
        identity_indices = identity_samples[identity]
        
        # Split into training (template) and testing samples
        shuffled_indices = random.sample(identity_indices, len(identity_indices))
        training_indices = shuffled_indices[:TRAINING_FACES_PER_IDENTITY]
        remaining_indices = shuffled_indices[TRAINING_FACES_PER_IDENTITY:]
        
        # Get genuine testing samples
        genuine_count = min(len(remaining_indices), MAX_TESTING_FACES // 2)
        if genuine_count < MIN_TESTING_GENUINE_FACES:
            continue  # Skip if not enough genuine samples
            
        genuine_indices = remaining_indices[:genuine_count]
        
        # Prepare training data for this identity
        X_train = []
        y_train = []
        
        # Add template samples (labeled as 1 - genuine)
        for idx in training_indices:
            X_train.append(features[idx])
            y_train.append(1)  # Genuine
        
        # Add impostor samples (labeled as 0 - impostor)
        # Find all samples from other identities
        other_identity_indices = []
        for other_id in valid_identities:
            if other_id != identity:
                other_identity_indices.extend(identity_samples[other_id])
        
        # Sample impostors for training
        impostor_train_count = min(len(other_identity_indices), TRAINING_FACES_PER_IDENTITY * 3)
        impostor_train_indices = random.sample(other_identity_indices, impostor_train_count)
        
        for idx in impostor_train_indices:
            X_train.append(features[idx])
            y_train.append(0)  # Impostor
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Verify we have both classes in training data
        if len(set(y_train)) < 2:
            print(f"Warning: Identity {identity} has less than 2 classes in training data")
            continue
        
        # Create and train the classifier
        try:
            clf = get_classifier()
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error training classifier for identity {identity}: {e}")
            continue
        
        # Test against genuine samples
        for idx in genuine_indices:
            try:
                test_features = features[idx].reshape(1, -1)
                probability = clf.predict_proba(test_features)[0]
                
                # Get probability of genuine class
                if len(probability) >= 2:
                    genuine_prob = probability[1]
                else:
                    # For classifiers that return only one class probability
                    genuine_prob = probability[0] if y_train[0] == 1 else 1 - probability[0]
                
                genuine_scores.append(genuine_prob)
                
                # Update confusion matrix
                if genuine_prob >= MATCH_THRESHOLD:
                    confusion_matrix['tp'] += 1  # True positive
                else:
                    confusion_matrix['fn'] += 1  # False negative
                    
                # Print some debug info occasionally
                if len(genuine_scores) % 50 == 0:
                    print(f"Genuine test {len(genuine_scores)}: {metadata[idx]['identity']}, score: {genuine_prob:.4f}")
            except Exception as e:
                print(f"Error testing genuine sample: {e}")
        
        # Test against impostor samples
        # Select a set of impostor test samples
        impostor_test_count = min(MAX_TESTING_FACES - genuine_count, len(other_identity_indices))
        impostor_test_indices = random.sample(other_identity_indices, impostor_test_count)
        
        for idx in impostor_test_indices:
            try:
                test_features = features[idx].reshape(1, -1)
                probability = clf.predict_proba(test_features)[0]
                
                # Get probability of genuine class
                if len(probability) >= 2:
                    genuine_prob = probability[1]
                else:
                    # For classifiers that return only one class probability
                    genuine_prob = probability[0] if y_train[0] == 1 else 1 - probability[0]
                
                imposter_scores.append(genuine_prob)
                
                # Update confusion matrix
                if genuine_prob >= MATCH_THRESHOLD:
                    confusion_matrix['fp'] += 1  # False positive
                else:
                    confusion_matrix['tn'] += 1  # True negative
                    
                # Print some debug info occasionally
                if len(imposter_scores) % 100 == 0:
                    print(f"Impostor test {len(imposter_scores)}: {metadata[idx]['identity']}, score: {genuine_prob:.4f}")
            except Exception as e:
                print(f"Error testing impostor sample: {e}")
    
    # Check if we have enough data
    if len(genuine_scores) == 0 or len(imposter_scores) == 0:
        print("Warning: Not enough data to calculate metrics")
        if len(genuine_scores) == 0:
            print("No genuine scores collected")
        if len(imposter_scores) == 0:
            print("No impostor scores collected")
    
    # Return results
    return {
        'genuine_scores': np.array(genuine_scores),
        'imposter_scores': np.array(imposter_scores),
        'confusion_matrix': confusion_matrix,
        'classifier': clf
    }

'''
    Performance Metrics and Visualization
'''
def calculate_metrics(results):
    """Calculate performance metrics and generate visualizations"""
    cm = results['confusion_matrix']
    tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
    genuine_scores = results['genuine_scores']
    imposter_scores = results['imposter_scores']
    
    # Check if we have enough data
    if len(genuine_scores) == 0 or len(imposter_scores) == 0:
        print("Not enough data to calculate metrics.")
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'far': 0,
            'frr': 0,
            'confusion_matrix': (tp, tn, fp, fn)
        }
    
    # Calculate metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Print results
    print("\nBiometric Authentication Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    
    # Calculate Equal Error Rate (EER)
    thresholds = np.linspace(0, 1, 100)
    fars = []
    frrs = []
    
    for threshold in thresholds:
        false_accepts = sum(1 for score in imposter_scores if score >= threshold)
        false_rejects = sum(1 for score in genuine_scores if score < threshold)
        
        far_t = false_accepts / len(imposter_scores) if len(imposter_scores) > 0 else 0
        frr_t = false_rejects / len(genuine_scores) if len(genuine_scores) > 0 else 0
        
        fars.append(far_t)
        frrs.append(frr_t)
    
    # Find closest point to equal error rate
    eer_idx = np.argmin(np.abs(np.array(fars) - np.array(frrs)))
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    print(f"Equal Error Rate (EER): {eer:.4f} at threshold {eer_threshold:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    
    # Create labels for ROC curve
    y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy',lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1-FRR)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Mark the EER point
    plt.plot(fars[eer_idx], 1-frrs[eer_idx], 'ro', markersize=8, label=f'EER = {eer:.4f}')
    plt.legend(loc="lower right")
    
    plt.show()
    
    # Plot score distributions
    plt.figure(figsize=(10, 8))
    plt.hist(genuine_scores, bins=20, alpha=0.5, label='Genuine', color='green')
    plt.hist(imposter_scores, bins=20, alpha=0.5, label='Impostor', color='red')
    plt.xlabel('Match Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Match Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark the decision threshold and EER threshold
    plt.axvline(x=MATCH_THRESHOLD, color='blue', linestyle='-', 
                label=f'Decision Threshold ({MATCH_THRESHOLD:.2f})')
    plt.axvline(x=eer_threshold, color='red', linestyle='--', 
                label=f'EER Threshold ({eer_threshold:.2f})')
    plt.legend()
    plt.show()
    
    # Return metrics including EER
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'frr': frr,
        'eer': eer,
        'eer_threshold': eer_threshold
    }

'''
    Main Execution
'''
def main():
    # Start timing
    start_time = time.time()
    
    # Load IMDB dataset
    X, y, identity_to_idx, idx_to_identity, metadata = load_imdb_dataset()
    
    if X is None or len(X) == 0:
        print("Failed to load dataset or no valid images found.")
        return
    
    num_identities = len(set(y))
    print(f"Loaded {len(X)} face images for {num_identities} identities")
    
    # Print some identity mapping examples
    print("\nSample identities:")
    sample_identities = list(identity_to_idx.items())[:min(5, len(identity_to_idx))]
    for name, idx in sample_identities:
        print(f"  {name} -> {idx}")

    # Set up scaled features for evaluation
    print("Preparing features for evaluation...")
    scaler = MinMaxScaler()
    features = scaler.fit_transform(X)
    print("Feature scaling complete.")

    # Evaluate biometric system
    results = evaluate_biometric_system(features, y, metadata)
    
    # Calculate metrics and create visualizations
    metrics = calculate_metrics(results)
    
    # Update the optimal threshold based on EER
    global MATCH_THRESHOLD
    if 'eer_threshold' in metrics:
        optimal_threshold = metrics['eer_threshold']
        print(f"Optimal threshold based on EER: {optimal_threshold:.4f}")
        
        # Update threshold
        MATCH_THRESHOLD = optimal_threshold
        
        # Recalculate metrics with new threshold
        if len(results['genuine_scores']) > 0 and len(results['imposter_scores']) > 0:
            genuine_scores = results['genuine_scores']
            imposter_scores = results['imposter_scores']
            
            tp = sum(1 for score in genuine_scores if score >= MATCH_THRESHOLD)
            fn = sum(1 for score in genuine_scores if score < MATCH_THRESHOLD)
            fp = sum(1 for score in imposter_scores if score >= MATCH_THRESHOLD)
            tn = sum(1 for score in imposter_scores if score < MATCH_THRESHOLD)
            
            # Calculate updated metrics
            total = tp + tn + fp + fn
            updated_accuracy = (tp + tn) / total if total > 0 else 0
            updated_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            updated_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            updated_f1 = 2 * updated_precision * updated_recall / (updated_precision + updated_recall) if (updated_precision + updated_recall) > 0 else 0
            updated_far = fp / (fp + tn) if (fp + tn) > 0 else 0
            updated_frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print("\nUpdated Performance Metrics with Optimal Threshold:")
            print(f"Accuracy: {updated_accuracy:.4f}")
            print(f"Precision: {updated_precision:.4f}")
            print(f"Recall: {updated_recall:.4f}")
            print(f"F1 Score: {updated_f1:.4f}")
            print(f"True Positives: {tp}")
            print(f"True Negatives: {tn}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"False Acceptance Rate (FAR): {updated_far:.4f}")
            print(f"False Rejection Rate (FRR): {updated_frr:.4f}")
    
    # Save the model
    try:
        with open('imdb_biometric_auth_model.pkl', 'wb') as f:
            pickle.dump({
                'classifier': results['classifier'],
                'scaler': scaler,
                'threshold': MATCH_THRESHOLD,
                'identity_to_idx': identity_to_idx,
                'idx_to_identity': idx_to_identity
            }, f)
        print("Model saved to 'imdb_biometric_auth_model.pkl'")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Calculate and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    # Print system summary
    print("\n=== Biometric System Performance Summary ===")
    print(f"System Type: Face Recognition Biometric Authentication")
    print(f"Dataset: IMDB Dataset using celebrity identities")
    print(f"Classifier: {CLASSIFIER_TYPE.upper()}")
    print(f"Number of identities: {num_identities}")
    print(f"Number of images: {len(X)}")
    print(f"Training protocol: {TRAINING_FACES_PER_IDENTITY} faces per identity for template")
    print(f"Testing protocol: Up to {MAX_TESTING_FACES} test faces with at least {MIN_TESTING_GENUINE_FACES} genuine")
    print(f"Decision threshold: {MATCH_THRESHOLD:.4f}")
    print(f"False Acceptance Rate (FAR): {metrics.get('far', 0.0):.4f}")
    print(f"False Rejection Rate (FRR): {metrics.get('frr', 0.0):.4f}")
    if 'eer' in metrics:
        print(f"Equal Error Rate (EER): {metrics['eer']:.4f}")

# Run the program
if __name__ == "__main__":
    main()
