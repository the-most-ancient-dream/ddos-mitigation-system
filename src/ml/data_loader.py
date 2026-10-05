"""
CIC-DDoS-2019 Dataset Loader
Handles loading, preprocessing, and preparing the dataset for ML training
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .feature_extractor import FEATURE_NAMES
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# CIC-DDoS-2019 attack types (label mapping)
ATTACK_LABELS = {
    'BENIGN': 0,
    'DrDoS_DNS': 1,
    'DrDoS_LDAP': 2,
    'DrDoS_MSSQL': 3,
    'DrDoS_NTP': 4,
    'DrDoS_NetBIOS': 5,
    'DrDoS_SNMP': 6,
    'DrDoS_SSDP': 7,
    'DrDoS_UDP': 8,
    'Syn': 9,
    'TFTP': 10,
    'UDP-lag': 11,
    'WebDDoS': 12,
    'Portmap': 13,
    'NetBIOS': 14,
    'LDAP': 15,
    'MSSQL': 16,
    'UDP': 17,
}

# Train with the same ordered features that the live extractor emits.
SELECTED_FEATURES = FEATURE_NAMES.copy()


class CICDataLoader:
    """Load and preprocess CSV or Parquet CIC-DDoS-2019 files."""
    
    def __init__(self, data_dir: str = 'data/cic-ddos-2019'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing CIC-DDoS-2019 CSV or Parquet files
        """
        self.data_dir = Path(data_dir)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = SELECTED_FEATURES.copy()
        self._fitted = False
        
    def find_csv_files(self) -> List[Path]:
        """Find CSV files for compatibility with existing callers."""
        csv_files = sorted(self.data_dir.rglob('*.csv'))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        return csv_files

    def find_data_files(self) -> List[Path]:
        """Find supported CSV and Parquet files in the data directory."""
        files = sorted(
            path for path in self.data_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in {'.csv', '.parquet'}
        )
        logger.info(f"Found {len(files)} CSV/Parquet files in {self.data_dir}")
        return files
    
    def load_single_file(self, filepath: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load a single CSV file with proper handling
        
        Args:
            filepath: Path to CSV file
            sample_size: Optional sample size per file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            if filepath.suffix.lower() == '.parquet':
                df = pd.read_parquet(filepath)
            else:
                # CIC CSV releases vary in whitespace and column naming.
                df = pd.read_csv(filepath, low_memory=False, encoding='utf-8')
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Sample if needed
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            logger.info(f"Loaded {len(df)} samples from {filepath.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

    def load_single_csv(self, filepath: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Backward-compatible CSV loader."""
        return self.load_single_file(filepath, sample_size)

    def _load_files(
        self,
        files: List[Path],
        samples_per_file: Optional[int],
        balance_classes: bool = False,
    ) -> pd.DataFrame:
        """Read a known group of files and optionally balance its labels."""
        dfs = []
        for filepath in files:
            df = self.load_single_file(filepath, samples_per_file)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} total samples")
        label_col = next(
            (column for column in combined_df.columns if str(column).strip() == 'Label'),
            None,
        )
        if balance_classes and label_col is not None:
            combined_df = self._balance_classes(combined_df)
        return combined_df
    
    def load_dataset(self, 
                     max_files: Optional[int] = None,
                     samples_per_file: Optional[int] = 50000,
                     balance_classes: bool = True) -> pd.DataFrame:
        """
        Load the complete dataset from multiple CSV files
        
        Args:
            max_files: Maximum number of files to load
            samples_per_file: Maximum samples per file
            balance_classes: Whether to balance attack/benign classes
            
        Returns:
            Combined DataFrame
        """
        data_files = self.find_data_files()
        
        if not data_files:
            logger.warning(f"No CSV or Parquet files found in {self.data_dir}")
            return pd.DataFrame()
        
        if max_files:
            data_files = data_files[:max_files]
        
        return self._load_files(data_files, samples_per_file, balance_classes)
    
    def _balance_classes(self, df: pd.DataFrame, 
                         benign_ratio: float = 0.3) -> pd.DataFrame:
        """
        Balance benign and attack samples
        
        Args:
            df: Input DataFrame
            benign_ratio: Target ratio of benign samples
            
        Returns:
            Balanced DataFrame
        """
        label_col = next(
            (column for column in df.columns if str(column).strip() == 'Label'),
            None,
        )
        if label_col is None:
            return df

        benign_mask = df[label_col].astype(str).str.strip().str.upper() == 'BENIGN'
        attack_df = df[~benign_mask]
        benign_df = df[benign_mask]
        
        # Calculate target benign count
        target_benign = int(len(attack_df) * benign_ratio / (1 - benign_ratio))
        target_benign = min(target_benign, len(benign_df))
        
        if target_benign < len(benign_df):
            benign_df = benign_df.sample(n=target_benign, random_state=42)
        
        balanced_df = pd.concat([attack_df, benign_df], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Balanced dataset: {len(attack_df)} attacks, {len(benign_df)} benign")
        return balanced_df
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess the dataset for ML training
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (features array, labels array, feature names)
        """
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Get label column
        label_col = next(
            (column for column in df.columns if str(column).strip() == 'Label'),
            None,
        )
        if label_col is None:
            logger.error("Dataset must contain a 'Label' column")
            return np.array([]), np.array([]), []

        labels = df[label_col].astype(str).str.strip().values
        labels = np.asarray([
            'BENIGN' if label.upper() == 'BENIGN' else label
            for label in labels
        ])
        
        # Find available features (column names may have spaces)
        columns_by_name = {
            str(column).strip().casefold(): column for column in df.columns
        }
        # CICFlowMeter v3 Parquet files renamed several columns while retaining
        # the same measurements. Map those names to the live extractor schema.
        aliases = {
            'Total Length of Fwd Packets': ('Fwd Packets Length Total',),
            'Total Length of Bwd Packets': ('Bwd Packets Length Total',),
            'Min Packet Length': ('Packet Length Min',),
            'Max Packet Length': ('Packet Length Max',),
            'Average Packet Size': ('Avg Packet Size',),
            'Init_Win_bytes_forward': ('Init Fwd Win Bytes', 'Init_Win_bytes_forward'),
            'Init_Win_bytes_backward': ('Init Bwd Win Bytes', 'Init_Win_bytes_backward'),
            'act_data_pkt_fwd': ('Fwd Act Data Packets', 'act_data_pkt_fwd'),
            'min_seg_size_forward': ('Fwd Seg Size Min', 'min_seg_size_forward'),
        }
        selected_columns = []
        available_features = []
        for feature in SELECTED_FEATURES:
            source = columns_by_name.get(feature.casefold())
            if source is None:
                source = next(
                    (
                        columns_by_name[alias.casefold()]
                        for alias in aliases.get(feature, ())
                        if alias.casefold() in columns_by_name
                    ),
                    None,
                )
            if source is not None:
                selected_columns.append(source)
                available_features.append(feature)
        
        if not available_features:
            logger.error("No matching features found in dataset")
            logger.info(f"Dataset columns: {list(df.columns[:20])}")
            return np.array([]), np.array([]), []
        
        logger.info(f"Using {len(available_features)} features")
        
        # Extract features
        X = df[selected_columns].copy()
        X.columns = available_features
        self.feature_names = available_features.copy()
        
        # Handle missing values - replace with 0
        X = X.fillna(0)
        
        # Handle infinity values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Convert to numpy array
        X_array = X.values.astype(np.float32)
        
        # Clip extreme values
        X_array = np.clip(X_array, -1e10, 1e10)
        
        return X_array, labels, self.feature_names
    
    def encode_labels(self, labels: np.ndarray, binary: bool = True) -> np.ndarray:
        """
        Encode string labels to numeric
        
        Args:
            labels: String labels array
            binary: If True, encode as binary (0=benign, 1=attack)
            
        Returns:
            Encoded labels
        """
        if binary:
            # Binary classification: BENIGN = 0, Attack = 1
            encoded = np.array([
                0 if str(label).strip().upper() == 'BENIGN' else 1 
                for label in labels
            ])
        else:
            # Multi-class classification
            if not self._fitted:
                self.label_encoder.fit(labels)
                self._fitted = True
            encoded = self.label_encoder.transform(labels)
        
        return encoded
    
    def scale_features(self, X_train: np.ndarray, 
                       X_test: Optional[np.ndarray] = None,
                       fit: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Optional test features
            fit: Whether to fit the scaler (True for training)
            
        Returns:
            Scaled feature arrays
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, 
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     binary: bool = True,
                     scale: bool = True,
                     max_files: Optional[int] = None,
                     samples_per_file: int = 50000) -> Dict:
        """
        Complete data preparation pipeline
        
        Args:
            test_size: Fraction for test set
            val_size: Fraction for validation set
            binary: Binary classification mode
            scale: Whether to scale features
            max_files: Maximum CSV files to load
            samples_per_file: Samples per file
            
        Returns:
            Dictionary with train/val/test splits and metadata
        """
        data_files = self.find_data_files()
        if max_files:
            data_files = data_files[:max_files]

        # CIC-DDoS2019 Parquet mirrors often preserve the supplied train/test
        # split in filenames. Keep that official holdout separate for binary
        # training instead of mixing rows from the same captures.
        train_files = [
            path for path in data_files
            if path.stem.lower().endswith(('-training', '_training'))
        ]
        test_files = [
            path for path in data_files
            if path.stem.lower().endswith(('-testing', '_testing'))
        ]

        if binary and train_files and test_files:
            logger.info("Using dataset-provided training/testing file split")
            train_df = self._load_files(
                train_files, samples_per_file, balance_classes=True
            )
            test_df = self._load_files(
                test_files, samples_per_file, balance_classes=False
            )
            if train_df.empty or test_df.empty:
                logger.error("Could not load both training and testing files")
                return {}

            X_train_all, y_train_str, feature_names = self.preprocess(train_df)
            X_test, y_test_str, test_feature_names = self.preprocess(test_df)
            if len(X_train_all) == 0 or len(X_test) == 0:
                logger.error("Preprocessing failed")
                return {}
            if feature_names != test_feature_names:
                logger.error("Training and testing feature schemas differ")
                return {}

            y_train_all = self.encode_labels(y_train_str, binary=True)
            y_test = self.encode_labels(y_test_str, binary=True)
            validation_fraction = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_all,
                y_train_all,
                test_size=validation_fraction,
                random_state=42,
                stratify=y_train_all,
            )
            if scale:
                X_train, _ = self.scale_features(X_train, fit=True)
                X_val, _ = self.scale_features(X_val, fit=False)
                X_test, _ = self.scale_features(X_test, fit=False)

            logger.info(f"Dataset shape: train={len(X_train_all)}, test={len(X_test)}")
            logger.info(
                f"Label distribution: train={np.bincount(y_train_all)}, "
                f"test={np.bincount(y_test)}"
            )
            logger.info(
                f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            )
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'feature_names': feature_names,
                'scaler': self.scaler,
                'label_encoder': None,
                'binary_mode': True,
                'original_labels': y_train_str,
            }

        # Fall back to a stratified random split for datasets without supplied
        # train/test groups (and for multiclass label encoding).
        df = self._load_files(data_files, samples_per_file, balance_classes=True)
        
        if df.empty:
            logger.error("Failed to load dataset")
            return {}
        
        # Preprocess
        X, y_str, feature_names = self.preprocess(df)
        
        if len(X) == 0:
            logger.error("Preprocessing failed")
            return {}
        
        # Encode labels
        y = self.encode_labels(y_str, binary=binary)
        
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Label distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), 
            random_state=42, stratify=y
        )
        
        val_fraction = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_fraction),
            random_state=42, stratify=y_temp
        )
        
        # Scale features
        if scale:
            X_train, _ = self.scale_features(X_train, fit=True)
            X_val, _ = self.scale_features(X_val, fit=False)
            X_test, _ = self.scale_features(X_test, fit=False)
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder if not binary else None,
            'binary_mode': binary,
            'original_labels': y_str,
        }
    
    def get_attack_label_name(self, encoded_label: int, binary: bool = True) -> str:
        """
        Get human-readable label name from encoded label
        
        Args:
            encoded_label: Numeric label
            binary: Whether binary mode was used
            
        Returns:
            Label name string
        """
        if binary:
            return 'BENIGN' if encoded_label == 0 else 'ATTACK'
        else:
            return self.label_encoder.inverse_transform([encoded_label])[0]


def create_synthetic_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create synthetic dataset for testing when CIC dataset is unavailable
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Synthetic DataFrame mimicking CIC-DDoS-2019 structure
    """
    np.random.seed(42)
    
    # Generate benign traffic features
    n_benign = n_samples // 2
    n_attack = n_samples - n_benign
    
    data = []
    
    # Benign traffic characteristics
    for _ in range(n_benign):
        sample = {
            'Flow Duration': np.random.exponential(1000000),
            'Total Fwd Packets': np.random.poisson(10),
            'Total Backward Packets': np.random.poisson(8),
            'Total Length of Fwd Packets': np.random.exponential(1000),
            'Total Length of Bwd Packets': np.random.exponential(800),
            'Fwd Packet Length Mean': np.random.exponential(100),
            'Bwd Packet Length Mean': np.random.exponential(80),
            'Flow Bytes/s': np.random.exponential(10000),
            'Flow Packets/s': np.random.exponential(100),
            'SYN Flag Count': np.random.poisson(1),
            'ACK Flag Count': np.random.poisson(5),
            'Average Packet Size': np.random.exponential(200),
            ' Label': 'BENIGN'
        }
        data.append(sample)
    
    # Attack traffic characteristics (higher rates, different patterns)
    attack_types = ['DrDoS_UDP', 'Syn', 'DrDoS_DNS', 'DrDoS_LDAP']
    for _ in range(n_attack):
        attack_type = np.random.choice(attack_types)
        sample = {
            'Flow Duration': np.random.exponential(100000),  # Shorter
            'Total Fwd Packets': np.random.poisson(100),  # More packets
            'Total Backward Packets': np.random.poisson(2),  # Few responses
            'Total Length of Fwd Packets': np.random.exponential(10000),
            'Total Length of Bwd Packets': np.random.exponential(100),
            'Fwd Packet Length Mean': np.random.exponential(200),
            'Bwd Packet Length Mean': np.random.exponential(20),
            'Flow Bytes/s': np.random.exponential(100000),  # Higher
            'Flow Packets/s': np.random.exponential(1000),  # Higher
            'SYN Flag Count': np.random.poisson(50) if attack_type == 'Syn' else np.random.poisson(2),
            'ACK Flag Count': np.random.poisson(1),
            'Average Packet Size': np.random.exponential(100),
            ' Label': attack_type
        }
        data.append(sample)
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created synthetic dataset with {len(df)} samples")
    return df


if __name__ == '__main__':
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = CICDataLoader()
    
    # Try loading real dataset
    data_files = loader.find_data_files()
    
    if data_files:
        print("Loading real CIC-DDoS-2019 dataset...")
        data = loader.prepare_data(max_files=2, samples_per_file=10000)
        if data:
            print(f"Loaded {len(data['X_train'])} training samples")
            print(f"Features: {len(data['feature_names'])}")
    else:
        print("No dataset found. Creating synthetic dataset for testing...")
        df = create_synthetic_dataset(1000)
        print(df.head())
        print(f"\nLabel distribution:\n{df[' Label'].value_counts()}")
