import torch
import joblib
import numpy as np
from config import settings

# Import the PyTorch blueprint we built earlier
from core.inference.fcn import fcn_wang

class FCNWangPipeline:
    def __init__(self):
        print("Initializing Wang ML Pipeline...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Scikit-Learn artifacts (Scaler and MultiLabelBinarizer)
        try:
            self.scaler = joblib.load(settings.SCALER_PATH)
            self.mlb = joblib.load(settings.MLB_PATH)
            self.classes = self.mlb.classes_
        except Exception as e:
            print(f"Error loading pkl artifacts: {e}")
            raise e
        
        # 2. Initialize the empty PyTorch blueprint (71 classes, 12 leads)
        self.model = fcn_wang(num_classes=71, input_channels=12, lin_ftrs_head=[128])
        
        # 3. Load the FastAI Checkpoint
        # We use weights_only=False because FastAI saves custom metadata/numpy types
        checkpoint = torch.load(
            settings.MODEL_WEIGHTS_PATH, 
            map_location=self.device, 
            weights_only=False
        )
        
        # 4. Unpack the FastAI Wrapper
        # FastAI saves weights under the 'model' key. We must extract it.
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 5. Clean FastAI prefixes
        # FastAI often adds 'model.' to every key; we strip it for the raw PyTorch blueprint
        clean_state_dict = {
            (k.replace('model.', '') if k.startswith('model.') else k): v 
            for k, v in state_dict.items()
        }
            
        # 6. Load weights into the blueprint
        self.model.load_state_dict(clean_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Pipeline Ready.")

    def predict(self, ecg_signal: np.ndarray) -> dict:
        """
        Input ecg_signal: (1000, 12)
        """
        # Step 1: Scaling (Fixed for exp0)
        # We flatten to (-1, 1), scale, then reshape back to (1000, 12)
        original_shape = ecg_signal.shape # (1000, 12)
        flattened_signal = ecg_signal.reshape(-1, 1) 
        scaled_flattened = self.scaler.transform(flattened_signal)
        scaled_signal = scaled_flattened.reshape(original_shape)
        
        # Step 2: Tensor Conversion (1000, 12) -> (1, 12, 1000)
        transposed_signal = scaled_signal.T
        batched_signal = np.expand_dims(transposed_signal, axis=0)
        
        tensor_input = torch.tensor(batched_signal, dtype=torch.float32).to(self.device)
        
        # Step 3: Forward Pass
        with torch.no_grad():
            logits = self.model(tensor_input)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
        # Step 4: Map Results
        results = {class_name: float(prob) for class_name, prob in zip(self.classes, probabilities)}
        detected = [name for name, prob in results.items() if prob > 0.5]

        return {
            "probabilities": results,
            "detected_conditions": detected
        }

# Instantiate as a singleton for the FastAPI app
try:
    ml_pipeline = FCNWangPipeline()
except Exception as e:
    print(f"WARNING: ML pipeline not loaded ({e}). Predict endpoint will be unavailable.")
    ml_pipeline = None