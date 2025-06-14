#!/usr/bin/env python3
"""
miDiscovery Sparse Autoencoder Training Demo for Neuronpedia Compatibility

This script demonstrates creating and training a sparse autoencoder on Phi-2 activations
and exporting the results in a format compatible with Neuronpedia.

Now with optional PostgreSQL database logging for experiment tracking.

Dependencies:
pip install torch transformers datasets accelerate wandb huggingface_hub
pip install sqlalchemy psycopg2-binary python-dotenv  # For database support
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import argparse
from datetime import datetime

# Import database utilities if available
try:
    from core.db_utils import (
        get_db, create_training_run, log_discovered_features,
        update_training_run_completion, TrainingRun
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    # Create a dummy TrainingRun class for type hints when DB is not available
    class TrainingRun:
        pass
    print("Database utilities not available. Run without --use-db flag.")


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder training"""
    # Model settings
    model_name: str = "microsoft/phi-2"  # Open model, no auth required
    layer_name: str = "model.layers.16.mlp.fc2"  # Target layer (Phi-2 MLP uses fc1/fc2)
    hook_point: str = "model.layers.16.mlp.fc2"
    
    # SAE architecture
    d_model: int = 2560  # Phi-2 hidden dimension
    expansion_factor: int = 8  # SAE hidden dimension = d_model * expansion_factor
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    l1_coefficient: float = 5e-4
    num_epochs: int = 5
    max_samples: int = 10000  # Limit for demo purposes
    
    # Data settings
    dataset_name: str = "openwebtext"
    seq_length: int = 512
    
    # Output settings
    output_dir: str = "./sae_outputs"
    save_top_k_features: int = 100
    neuronpedia_format: bool = True
    
    # Database settings
    use_db: bool = False


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder implementation compatible with Neuronpedia format
    """
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # Encoder: maps from model activations to sparse representation
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        
        # Decoder: maps from sparse representation back to model activations
        self.decoder = nn.Linear(d_hidden, d_model, bias=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following best practices for SAEs"""
        # Initialize encoder weights with Xavier uniform
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder as transpose of encoder (tied weights concept)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            nn.init.zeros_(self.decoder.bias)
            
        # Normalize decoder weights (important for SAE stability)
        self._normalize_decoder_weights()
    
    def _normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm (important for SAE training)"""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation"""
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to original space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both reconstruction and sparse codes"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class ActivationCapture:
    """Utility class to capture activations from Phi-2"""
    
    def __init__(self, model, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.activations = []
        self.hook_handle = None
        
    def hook_fn(self, module, input, output):
        """Hook function to capture activations"""
        # Store activations on CPU to save GPU memory
        if isinstance(output, tuple):
            activation = output[0].detach().cpu()
        else:
            activation = output.detach().cpu()
        self.activations.append(activation)
    
    def register_hook(self):
        """Register the hook on the specified layer"""
        target_module = self.model
        for name in self.layer_name.split('.'):
            target_module = getattr(target_module, name)
        
        self.hook_handle = target_module.register_forward_hook(self.hook_fn)
        print(f"Registered hook on {self.layer_name}")
    
    def remove_hook(self):
        """Remove the hook"""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def get_activations(self) -> torch.Tensor:
        """Get all captured activations as a single tensor"""
        if not self.activations:
            return torch.empty(0)
        
        # Concatenate all activations
        activations = torch.cat(self.activations, dim=0)
        self.activations = []  # Clear memory
        return activations


def load_model(model_name: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Phi-2 model and tokenizer (no authentication required)"""
    print(f"Loading {model_name}...")
    
    # Clear CUDA cache first
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check GPU memory and choose the best setup
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        gpu_memory.append(mem)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {mem:.1f}GB")
    
    # Prioritize the larger GPU (3090 should be GPU 0 with 24GB)
    if len(gpu_memory) >= 2:
        # Use the GPU with more memory as primary
        primary_gpu = 0 if gpu_memory[0] > gpu_memory[1] else 1
        print(f"Using GPU {primary_gpu} as primary ({gpu_memory[primary_gpu]:.1f}GB)")
        
        # Custom device map to ensure model goes to the right GPU
        device_map = {
            "model.embed_tokens": primary_gpu,
            "model.final_layernorm": primary_gpu,
            "lm_head": primary_gpu,
        }
        
        # Distribute layers across GPUs, prioritizing the larger one
        total_layers = 32  # Phi-2 has 32 layers
        if gpu_memory[primary_gpu] >= 20:  # If primary GPU has 20GB+, put most layers there
            primary_layers = int(total_layers * 0.8)  # 80% on primary
            secondary_layers = total_layers - primary_layers
            secondary_gpu = 1 - primary_gpu
            
            for i in range(primary_layers):
                device_map[f"model.layers.{i}"] = primary_gpu
            for i in range(primary_layers, total_layers):
                device_map[f"model.layers.{i}"] = secondary_gpu
        else:
            # More balanced distribution
            for i in range(total_layers):
                device_map[f"model.layers.{i}"] = i % len(gpu_memory)
        
        max_memory = {
            0: f"{int(gpu_memory[0] * 0.9)}GB",  # Use 90% of available memory
            1: f"{int(gpu_memory[1] * 0.9)}GB",
            "cpu": "30GB"
        }
    else:
        # Single GPU setup
        device_map = "auto"
        max_memory = {0: f"{int(gpu_memory[0] * 0.9)}GB", "cpu": "30GB"}
    
    print(f"Memory allocation: {max_memory}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map=device_map,
        trust_remote_code=True,
        max_memory=max_memory,
        offload_folder="./offload_cache",  # Enable disk offloading if needed
        low_cpu_mem_usage=True,
    )
    
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    return model, tokenizer


def prepare_dataset(config: SAEConfig, tokenizer: AutoTokenizer) -> List[str]:
    """Prepare dataset for training"""
    print(f"Loading dataset: {config.dataset_name}")
    
    # Load dataset
    if config.dataset_name == "openwebtext":
        dataset = load_dataset("openwebtext", split="train", streaming=True)
    else:
        dataset = load_dataset(config.dataset_name, split="train", streaming=True)
    
    # Extract text samples
    texts = []
    for i, example in enumerate(dataset):
        if i >= config.max_samples:
            break
        
        text = example.get('text', '')
        if len(text) > 100:  # Filter out very short texts
            texts.append(text)
    
    print(f"Prepared {len(texts)} text samples")
    return texts


def collect_activations(model, tokenizer, texts: List[str], config: SAEConfig, device: str) -> torch.Tensor:
    """Collect activations from the target layer"""
    print(f"Collecting activations from layer: {config.layer_name}")
    
    # Set up activation capture
    capture = ActivationCapture(model, config.layer_name)
    capture.register_hook()
    
    all_activations = []
    
    # Reduce batch size for memory efficiency
    effective_batch_size = min(config.batch_size, 8)  # Smaller batches for activation collection
    print(f"Using batch size {effective_batch_size} for activation collection")
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), effective_batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i + effective_batch_size]
                
                # Use shorter sequences to save memory
                effective_seq_length = min(config.seq_length, 256)
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=effective_seq_length
                )
                
                # Move inputs to the same device as the first model parameter
                first_param_device = next(model.parameters()).device
                inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
                
                # Forward pass to collect activations
                _ = model(**inputs)
                
                # Get activations from this batch
                batch_activations = capture.get_activations()
                if batch_activations.numel() > 0:
                    # Reshape to (batch_size * seq_len, d_model)
                    batch_activations = batch_activations.view(-1, config.d_model)
                    all_activations.append(batch_activations)
                
                # Clear GPU cache periodically
                if i % (effective_batch_size * 10) == 0:
                    torch.cuda.empty_cache()
    
    finally:
        capture.remove_hook()
    
    # Concatenate all activations
    if all_activations:
        activations = torch.cat(all_activations, dim=0)
        print(f"Collected {activations.shape[0]} activation vectors of dimension {activations.shape[1]}")
        return activations
    else:
        raise ValueError("No activations were collected!")


def train_sparse_autoencoder(activations: torch.Tensor, config: SAEConfig, device: str) -> Tuple[SparseAutoencoder, float]:
    """Train the sparse autoencoder and return both the model and final loss"""
    print("Training Sparse Autoencoder...")
    
    # Initialize SAE
    d_hidden = config.d_model * config.expansion_factor
    sae = SparseAutoencoder(config.d_model, d_hidden)
    
    # Move SAE to the best available GPU (largest memory)
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        gpu_memory.append(mem)
    
    best_gpu = 0 if len(gpu_memory) == 0 or gpu_memory[0] > (gpu_memory[1] if len(gpu_memory) > 1 else 0) else 1
    sae_device = f"cuda:{best_gpu}" if torch.cuda.is_available() else "cpu"
    sae = sae.to(sae_device)
    
    print(f"Training SAE on {sae_device}" + (f" ({gpu_memory[best_gpu]:.1f}GB)" if gpu_memory else ""))
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    
    # Move activations to SAE device and ensure consistent dtype
    activations = activations.to(sae_device).to(sae.encoder.weight.dtype)
    
    # Training loop
    sae.train()
    final_loss = 0.0
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_reconstruction_loss = 0.0
        num_batches = 0
        
        # Create batches
        num_samples = activations.shape[0]
        indices = torch.randperm(num_samples)
        
        # Use smaller batch size for training to fit in memory
        training_batch_size = min(config.batch_size, 16)
        
        for i in tqdm(range(0, num_samples, training_batch_size), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            batch_indices = indices[i:i + training_batch_size]
            batch_activations = activations[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed, sparse_codes = sae(batch_activations)
            
            # Compute losses
            reconstruction_loss = F.mse_loss(reconstructed, batch_activations)
            l1_loss = torch.mean(torch.abs(sparse_codes))
            total_loss = reconstruction_loss + config.l1_coefficient * l1_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Normalize decoder weights after each step
            sae._normalize_decoder_weights()
            
            # Track losses
            epoch_loss += total_loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            num_batches += 1
        
        # Print epoch statistics
        avg_loss = epoch_loss / num_batches
        avg_l1 = epoch_l1_loss / num_batches
        avg_recon = epoch_reconstruction_loss / num_batches
        final_loss = avg_loss  # Store the final epoch's loss
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, Reconstruction={avg_recon:.6f}, L1={avg_l1:.6f}")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "total_loss": avg_loss,
                "reconstruction_loss": avg_recon,
                "l1_loss": avg_l1
            })
    
    sae.eval()
    return sae, final_loss


def analyze_features(sae: SparseAutoencoder, activations: torch.Tensor, config: SAEConfig,
                    training_run: Optional[TrainingRun] = None) -> Dict:
    """Analyze learned features and prepare data for Neuronpedia"""
    print("Analyzing learned features...")
    
    device = next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    
    # Move activations to same device and dtype as SAE
    activations = activations.to(device).to(sae_dtype)
    
    print(f"SAE device: {device}, dtype: {sae_dtype}")
    print(f"Activations device: {activations.device}, dtype: {activations.dtype}")
    
    feature_analysis = {
        "model_name": config.model_name,
        "layer_name": config.layer_name,
        "d_model": config.d_model,
        "d_hidden": sae.d_hidden,
        "features": []
    }
    
    with torch.no_grad():
        # Process activations in chunks to save memory
        chunk_size = min(10000, activations.shape[0])
        all_sparse_codes = []
        
        print(f"Processing {activations.shape[0]} activations in chunks of {chunk_size}")
        
        for i in tqdm(range(0, activations.shape[0], chunk_size), desc="Computing sparse codes"):
            chunk = activations[i:i + chunk_size]
            chunk_codes = sae.encode(chunk)
            all_sparse_codes.append(chunk_codes.cpu())  # Move to CPU to save GPU memory
            
        # Concatenate all sparse codes
        sparse_codes = torch.cat(all_sparse_codes, dim=0)
        
        # Analyze each feature
        for feature_idx in tqdm(range(sae.d_hidden), desc="Analyzing features"):
            feature_activations = sparse_codes[:, feature_idx]
            
            # Calculate statistics
            max_activation = float(feature_activations.max())
            mean_activation = float(feature_activations.mean())
            activation_frequency = float((feature_activations > 0).float().mean())
            
            # Get top activating examples
            top_indices = torch.topk(feature_activations, k=min(20, len(feature_activations)))[1]
            top_activations = [float(feature_activations[idx]) for idx in top_indices]
            
            # Get decoder direction (what this feature represents)
            decoder_weights = sae.decoder.weight[:, feature_idx].cpu().numpy()
            
            feature_info = {
                "feature_idx": int(feature_idx),
                "max_activation": max_activation,
                "mean_activation": mean_activation,
                "activation_frequency": activation_frequency,
                "top_activations": top_activations,
                "decoder_norm": float(np.linalg.norm(decoder_weights)),
                "decoder_weights": decoder_weights.tolist()  # For Neuronpedia
            }
            
            feature_analysis["features"].append(feature_info)
    
    # Sort features by activation frequency for easier analysis
    feature_analysis["features"].sort(key=lambda x: x["activation_frequency"], reverse=True)
    
    return feature_analysis


def save_neuronpedia_format(sae: SparseAutoencoder, feature_analysis: Dict, config: SAEConfig,
                          training_run: Optional[TrainingRun] = None):
    """Save SAE in Neuronpedia-compatible format"""
    print("Saving in Neuronpedia format...")
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save the PyTorch model
    model_path = output_dir / "sae_model.pt"
    model_data = {
        'model_state_dict': sae.state_dict(),
        'config': {
            'd_model': config.d_model,
            'd_hidden': sae.d_hidden,
            'model_name': config.model_name,
            'layer_name': config.layer_name
        }
    }
    
    # Add training run ID if using database
    if training_run:
        model_data['config']['training_run_id'] = str(training_run.id)
    
    torch.save(model_data, model_path)
    
    # Save feature analysis
    analysis_path = output_dir / "feature_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(feature_analysis, f, indent=2)
    
    # Create Neuronpedia metadata file
    metadata = {
        "model_name": config.model_name,
        "layer": config.layer_name,
        "d_model": config.d_model,
        "d_sae": sae.d_hidden,
        "l1_coefficient": config.l1_coefficient,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "created_at": datetime.now().isoformat(),
        "num_features": len(feature_analysis["features"]),
        "active_features": sum(1 for f in feature_analysis["features"] if f["activation_frequency"] > 0.001)
    }
    
    # Add training run ID if using database
    if training_run:
        metadata["training_run_id"] = str(training_run.id)
    
    metadata_path = output_dir / "neuronpedia_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save top features for easier inspection
    top_features = feature_analysis["features"][:config.save_top_k_features]
    top_features_path = output_dir / f"top_{config.save_top_k_features}_features.json"
    with open(top_features_path, 'w') as f:
        json.dump(top_features, f, indent=2)
    
    # Save decoder weights as numpy array (useful for Neuronpedia)
    decoder_weights = sae.decoder.weight.detach().cpu().numpy()
    weights_path = output_dir / "decoder_weights.npy"
    np.save(weights_path, decoder_weights)
    
    print(f"Saved SAE artifacts to {output_dir}")
    print(f"- Model: {model_path}")
    print(f"- Feature analysis: {analysis_path}")
    print(f"- Metadata: {metadata_path}")
    print(f"- Top features: {top_features_path}")
    print(f"- Decoder weights: {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="miDiscovery: Train Sparse Autoencoder on Language Models")
    parser.add_argument("--model-name", default="microsoft/phi-2", help="Model name")
    parser.add_argument("--layer-name", default="model.layers.16.mlp.fc2", help="Target layer")
    parser.add_argument("--output-dir", default="./sae_outputs", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max training samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--l1-coef", type=float, default=5e-4, help="L1 regularization coefficient")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--use-db", action="store_true", help="Log training to PostgreSQL database")
    parser.add_argument("--inspect", action="store_true", help="Inspect model architecture and exit")
    
    args = parser.parse_args()
    
    # Check database availability if requested
    if args.use_db and not DB_AVAILABLE:
        print("❌ Database logging requested but db_utils not available!")
        print("Make sure PostgreSQL is running and src/core/db_utils.py is present.")
        return
    
    # Set environment variables for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project="miDiscovery-sae-demo", config=vars(args))
    
    # Create config
    config = SAEConfig(
        model_name=args.model_name,
        layer_name=args.layer_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        l1_coefficient=args.l1_coef,
        use_db=args.use_db
    )
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create training run in database if enabled
    training_run = None
    if config.use_db and DB_AVAILABLE:
        try:
            db = get_db()
            training_run = create_training_run(config, db)
            print(f"📝 Created training run: {training_run.id}")
            db.close()
        except Exception as e:
            print(f"⚠️ Warning: Could not create training run: {e}")
            training_run = None
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(config.model_name, device)
        
        # Prepare dataset
        texts = prepare_dataset(config, tokenizer)
        
        # Collect activations
        activations = collect_activations(model, tokenizer, texts, config, device)
        
        # Clear model from memory after activation collection
        del model
        torch.cuda.empty_cache()
        print("Cleared model from memory after activation collection")
        
        # Train SAE
        sae, final_loss = train_sparse_autoencoder(activations, config, device)
        
        # Analyze features
        feature_analysis = analyze_features(sae, activations, config, training_run)
        
        # Calculate metrics
        active_threshold = 0.001  # 0.1% activation frequency
        active_features = sum(1 for f in feature_analysis['features'] if f['activation_frequency'] > active_threshold)
        total_features = len(feature_analysis['features'])
        sparsity_percent = (active_features / total_features) * 100
        
        print(f"📊 Feature Statistics:")
        print(f"   Total features: {total_features}")
        print(f"   Active features (>{active_threshold*100}% freq): {active_features}")
        print(f"   Sparsity: {sparsity_percent:.1f}% (active features)")
        print(f"   Actually sparse: {100 - sparsity_percent:.1f}% (inactive features)")
        
        # Save results
        save_neuronpedia_format(sae, feature_analysis, config, training_run)
        
        print("\n✅ SAE training completed successfully!")
        print(f"📁 Results saved to: {config.output_dir}")
        print(f"🔍 Found {len(feature_analysis['features'])} features")
        print(f"⚡ Active features (>0.1% activation): {active_features}")
        
        if training_run:
            print(f"🔗 Training run ID: {training_run.id}")
            print(f"📊 View in database for full metrics and history")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise
    
    finally:
        if args.wandb:
            wandb.finish()
        
        # Final cleanup
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()