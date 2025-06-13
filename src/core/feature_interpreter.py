#!/usr/bin/env python3
"""
Feature Interpretation Script for miDiscovery SAE

This script helps interpret what natural language concepts your discovered features represent
by analyzing their activation patterns on real text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder implementation - copied for standalone use"""
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        
        # Encoder: maps from model activations to sparse representation
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        
        # Decoder: maps from sparse representation back to model activations
        self.decoder = nn.Linear(d_hidden, d_model, bias=True)
    
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


class FeatureInterpreter:
    """Interpret SAE features by analyzing their activations on text"""
    
    def __init__(self, sae_dir: str, model_name: str = "microsoft/phi-2"):
        self.sae_dir = Path(sae_dir)
        self.model_name = model_name
        
        # Load SAE
        self.sae = self.load_sae()
        self.feature_analysis = self.load_feature_analysis()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.load_model()
        
    def load_sae(self):
        """Load the trained SAE"""
        model_path = self.sae_dir / "sae_model.pt"
        checkpoint = torch.load(model_path, map_location='cpu')
        
        config = checkpoint['config']
        
        # Reconstruct SAE architecture
        sae = SparseAutoencoder(config['d_model'], config['d_hidden'])
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.eval()
        
        return sae
    
    def load_feature_analysis(self):
        """Load feature analysis data"""
        with open(self.sae_dir / "feature_analysis.json", 'r') as f:
            return json.load(f)
    
    def load_model(self):
        """Load the original model for activation extraction"""
        print(f"Loading {self.model_name} for interpretation...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        return model, tokenizer
    
    def get_text_activations(self, texts: List[str], layer_name: str) -> torch.Tensor:
        """Get activations for a list of texts"""
        all_activations = []
        
        class ActivationHook:
            def __init__(self):
                self.activations = []
            
            def __call__(self, module, input, output):
                if isinstance(output, tuple):
                    self.activations.append(output[0].detach().cpu())
                else:
                    self.activations.append(output.detach().cpu())
        
        # Register hook
        hook = ActivationHook()
        target_module = self.model
        for name in layer_name.split('.'):
            target_module = getattr(target_module, name)
        
        handle = target_module.register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                for text in tqdm(texts, desc="Processing texts"):
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
                    
                    hook.activations = []  # Clear previous activations
                    _ = self.model(**inputs)
                    
                    if hook.activations:
                        # Get activations and reshape
                        activations = hook.activations[0]  # Take first (and only) activation
                        # Reshape to (seq_len, d_model)
                        if len(activations.shape) > 2:
                            activations = activations.view(-1, activations.shape[-1])
                        all_activations.append(activations)
        
        finally:
            handle.remove()
        
        if all_activations:
            return torch.cat(all_activations, dim=0)
        else:
            return torch.empty(0)
    
    def find_max_activating_texts(self, feature_idx: int, test_texts: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Find texts that maximally activate a specific feature"""
        print(f"  Analyzing feature {feature_idx} on {len(test_texts)} texts...")
        
        # Get layer name from metadata
        layer_name = self.feature_analysis['layer_name']
        
        # Get activations for all test texts
        activations = self.get_text_activations(test_texts, layer_name)
        
        if activations.numel() == 0:
            return []
        
        # Convert to SAE device and dtype
        sae_device = next(self.sae.parameters()).device
        sae_dtype = next(self.sae.parameters()).dtype
        activations = activations.to(sae_device).to(sae_dtype)
        
        # Get sparse codes for this feature
        with torch.no_grad():
            sparse_codes = self.sae.encode(activations)
            feature_activations = sparse_codes[:, feature_idx].cpu().numpy()
        
        # Find top activating examples
        top_indices = np.argsort(feature_activations)[-top_k:][::-1]  # Descending order
        
        results = []
        activation_idx = 0
        
        for text in test_texts:
            # Count tokens in this text to know how many activations it contributed
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            num_tokens = inputs['input_ids'].shape[1]
            
            # Check if any of the top indices fall in this text's range
            text_indices = list(range(activation_idx, activation_idx + num_tokens))
            
            for top_idx in top_indices:
                if top_idx in text_indices:
                    activation_value = feature_activations[top_idx]
                    if activation_value > 0:  # Only include active features
                        results.append((text, float(activation_value)))
            
            activation_idx += num_tokens
        
        # Sort by activation value and remove duplicates
        results = list(set(results))  # Remove duplicates
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def interpret_top_features(self, test_texts: List[str], top_k_features: int = 10, top_k_texts: int = 10):
        """Interpret the top K features by finding their max activating texts"""
        
        # Get top features by activation frequency
        top_features = sorted(
            self.feature_analysis['features'], 
            key=lambda x: x['activation_frequency'], 
            reverse=True
        )[:top_k_features]
        
        print(f"\n🔍 Interpreting top {top_k_features} features...")
        print("=" * 80)
        
        interpretations = {}
        
        for i, feature_info in enumerate(top_features):
            feature_idx = feature_info['feature_idx']
            activation_freq = feature_info['activation_frequency']
            max_activation = feature_info['max_activation']
            
            print(f"\n📋 Feature {feature_idx} (Rank #{i+1})")
            print(f"   Activation frequency: {activation_freq:.4f}")
            print(f"   Max activation: {max_activation:.4f}")
            
            # Find texts that maximally activate this feature
            max_texts = self.find_max_activating_texts(feature_idx, test_texts, top_k_texts)
            
            if max_texts:
                print(f"   📝 Top activating text examples:")
                for j, (text, activation) in enumerate(max_texts[:5]):  # Show top 5
                    # Truncate long texts
                    display_text = text[:100] + "..." if len(text) > 100 else text
                    print(f"      {j+1}. [{activation:.3f}] {repr(display_text)}")
                
                interpretations[feature_idx] = {
                    'rank': i + 1,
                    'activation_frequency': activation_freq,
                    'max_activation': max_activation,
                    'top_texts': max_texts
                }
            else:
                print(f"   ❌ No strongly activating texts found")
                interpretations[feature_idx] = {
                    'rank': i + 1,
                    'activation_frequency': activation_freq,
                    'max_activation': max_activation,
                    'top_texts': []
                }
        
        return interpretations
    
    def save_interpretations(self, interpretations: Dict, output_file: str = "feature_interpretations.json"):
        """Save interpretations to file"""
        output_path = self.sae_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(interpretations, f, indent=2)
        print(f"\n💾 Saved interpretations to: {output_path}")


def get_sample_texts() -> List[str]:
    """Get a diverse set of sample texts for interpretation"""
    return [
        # Scientific/Technical
        "The neural network processed the input data through multiple layers of computation.",
        "DNA replication is a fundamental process in cellular biology.",
        "The algorithm converged after 100 iterations of gradient descent.",
        "Quantum mechanics describes the behavior of particles at atomic scales.",
        
        # Literature/Creative
        "The old man walked slowly down the moonlit street, his cane tapping against the cobblestones.",
        "She opened the book and began reading the first chapter with great enthusiasm.",
        "The sunset painted the sky in brilliant shades of orange and purple.",
        "Poetry has the power to express emotions that prose cannot capture.",
        
        # News/Current Events
        "The president announced new policies during yesterday's press conference.",
        "Stock markets showed significant volatility following the economic announcement.",
        "Climate change continues to be a major concern for environmental scientists.",
        "The new technology promises to revolutionize how we communicate.",
        
        # Conversational/Informal
        "Hey, did you see that movie last night? It was pretty good!",
        "I'm thinking about getting pizza for dinner, what do you think?",
        "Thanks for helping me with that project, I really appreciate it.",
        "LOL that's hilarious! I can't believe that actually happened.",
        
        # Mathematical/Logical
        "If A equals B and B equals C, then A must equal C by transitivity.",
        "The probability of rolling a six on a fair die is one-sixth.",
        "To solve this equation, we need to isolate the variable on one side.",
        "The derivative of x squared is two x.",
        
        # Historical/Factual
        "World War II ended in 1945 with the surrender of Japan.",
        "The Great Wall of China was built over many centuries to defend against invasions.",
        "Shakespeare wrote Romeo and Juliet in the late 16th century.",
        "The invention of the printing press revolutionized the spread of knowledge.",
        
        # Emotional/Social
        "I felt so happy when I heard the good news about my promotion.",
        "She was worried about her exam results and couldn't sleep.",
        "The community came together to help the family after the disaster.",
        "Love is one of the most powerful emotions humans experience.",
        
        # Punctuation and structure examples
        "Hello, world!",
        "What time is it?",
        "This is a test.",
        "End of sentence.",
        "Beginning of paragraph.",
        "1. First item in list",
        "2. Second item in list", 
        "The end.",
    ]


def main():
    parser = argparse.ArgumentParser(description="Interpret miDiscovery SAE features")
    parser.add_argument("sae_dir", help="Directory containing SAE outputs")
    parser.add_argument("--model-name", default="microsoft/phi-2", help="Original model name")
    parser.add_argument("--top-features", type=int, default=10, help="Number of top features to interpret")
    parser.add_argument("--top-texts", type=int, default=10, help="Number of top texts per feature")
    parser.add_argument("--custom-texts", help="File containing custom test texts (one per line)")
    
    args = parser.parse_args()
    
    # Load test texts
    if args.custom_texts and Path(args.custom_texts).exists():
        with open(args.custom_texts, 'r') as f:
            test_texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(test_texts)} custom test texts")
    else:
        test_texts = get_sample_texts()
        print(f"Using {len(test_texts)} built-in sample texts")
    
    try:
        # Create interpreter
        interpreter = FeatureInterpreter(args.sae_dir, args.model_name)
        
        # Interpret features
        interpretations = interpreter.interpret_top_features(
            test_texts, 
            args.top_features, 
            args.top_texts
        )
        
        # Save results
        interpreter.save_interpretations(interpretations)
        
        print(f"\n🎉 Feature interpretation complete!")
        print(f"📊 Analyzed {args.top_features} features on {len(test_texts)} texts")
        print(f"💡 Review the results to understand what concepts your features represent")
        
    except Exception as e:
        print(f"❌ Error during interpretation: {e}")
        print("This might be due to GPU memory limitations or model loading issues.")
        print("Try with fewer features: --top-features 3")


if __name__ == "__main__":
    main()