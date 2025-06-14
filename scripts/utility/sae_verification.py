#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

"""
Verification script for SAE outputs - ensures compatibility with Neuronpedia
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse


def verify_sae_outputs(output_dir: str, verbose: bool = False) -> Dict[str, bool]:
    """Verify that all required SAE outputs are present and valid"""
    
    output_path = Path(output_dir)
    verification_results = {}
    
    print(f"🔍 Verifying SAE outputs in: {output_path}")
    print("=" * 60)
    
    # Check required files
    required_files = {
        "sae_model.pt": "PyTorch model checkpoint",
        "feature_analysis.json": "Feature analysis data",
        "neuronpedia_metadata.json": "Neuronpedia metadata",
        "decoder_weights.npy": "Decoder weights array",
        "top_100_features.json": "Top features analysis"
    }
    
    print("📁 Checking required files:")
    for filename, description in required_files.items():
        file_path = output_path / filename
        exists = file_path.exists()
        verification_results[f"file_{filename}"] = exists
        status = "✅" if exists else "❌"
        print(f"  {status} {filename} - {description}")
        
        if not exists:
            print(f"    🚨 Missing file: {filename}")
    
    print()
    
    # Verify model checkpoint
    print("🤖 Verifying model checkpoint:")
    model_path = output_path / "sae_model.pt"
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            required_keys = ['model_state_dict', 'config']
            
            for key in required_keys:
                has_key = key in checkpoint
                verification_results[f"checkpoint_{key}"] = has_key
                status = "✅" if has_key else "❌"
                print(f"  {status} Has '{key}' in checkpoint")
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                config_keys = ['d_model', 'd_hidden', 'model_name', 'layer_name']
                for key in config_keys:
                    has_key = key in config
                    verification_results[f"config_{key}"] = has_key
                    status = "✅" if has_key else "❌"
                    value = config.get(key, 'MISSING')
                    # Only show value if verbose or it's missing
                    if verbose or value == 'MISSING':
                        print(f"  {status} Config has '{key}': {value}")
                    else:
                        print(f"  {status} Config has '{key}'")
                    
        except Exception as e:
            verification_results["checkpoint_valid"] = False
            print(f"  ❌ Error loading checkpoint: {e}")
    else:
        verification_results["checkpoint_valid"] = False
        print("  ❌ Model checkpoint file missing")
    
    print()
    
    # Verify feature analysis
    print("🔬 Verifying feature analysis:")
    analysis_path = output_path / "feature_analysis.json"
    if analysis_path.exists():
        try:
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
            
            required_keys = ['model_name', 'layer_name', 'd_model', 'd_hidden', 'features']
            for key in required_keys:
                has_key = key in analysis
                verification_results[f"analysis_{key}"] = has_key
                status = "✅" if has_key else "❌"
                value = analysis.get(key, 'MISSING')
                # Only show value if verbose, it's missing, or it's a count
                if verbose or value == 'MISSING' or key in ['d_model', 'd_hidden']:
                    print(f"  {status} Has '{key}': {value}")
                else:
                    print(f"  {status} Has '{key}'")
            
            if 'features' in analysis:
                num_features = len(analysis['features'])
                verification_results["analysis_num_features"] = num_features > 0
                print(f"  ✅ Number of features: {num_features}")
                
                if num_features > 0:
                    sample_feature = analysis['features'][0]
                    feature_keys = ['feature_idx', 'max_activation', 'mean_activation', 
                                  'activation_frequency', 'decoder_weights']
                    for key in feature_keys:
                        has_key = key in sample_feature
                        verification_results[f"feature_{key}"] = has_key
                        status = "✅" if has_key else "❌"
                        
                        if key == 'decoder_weights' and has_key and not verbose:
                            # Don't print the actual weights unless verbose
                            weights_len = len(sample_feature[key]) if isinstance(sample_feature[key], list) else "unknown"
                            print(f"  {status} Feature has '{key}' ({weights_len} values)")
                        else:
                            print(f"  {status} Feature has '{key}'")
                        
        except Exception as e:
            verification_results["analysis_valid"] = False
            print(f"  ❌ Error loading feature analysis: {e}")
    else:
        verification_results["analysis_valid"] = False
        print("  ❌ Feature analysis file missing")
    
    print()
    
    # Verify metadata
    print("📋 Verifying Neuronpedia metadata:")
    metadata_path = output_path / "neuronpedia_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            required_keys = ['model_name', 'layer', 'd_model', 'd_sae', 'l1_coefficient', 
                           'learning_rate', 'num_epochs', 'created_at']
            for key in required_keys:
                has_key = key in metadata
                verification_results[f"metadata_{key}"] = has_key
                status = "✅" if has_key else "❌"
                value = metadata.get(key, 'MISSING')
                # Show values for key metadata
                if key in ['model_name', 'layer', 'd_model', 'd_sae', 'num_epochs'] or verbose:
                    print(f"  {status} Has '{key}': {value}")
                else:
                    print(f"  {status} Has '{key}'")
                
        except Exception as e:
            verification_results["metadata_valid"] = False
            print(f"  ❌ Error loading metadata: {e}")
    else:
        verification_results["metadata_valid"] = False
        print("  ❌ Metadata file missing")
    
    print()
    
    # Verify decoder weights
    print("🎯 Verifying decoder weights:")
    weights_path = output_path / "decoder_weights.npy"
    if weights_path.exists():
        try:
            weights = np.load(weights_path)
            verification_results["weights_shape_valid"] = len(weights.shape) == 2
            verification_results["weights_not_empty"] = weights.size > 0
            
            print(f"  ✅ Weights shape: {weights.shape}")
            print(f"  ✅ Weights dtype: {weights.dtype}")
            print(f"  ✅ Weights range: [{weights.min():.6f}, {weights.max():.6f}]")
            print(f"  ✅ Non-zero elements: {np.count_nonzero(weights)}")
            
        except Exception as e:
            verification_results["weights_valid"] = False
            print(f"  ❌ Error loading weights: {e}")
    else:
        verification_results["weights_valid"] = False
        print("  ❌ Decoder weights file missing")
    
    return verification_results


def generate_feature_visualizations(output_dir: str, top_k: int = 10):
    """Generate visualizations of top features"""
    
    output_path = Path(output_dir)
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    print(f"📊 Generating feature visualizations in: {viz_dir}")
    
    # Load data
    try:
        with open(output_path / "feature_analysis.json", 'r') as f:
            analysis = json.load(f)
        
        weights = np.load(output_path / "decoder_weights.npy")
        
    except Exception as e:
        print(f"❌ Error loading data for visualization: {e}")
        return
    
    features = analysis['features'][:top_k]
    
    # 1. Feature activation frequency distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    frequencies = [f['activation_frequency'] for f in analysis['features']]
    
    # Adaptive binning for frequency distribution
    freq_array = np.array(frequencies)
    unique_freqs = len(np.unique(freq_array))
    
    # Handle edge case where frequencies are too similar
    if unique_freqs < 3 or (freq_array.max() - freq_array.min()) < 1e-6:
        # Create a simple text display instead of histogram
        plt.text(0.5, 0.5, f'Activation frequencies\nMean: {np.mean(freq_array):.4f}\nStd: {np.std(freq_array):.4f}\nRange: [{freq_array.min():.4f}, {freq_array.max():.4f}]', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Feature Activation Frequency (Summary)')
    else:
        freq_bins = min(50, max(3, unique_freqs // 2))
        plt.hist(frequencies, bins=freq_bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Activation Frequency')
        plt.ylabel('Number of Features')
        plt.title('Feature Activation Frequency Distribution')
        plt.yscale('log')
    
    # 2. Top features by activation frequency
    plt.subplot(2, 2, 2)
    top_names = [f"Feature {f['feature_idx']}" for f in features]
    top_freqs = [f['activation_frequency'] for f in features]
    plt.barh(range(len(top_names)), top_freqs)
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel('Activation Frequency')
    plt.title(f'Top {top_k} Features by Activation Frequency')
    
    # 3. Max activation vs frequency scatter
    plt.subplot(2, 2, 3)
    max_acts = [f['max_activation'] for f in analysis['features']]
    freqs = [f['activation_frequency'] for f in analysis['features']]
    
    # Remove zero frequencies for log scale
    non_zero_mask = np.array(freqs) > 0
    if np.any(non_zero_mask):
        freqs_nz = np.array(freqs)[non_zero_mask]
        max_acts_nz = np.array(max_acts)[non_zero_mask]
        plt.scatter(freqs_nz, max_acts_nz, alpha=0.6, s=1)
        plt.xlabel('Activation Frequency')
        plt.ylabel('Max Activation')
        plt.title('Max Activation vs Frequency')
        plt.xscale('log')
    else:
        plt.text(0.5, 0.5, 'No active features found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Max Activation vs Frequency')
    
    # 4. Decoder weight norms (FIXED)
    plt.subplot(2, 2, 4)
    norms = np.linalg.norm(weights, axis=0)
    
    # Handle case where all norms are very similar (causing "too many bins" error)
    unique_norms = len(np.unique(norms))
    norm_range = norms.max() - norms.min()
    
    if unique_norms < 3 or norm_range < 1e-6:  # If there's essentially no variation
        # Create a simple summary display instead
        plt.text(0.5, 0.5, f'Decoder Weight Norms\nMean: {np.mean(norms):.6f}\nStd: {np.std(norms):.6f}\nRange: [{norms.min():.6f}, {norms.max():.6f}]\nAll norms very similar', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=11)
        plt.title('Decoder Weight Norm (Uniform Distribution)')
    else:
        n_bins = min(50, max(3, unique_norms // 2))  # Adaptive bin count
        plt.hist(norms, bins=n_bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Decoder Weight Norm')
        plt.ylabel('Number of Features')
        plt.title('Decoder Weight Norm Distribution')
        
        # Add statistics
        plt.axvline(np.mean(norms), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(norms):.3f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(viz_dir / "feature_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Individual top feature visualizations
    for i, feature in enumerate(features[:5]):  # Top 5 features
        plt.figure(figsize=(10, 6))
        
        # Decoder weights
        plt.subplot(1, 2, 1)
        decoder_weights = np.array(feature['decoder_weights'])
        plt.plot(decoder_weights)
        plt.title(f"Feature {feature['feature_idx']} - Decoder Weights")
        plt.xlabel('Model Dimension')
        plt.ylabel('Weight Value')
        
        # Activation statistics
        plt.subplot(1, 2, 2)
        stats = {
            'Max Activation': feature['max_activation'],
            'Mean Activation': feature['mean_activation'],
            'Activation Freq': feature['activation_frequency']
        }
        bars = plt.bar(range(len(stats)), list(stats.values()))
        plt.xticks(range(len(stats)), list(stats.keys()), rotation=45)
        plt.title(f"Feature {feature['feature_idx']} - Statistics")
        plt.ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f"feature_{feature['feature_idx']}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Generated visualizations for top {len(features)} features")


def create_neuronpedia_package(output_dir: str, package_name: str = None):
    """Create a package ready for Neuronpedia upload"""
    
    output_path = Path(output_dir)
    
    if package_name is None:
        package_name = f"miDiscovery_sae_{output_path.name}"
    
    package_dir = output_path / f"{package_name}_neuronpedia"
    package_dir.mkdir(exist_ok=True)
    
    print(f"📦 Creating Neuronpedia package: {package_dir}")
    
    # Copy essential files
    essential_files = [
        "neuronpedia_metadata.json",
        "decoder_weights.npy", 
        "feature_analysis.json",
        "sae_model.pt"
    ]
    
    for filename in essential_files:
        src = output_path / filename
        dst = package_dir / filename
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            print(f"  ✅ Copied {filename}")
        else:
            print(f"  ❌ Missing {filename}")
    
    # Create upload instructions
    instructions = """
# Neuronpedia Upload Instructions

## Files in this package:
- `neuronpedia_metadata.json`: Model and training metadata
- `decoder_weights.npy`: Feature decoder directions  
- `feature_analysis.json`: Complete feature analysis
- `sae_model.pt`: Full model checkpoint

## Upload steps:
1. Visit https://neuronpedia.org
2. Navigate to "Upload SAE" or "Contribute" section
3. Upload the metadata file first
4. Upload the decoder weights
5. Optionally upload the feature analysis for detailed view
6. Provide description of training methodology

## Training details:
- Model: Extracted from neuronpedia_metadata.json
- Training: See metadata for hyperparameters
- Quality: Check feature_analysis.json for active features count

## Verification:
Run the verification script on this directory to ensure completeness.
"""
    
    with open(package_dir / "UPLOAD_INSTRUCTIONS.md", 'w') as f:
        f.write(instructions)
    
    print(f"✅ Created Neuronpedia package in: {package_dir}")
    print(f"📋 Read {package_dir}/UPLOAD_INSTRUCTIONS.md for upload steps")


def main():
    parser = argparse.ArgumentParser(description="Verify miDiscovery SAE outputs for Neuronpedia compatibility")
    parser.add_argument("output_dir", help="Directory containing SAE outputs")
    parser.add_argument("--visualize", action="store_true", help="Generate feature visualizations")
    parser.add_argument("--package", action="store_true", help="Create Neuronpedia upload package")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top features to visualize")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output including data values")
    
    args = parser.parse_args()
    
    # Verify outputs
    results = verify_sae_outputs(args.output_dir, verbose=args.verbose)
    
    # Count successful verifications
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print()
    print("=" * 60)
    print(f"📊 VERIFICATION SUMMARY")
    print(f"   Passed: {passed_checks}/{total_checks} checks")
    print(f"   Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if passed_checks == total_checks:
        print("   🎉 All checks passed! Ready for Neuronpedia upload.")
    else:
        print("   ⚠️  Some checks failed. Review the issues above.")
        failed_checks = [key for key, value in results.items() if not value]
        print(f"   Failed checks: {', '.join(failed_checks)}")
    
    print("=" * 60)
    
    # Generate visualizations if requested
    if args.visualize:
        print()
        try:
            generate_feature_visualizations(args.output_dir, args.top_k)
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            print("   This doesn't affect Neuronpedia compatibility.")
    
    # Create package if requested
    if args.package:
        print()
        try:
            create_neuronpedia_package(args.output_dir)
        except Exception as e:
            print(f"❌ Error creating package: {e}")
    
    print()
    print("🚀 Next steps:")
    print("   1. Review any failed verification checks")
    print("   2. Use --visualize to inspect feature quality")  
    print("   3. Use --package to create Neuronpedia upload package")
    print("   4. Upload to Neuronpedia.org when ready")
    if not args.verbose:
        print("   5. Use --verbose/-v to see detailed values if debugging needed")


if __name__ == "__main__":
    main()