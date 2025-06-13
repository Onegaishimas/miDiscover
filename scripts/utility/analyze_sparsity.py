#!/usr/bin/env python3
"""
Analyze the sparsity of the most recent SAE training
"""

import json
import numpy as np
from pathlib import Path

def analyze_sae_sparsity(output_dir="./sae_outputs"):
    """Analyze the actual sparsity of trained SAE"""
    
    # Load feature analysis
    analysis_path = Path(output_dir) / "feature_analysis.json"
    
    if not analysis_path.exists():
        print(f"❌ Feature analysis not found at {analysis_path}")
        return
    
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    
    features = analysis['features']
    activation_frequencies = [f['activation_frequency'] for f in features]
    
    print(f"🔍 SAE Sparsity Analysis")
    print(f"=" * 50)
    print(f"Model: {analysis['model_name']}")
    print(f"Layer: {analysis['layer_name']}")
    print(f"Total features: {len(features)}")
    print()
    
    # Calculate statistics
    freq_array = np.array(activation_frequencies)
    
    print(f"📊 Activation Frequency Statistics:")
    print(f"   Mean: {freq_array.mean():.6f}")
    print(f"   Median: {np.median(freq_array):.6f}")
    print(f"   Std: {freq_array.std():.6f}")
    print(f"   Min: {freq_array.min():.6f}")
    print(f"   Max: {freq_array.max():.6f}")
    print()
    
    # Different sparsity thresholds
    thresholds = [0.0001, 0.001, 0.01, 0.05, 0.1]
    
    print(f"🎯 Active Features by Threshold:")
    for threshold in thresholds:
        active = np.sum(freq_array > threshold)
        percentage = (active / len(features)) * 100
        print(f"   >{threshold*100:5.1f}%: {active:5d} features ({percentage:5.1f}% active)")
    
    print()
    
    # Show distribution
    print(f"📈 Activation Frequency Distribution:")
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist, _ = np.histogram(freq_array, bins=bins)
    
    for i in range(len(bins)-1):
        count = hist[i]
        percentage = (count / len(features)) * 100
        print(f"   {bins[i]:5.3f} - {bins[i+1]:5.3f}: {count:5d} features ({percentage:5.1f}%)")
    
    print()
    
    # Top 10 most active features
    print(f"🏆 Top 10 Most Active Features:")
    sorted_features = sorted(features, key=lambda x: x['activation_frequency'], reverse=True)
    for i, feature in enumerate(sorted_features[:10]):
        print(f"   {i+1:2d}. Feature {feature['feature_idx']:5d}: {feature['activation_frequency']:.6f} freq, {feature['max_activation']:.3f} max")
    
    print()
    
    # Recommend proper sparsity threshold
    # A good SAE should have most features inactive
    recommended_threshold = 0.01  # 1% activation frequency
    active_at_rec = np.sum(freq_array > recommended_threshold)
    rec_percentage = (active_at_rec / len(features)) * 100
    
    print(f"💡 Recommendations:")
    print(f"   For interpretability, aim for <20% active features")
    print(f"   Current: {rec_percentage:.1f}% active at {recommended_threshold*100}% threshold")
    
    if rec_percentage > 50:
        print(f"   ⚠️  Very dense activation - increase L1 coefficient or train longer")
    elif rec_percentage > 20:
        print(f"   ⚠️  Moderately dense - consider increasing L1 coefficient")
    else:
        print(f"   ✅ Good sparsity level")
    
    # Calculate average sparsity across all activations
    mean_activation_freq = np.mean(freq_array)
    print(f"   Average activation frequency: {mean_activation_freq:.4f} ({mean_activation_freq*100:.2f}%)")
    print(f"   This means on average, {mean_activation_freq*100:.2f}% of features fire for any given input")

if __name__ == "__main__":
    analyze_sae_sparsity()