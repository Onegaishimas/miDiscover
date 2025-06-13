#!/usr/bin/env python3
"""
Working script to explore features stored in the database
"""

from db_utils import get_db, TrainingRun, DiscoveredFeature
import argparse
import json

def find_run_by_partial_id(partial_id):
    """Find a training run by partial ID"""
    try:
        db = get_db()
        
        # Get all runs and find one that starts with the partial ID
        runs = db.query(TrainingRun).all()
        
        for run in runs:
            if str(run.id).startswith(partial_id):
                db.close()
                return run
        
        db.close()
        return None
        
    except Exception as e:
        print(f"Error finding run: {e}")
        try:
            db.close()
        except:
            pass
        return None

def list_training_runs():
    """List all training runs with their feature counts"""
    
    try:
        db = get_db()
        
        runs = db.query(TrainingRun).order_by(TrainingRun.started_at.desc()).all()
        
        print(f"\n🗂️  Available Training Runs:")
        print("=" * 100)
        print(f"{'ID':<10} {'Date':<12} {'Model':<10} {'Layer':<8} {'Status':<10} {'Loss':<12} {'Features':<10} {'Active':<8} {'Duration':<8}")
        print("-" * 100)
        
        for run in runs:
            # Count features for this run
            feature_count = db.query(DiscoveredFeature).filter(
                DiscoveredFeature.training_run_id == run.id
            ).count()
            
            active_features = db.query(DiscoveredFeature).filter(
                DiscoveredFeature.training_run_id == run.id,
                DiscoveredFeature.activation_frequency > 0.001
            ).count()
            
            date_str = run.started_at.strftime("%m-%d %H:%M") if run.started_at else "N/A"
            model_str = run.model_name.split('/')[-1] if run.model_name else "N/A"
            layer_str = run.layer_name.split('.')[-1] if run.layer_name else "N/A"
            loss_str = f"{run.final_loss:.6f}" if run.final_loss else "N/A"
            duration_str = f"{run.training_duration_seconds}s" if run.training_duration_seconds else "N/A"
            
            print(f"{str(run.id)[:8]:<10} {date_str:<12} {model_str:<10} {layer_str:<8} "
                  f"{run.status:<10} {loss_str:<12} {feature_count:<10} {active_features:<8} {duration_str:<8}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error listing training runs: {e}")
        try:
            db.close()
        except:
            pass

def show_top_features(partial_run_id, top_k=20, min_frequency=0.001):
    """Show top features for a specific training run"""
    
    run = find_run_by_partial_id(partial_run_id)
    if not run:
        print(f"❌ Training run starting with '{partial_run_id}' not found")
        return
    
    try:
        db = get_db()
        
        print(f"\n🏆 Top {top_k} Features for Training Run {str(run.id)[:8]}")
        print(f"Model: {run.model_name}, Layer: {run.layer_name}")
        print(f"Started: {run.started_at}, Status: {run.status}")
        print("=" * 100)
        
        # Get top features by activation frequency
        features = db.query(DiscoveredFeature).filter(
            DiscoveredFeature.training_run_id == run.id,
            DiscoveredFeature.activation_frequency > min_frequency
        ).order_by(DiscoveredFeature.activation_frequency.desc()).limit(top_k).all()
        
        if not features:
            print(f"No features found with activation frequency > {min_frequency}")
            return
        
        print(f"{'Rank':<6} {'Feature ID':<12} {'Frequency':<12} {'Max Act':<10} {'Mean Act':<12} {'Norm':<10} {'Concept':<30}")
        print("-" * 100)
        
        for i, feature in enumerate(features):
            concept = feature.interpreted_concept[:27] + "..." if feature.interpreted_concept and len(feature.interpreted_concept) > 30 else (feature.interpreted_concept or "Not interpreted")
            norm_str = f"{feature.decoder_norm:.4f}" if feature.decoder_norm else "N/A"
            
            print(f"{i+1:<6} {feature.feature_idx:<12} "
                  f"{feature.activation_frequency:<12.6f} "
                  f"{feature.max_activation:<10.4f} "
                  f"{feature.mean_activation:<12.6f} "
                  f"{norm_str:<10} {concept:<30}")
        
        db.close()
        
        print(f"\n💡 To see details of a feature:")
        print(f"  python working_explore_features.py details {partial_run_id} {features[0].feature_idx}")
        
    except Exception as e:
        print(f"❌ Error showing features: {e}")
        try:
            db.close()
        except:
            pass

def show_feature_details(partial_run_id, feature_idx):
    """Show detailed information about a specific feature"""
    
    run = find_run_by_partial_id(partial_run_id)
    if not run:
        print(f"❌ Training run starting with '{partial_run_id}' not found")
        return
    
    try:
        db = get_db()
        
        # Get the feature
        feature = db.query(DiscoveredFeature).filter(
            DiscoveredFeature.training_run_id == run.id,
            DiscoveredFeature.feature_idx == feature_idx
        ).first()
        
        if not feature:
            print(f"❌ Feature {feature_idx} not found in training run {str(run.id)[:8]}")
            return
        
        print(f"\n🔍 Feature {feature_idx} Details")
        print("=" * 50)
        print(f"Training Run: {str(run.id)[:8]} ({run.model_name}, {run.layer_name})")
        print(f"Feature Index: {feature.feature_idx}")
        print(f"Activation Frequency: {feature.activation_frequency:.6f} ({feature.activation_frequency*100:.3f}%)")
        print(f"Max Activation: {feature.max_activation:.6f}")
        print(f"Mean Activation: {feature.mean_activation:.6f}")
        print(f"Decoder Norm: {feature.decoder_norm:.6f}" if feature.decoder_norm else "Decoder Norm: N/A")
        print(f"Interpreted Concept: {feature.interpreted_concept or 'Not interpreted'}")
        print(f"Concept Confidence: {feature.concept_confidence:.3f}" if feature.concept_confidence else "Concept Confidence: N/A")
        
        # Show top activations if available
        if feature.top_activations:
            print(f"\nTop Activations:")
            if isinstance(feature.top_activations, str):
                top_acts = json.loads(feature.top_activations)
            else:
                top_acts = feature.top_activations
            
            for i, activation in enumerate(top_acts[:10]):
                print(f"  {i+1}: {activation:.6f}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error showing feature details: {e}")
        try:
            db.close()
        except:
            pass

def analyze_feature_distribution(partial_run_id):
    """Analyze the distribution of features for a training run"""
    
    run = find_run_by_partial_id(partial_run_id)
    if not run:
        print(f"❌ Training run starting with '{partial_run_id}' not found")
        return
    
    try:
        db = get_db()
        
        # Get all features for this run
        features = db.query(DiscoveredFeature).filter(
            DiscoveredFeature.training_run_id == run.id
        ).all()
        
        if not features:
            print(f"No features found for training run {str(run.id)[:8]}")
            return
        
        print(f"\n📊 Feature Distribution Analysis for {str(run.id)[:8]}")
        print(f"Model: {run.model_name}, Layer: {run.layer_name}")
        print("=" * 60)
        
        # Calculate statistics
        frequencies = [f.activation_frequency for f in features]
        max_activations = [f.max_activation for f in features]
        mean_activations = [f.mean_activation for f in features]
        
        import numpy as np
        
        print(f"Total Features: {len(features)}")
        print(f"\nActivation Frequency Statistics:")
        print(f"  Mean: {np.mean(frequencies):.6f}")
        print(f"  Median: {np.median(frequencies):.6f}")
        print(f"  Std: {np.std(frequencies):.6f}")
        print(f"  Min: {np.min(frequencies):.6f}")
        print(f"  Max: {np.max(frequencies):.6f}")
        
        print(f"\nMax Activation Statistics:")
        print(f"  Mean: {np.mean(max_activations):.6f}")
        print(f"  Median: {np.median(max_activations):.6f}")
        print(f"  Std: {np.std(max_activations):.6f}")
        print(f"  Min: {np.min(max_activations):.6f}")
        print(f"  Max: {np.max(max_activations):.6f}")
        
        # Show distribution by activation frequency bins
        print(f"\nFeature Distribution by Activation Frequency:")
        bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 1.0]
        
        for i in range(len(bins)-1):
            count = len([f for f in frequencies if bins[i] < f <= bins[i+1]])
            percentage = (count / len(features)) * 100
            print(f"  {bins[i]:5.3f} - {bins[i+1]:5.3f}: {count:5d} features ({percentage:5.1f}%)")
        
        # Show features with concepts interpreted
        interpreted_count = len([f for f in features if f.interpreted_concept])
        print(f"\nInterpretation Status:")
        print(f"  Interpreted: {interpreted_count} features ({interpreted_count/len(features)*100:.1f}%)")
        print(f"  Not interpreted: {len(features) - interpreted_count} features")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error analyzing features: {e}")
        try:
            db.close()
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Explore features stored in the database")
    parser.add_argument("command", choices=["list", "top", "details", "analyze"], 
                       help="Command to execute")
    parser.add_argument("run_id", nargs="?", help="Training run ID (first 8 characters)")
    parser.add_argument("feature_id", nargs="?", type=int, help="Feature ID for details command")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to show")
    parser.add_argument("--min-freq", type=float, default=0.001, help="Minimum activation frequency")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_training_runs()
        print(f"\n💡 Usage examples:")
        print(f"  python working_explore_features.py top 85037654")
        print(f"  python working_explore_features.py analyze 85037654")
        print(f"  python working_explore_features.py details 85037654 12345")
    
    elif args.command == "top":
        if not args.run_id:
            print("❌ run_id required for top command")
            print("Usage: python working_explore_features.py top 85037654")
            return
        show_top_features(args.run_id, args.top_k, args.min_freq)
    
    elif args.command == "details":
        if not args.run_id or args.feature_id is None:
            print("❌ Both run_id and feature_id required for details command")
            print("Usage: python working_explore_features.py details 85037654 12345")
            return
        show_feature_details(args.run_id, args.feature_id)
    
    elif args.command == "analyze":
        if not args.run_id:
            print("❌ run_id required for analyze command")
            print("Usage: python working_explore_features.py analyze 85037654")
            return
        analyze_feature_distribution(args.run_id)

if __name__ == "__main__":
    main()