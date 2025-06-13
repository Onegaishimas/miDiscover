#!/usr/bin/env python3
"""
Query and visualize training runs from the database
"""

from db_utils import (
    get_db, get_recent_training_runs, get_best_performing_runs,
    get_active_features_by_run, TrainingRun, EpochMetric
)
from sqlalchemy import func
import argparse
from datetime import datetime
from tabulate import tabulate


def show_recent_runs():
    """Display recent training runs"""
    db = get_db()
    
    runs = get_recent_training_runs(db, limit=10)
    
    if not runs:
        print("No training runs found in database.")
        return
    
    # Prepare data for table
    table_data = []
    for run in runs:
        duration = f"{run.training_duration_seconds}s" if run.training_duration_seconds else "N/A"
        sparsity = f"{run.sparsity_level*100:.1f}%" if run.sparsity_level else "N/A"
        
        table_data.append([
            run.created_at.strftime("%Y-%m-%d %H:%M"),
            run.model_name.split('/')[-1],  # Short model name
            run.layer_name.split('.')[-1],  # Short layer name
            run.status,
            f"{run.final_loss:.6f}" if run.final_loss else "N/A",
            sparsity,
            duration,
            str(run.id)[:8]  # Short ID
        ])
    
    headers = ["Started", "Model", "Layer", "Status", "Final Loss", "Sparsity", "Duration", "ID"]
    print("\n📊 Recent Training Runs:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    db.close()


def show_run_details(run_id: str):
    """Show detailed information about a specific run"""
    db = get_db()
    
    # Get the run
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not run:
        print(f"Training run {run_id} not found.")
        return
    
    print(f"\n🔍 Training Run Details: {run.id}")
    print("=" * 60)
    
    # Basic info
    print(f"Model: {run.model_name}")
    print(f"Layer: {run.layer_name}")
    print(f"Status: {run.status}")
    print(f"Started: {run.started_at}")
    print(f"Completed: {run.completed_at}")
    
    # Hyperparameters
    print(f"\n📐 Hyperparameters:")
    print(f"  - Model dimension: {run.d_model}")
    print(f"  - Hidden dimension: {run.d_hidden} ({run.expansion_factor}x expansion)")
    print(f"  - L1 coefficient: {run.l1_coefficient}")
    print(f"  - Learning rate: {run.learning_rate}")
    print(f"  - Epochs: {run.num_epochs}")
    print(f"  - Batch size: {run.batch_size}")
    
    # Results
    if run.status == 'completed':
        print(f"\n📈 Results:")
        print(f"  - Final loss: {run.final_loss:.6f}")
        print(f"  - Reconstruction loss: {run.final_reconstruction_loss:.6f}")
        print(f"  - L1 loss: {run.final_l1_loss:.6f}")
        print(f"  - Active features: {run.active_features}/{run.total_features} ({run.sparsity_level*100:.1f}%)")
        print(f"  - Training duration: {run.training_duration_seconds}s")
        
        # Get epoch metrics
        epochs = db.query(EpochMetric)\
            .filter(EpochMetric.training_run_id == run.id)\
            .order_by(EpochMetric.epoch)\
            .all()
        
        if epochs:
            print(f"\n📊 Training Progress:")
            epoch_data = []
            for e in epochs:
                epoch_data.append([
                    e.epoch,
                    f"{e.total_loss:.6f}",
                    f"{e.reconstruction_loss:.6f}",
                    f"{e.l1_loss:.6f}"
                ])
            
            headers = ["Epoch", "Total Loss", "Recon Loss", "L1 Loss"]
            print(tabulate(epoch_data, headers=headers, tablefmt="simple"))
    
    elif run.status == 'failed':
        print(f"\n❌ Error: {run.error_message}")
    
    db.close()


def compare_runs(model_name: str, layer_name: str):
    """Compare different runs for the same model/layer"""
    db = get_db()
    
    runs = get_best_performing_runs(db, model_name, layer_name, limit=10)
    
    if not runs:
        print(f"No completed runs found for {model_name} / {layer_name}")
        return
    
    print(f"\n🏆 Best Runs for {model_name} / {layer_name}:")
    
    table_data = []
    for i, run in enumerate(runs):
        table_data.append([
            i + 1,
            run.created_at.strftime("%Y-%m-%d"),
            f"{run.l1_coefficient:.1e}",
            run.num_epochs,
            f"{run.final_loss:.6f}",
            f"{run.sparsity_level*100:.1f}%",
            f"{run.active_features}",
            str(run.id)[:8]
        ])
    
    headers = ["Rank", "Date", "L1 Coef", "Epochs", "Loss", "Sparsity", "Active", "ID"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Query miDiscovery training runs")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Recent runs command
    parser_recent = subparsers.add_parser('recent', help='Show recent training runs')
    
    # Run details command
    parser_details = subparsers.add_parser('details', help='Show details of a specific run')
    parser_details.add_argument('run_id', help='Training run ID (or first 8 characters)')
    
    # Compare runs command
    parser_compare = subparsers.add_parser('compare', help='Compare runs for same model/layer')
    parser_compare.add_argument('--model', default='microsoft/phi-2', help='Model name')
    parser_compare.add_argument('--layer', default='model.layers.16.mlp.fc2', help='Layer name')
    
    args = parser.parse_args()
    
    if args.command == 'recent':
        show_recent_runs()
    elif args.command == 'details':
        # Handle partial IDs
        db = get_db()
        runs = db.query(TrainingRun).all()
        matching_run = None
        for run in runs:
            if str(run.id).startswith(args.run_id):
                matching_run = run
                break
        db.close()
        
        if matching_run:
            show_run_details(str(matching_run.id))
        else:
            print(f"No run found matching ID: {args.run_id}")
    elif args.command == 'compare':
        compare_runs(args.model, args.layer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
