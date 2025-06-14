#!/usr/bin/env python3
"""Query and display training runs from the database"""

import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from core.db_utils import get_db, TrainingRun
    from sqlalchemy import text
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure PostgreSQL is running and src/core/db_utils.py exists")
    sys.exit(1)

import argparse
from datetime import datetime, timedelta

def list_recent_runs(limit=10):
    """List recent training runs"""
    try:
        db = get_db()
        
        query = text("""
            SELECT id, model_name, layer_name, status, created_at, 
                   final_loss, active_features, total_features
            FROM discovery.training_runs 
            ORDER BY created_at DESC 
            LIMIT :limit
        """)
        
        result = db.execute(query, {"limit": limit})
        runs = result.fetchall()
        
        if not runs:
            print("No training runs found in database")
            return
        
        print(f"\n📊 Recent Training Runs (Last {len(runs)}):")
        print("=" * 100)
        
        for run in runs:
            run_id = str(run[0])[:8]  # Show first 8 chars of UUID
            model = run[1]
            layer = run[2] 
            status = run[3]
            created = run[4].strftime("%Y-%m-%d %H:%M:%S") if run[4] else "Unknown"
            loss = f"{run[5]:.6f}" if run[5] else "N/A"
            active = run[6] if run[6] else "N/A" 
            total = run[7] if run[7] else "N/A"
            
            print(f"🔹 {run_id} | {model} | {layer} | {status} | {created}")
            print(f"   Loss: {loss} | Features: {active}/{total}")
            print()
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error querying database: {e}")

def main():
    parser = argparse.ArgumentParser(description="Query training runs from database")
    parser.add_argument("command", choices=["recent", "details"], 
                       help="Command to execute", nargs='?', default='recent')
    parser.add_argument("--limit", type=int, default=10, 
                       help="Number of recent runs to show")
    
    args = parser.parse_args()
    
    if args.command == "recent":
        list_recent_runs(args.limit)

if __name__ == "__main__":
    main()
