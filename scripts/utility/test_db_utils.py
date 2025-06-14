
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.db_utils import get_db, get_recent_training_runs, TrainingRun
from datetime import datetime

# Test database connection
try:
    db = get_db()
    print("✅ Database connection successful!")
    
    # Check if we have any training runs
    runs = get_recent_training_runs(db)
    print(f"📊 Found {len(runs)} training runs in database")
    
    if runs:
        print("\nRecent training runs:")
        for run in runs[:3]:
            print(f"  - {run.model_name} / {run.layer_name} - Status: {run.status}")
    
    db.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
