#!/usr/bin/env python3
"""
Fix running training records that should be completed
"""

from db_utils import get_db, TrainingRun
from datetime import datetime
import uuid

def fix_running_records():
    """Fix training runs that are stuck in 'running' status"""
    
    try:
        db = get_db()
        
        # Find all running records
        running_runs = db.query(TrainingRun).filter(
            TrainingRun.status == 'running'
        ).all()
        
        print(f"Found {len(running_runs)} running records to check:")
        
        for run in running_runs:
            print(f"\n🔍 Checking run {str(run.id)[:8]}:")
            print(f"   Started: {run.started_at}")
            print(f"   Model: {run.model_name}")
            print(f"   Layer: {run.layer_name}")
            
            # Check if this run has features (indicating it completed)
            from db_utils import DiscoveredFeature
            feature_count = db.query(DiscoveredFeature).filter(
                DiscoveredFeature.training_run_id == run.id
            ).count()
            
            print(f"   Features found: {feature_count}")
            
            if feature_count > 0:
                # This run completed successfully, update it
                print(f"   ✅ Marking as completed (has {feature_count} features)")
                
                run.status = 'completed'
                
                # Handle timezone-aware completion time
                if run.started_at and run.started_at.tzinfo is not None:
                    from datetime import timezone
                    run.completed_at = datetime.now(timezone.utc)
                else:
                    run.completed_at = datetime.now()
                run.total_features = feature_count
                
                # Calculate active features
                active_features = db.query(DiscoveredFeature).filter(
                    DiscoveredFeature.training_run_id == run.id,
                    DiscoveredFeature.activation_frequency > 0.001
                ).count()
                
                run.active_features = active_features
                run.sparsity_level = (active_features / feature_count) * 100 if feature_count > 0 else 0
                
                # Calculate duration if possible
                if run.started_at:
                    # Handle timezone-aware vs naive datetime
                    now = datetime.now()
                    if run.started_at.tzinfo is not None:
                        from datetime import timezone
                        now = datetime.now(timezone.utc)
                    duration = (now - run.started_at).total_seconds()
                    run.training_duration_seconds = int(duration)
                
                db.add(run)
                
        # Commit all changes
        db.commit()
        db.close()
        
        print(f"\n✅ Fixed {len([r for r in running_runs if db.query(DiscoveredFeature).filter(DiscoveredFeature.training_run_id == r.id).count() > 0])} completed runs")
        
    except Exception as e:
        print(f"❌ Error fixing records: {e}")
        print(f"Error type: {type(e).__name__}")
        try:
            db.close()
        except:
            pass

def test_database_update():
    """Test if we can update a training run record"""
    
    try:
        db = get_db()
        
        # Find the most recent running record
        recent_run = db.query(TrainingRun).filter(
            TrainingRun.status == 'running'
        ).order_by(TrainingRun.started_at.desc()).first()
        
        if recent_run:
            print(f"🧪 Testing update on run {str(recent_run.id)[:8]}")
            
            # Try to update it
            recent_run.updated_at = datetime.now()
            db.add(recent_run)
            db.commit()
            
            print("✅ Database update test successful")
        else:
            print("No running records found to test")
            
        db.close()
        
    except Exception as e:
        print(f"❌ Database update test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        try:
            db.close()
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_database_update()
    else:
        fix_running_records()