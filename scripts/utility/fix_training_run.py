from db_utils import get_db, TrainingRun
import uuid

# Get the run ID from your output
run_id = uuid.UUID('9c6fac72-9ce2-49e0-bfa7-ab7dd9affa3e')

db = get_db()
run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()

if run:
    # Update with the values from your output
    run.status = 'completed'
    run.final_loss = 0.012309
    run.final_reconstruction_loss = 0.012287
    run.final_l1_loss = 0.044710
    run.active_features = 20429
    run.total_features = 20480
    run.sparsity_level = 20429 / 20480  # 99.8%
    run.training_duration_seconds = 195  # approximately based on your times
    
    db.commit()
    print(f"✅ Updated training run {run_id} to completed status")
else:
    print(f"❌ Training run {run_id} not found")

db.close()
