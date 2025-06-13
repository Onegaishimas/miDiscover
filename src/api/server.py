"""
Simple API server for MechInterp Discovery Service
Provides endpoints for SAE training and status monitoring
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import asyncio
import subprocess
from pathlib import Path

app = FastAPI(title="MechInterp Discovery API", version="1.0.0")

# In-memory storage for job status (use Redis in production)
jobs: Dict[str, Dict[str, Any]] = {}

class TrainingRequest(BaseModel):
    model_name: str = "microsoft/phi-2"
    layer_name: str = "model.layers.16.mlp.fc2"
    max_samples: int = 1000
    epochs: int = 3
    l1_coef: float = 1e-3
    use_db: bool = True

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    training_run_id: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "mechinterp-discovery"}

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str) -> JobStatus:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])

@app.post("/api/v1/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "job_id": job_id,
        "status": "starting",
        "progress": 0.0,
        "message": "Initializing training..."
    }
    
    # Start training in background
    background_tasks.add_task(run_training, job_id, request)
    
    return {"job_id": job_id, "status": "started"}

async def run_training(job_id: str, request: TrainingRequest):
    """Run SAE training in background"""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["message"] = "Starting SAE training..."
        
        # Build command
        cmd = [
            "python", "miDiscovery_sae_train.py",
            "--model-name", request.model_name,
            "--layer-name", request.layer_name,
            "--max-samples", str(request.max_samples),
            "--epochs", str(request.epochs),
            "--l1-coef", str(request.l1_coef)
        ]
        
        if request.use_db:
            cmd.append("--use-db")
        
        # Run training process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"
        )
        
        # Monitor progress (simplified)
        while process.poll() is None:
            jobs[job_id]["progress"] = min(jobs[job_id].get("progress", 0) + 0.1, 0.9)
            await asyncio.sleep(10)
        
        # Check result
        if process.returncode == 0:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["message"] = "Training completed successfully"
            
            # Try to extract training run ID from output
            stdout, _ = process.communicate()
            # Parse training run ID from logs if available
            
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Training failed with code {process.returncode}"
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"

@app.get("/api/v1/runs")
async def list_training_runs():
    """List recent training runs from database"""
    try:
        from db_utils import get_db, get_recent_training_runs
        db = get_db()
        runs = get_recent_training_runs(db, limit=20)
        db.close()
        
        return [
            {
                "id": str(run.id),
                "model_name": run.model_name,
                "layer_name": run.layer_name,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "final_loss": run.final_loss,
                "active_features": run.active_features,
                "total_features": run.total_features
            }
            for run in runs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
