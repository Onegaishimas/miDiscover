import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER', 'mechinterp')}:{os.getenv('POSTGRES_PASSWORD', 'mechinterp_dev_password')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'mechinterp_discovery')}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models (matching our schema)
class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = {"schema": "discovery"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False)
    layer_name = Column(String(255), nullable=False)
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))
    status = Column(String(50), default='running')
    
    # Hyperparameters
    d_model = Column(Integer, nullable=False)
    d_hidden = Column(Integer, nullable=False)
    expansion_factor = Column(Integer, nullable=False)
    l1_coefficient = Column(Float, nullable=False)
    learning_rate = Column(Float, nullable=False)
    num_epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    max_samples = Column(Integer)
    
    # Results
    final_loss = Column(Float)
    final_reconstruction_loss = Column(Float)
    final_l1_loss = Column(Float)
    active_features = Column(Integer)
    total_features = Column(Integer)
    sparsity_level = Column(Float)
    
    # Metadata
    gpu_used = Column(String(255))
    training_duration_seconds = Column(Integer)
    dataset_name = Column(String(255))
    dataset_size = Column(Integer)
    error_message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

class EpochMetric(Base):
    __tablename__ = "epoch_metrics"
    __table_args__ = {"schema": "discovery"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_run_id = Column(UUID(as_uuid=True), nullable=False)
    epoch = Column(Integer, nullable=False)
    total_loss = Column(Float, nullable=False)
    reconstruction_loss = Column(Float, nullable=False)
    l1_loss = Column(Float, nullable=False)
    learning_rate = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)

class DiscoveredFeature(Base):
    __tablename__ = "discovered_features"
    __table_args__ = {"schema": "discovery"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_run_id = Column(UUID(as_uuid=True), nullable=False)
    feature_idx = Column(Integer, nullable=False)
    
    # Statistics
    max_activation = Column(Float, nullable=False)
    mean_activation = Column(Float, nullable=False)
    activation_frequency = Column(Float, nullable=False)
    decoder_norm = Column(Float)
    
    # JSONB fields for complex data
    top_activations = Column(JSONB)
    concept_examples = Column(JSONB)
    
    # Interpretability
    interpreted_concept = Column(String(500))
    concept_confidence = Column(Float)
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

class SAEModel(Base):
    __tablename__ = "sae_models"
    __table_args__ = {"schema": "discovery"}
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_run_id = Column(UUID(as_uuid=True), nullable=False)
    
    # Storage location
    file_path = Column(String(500))
    file_size_bytes = Column(Integer)  # Changed from BIGINT to Integer for SQLAlchemy compatibility
    checksum = Column(String(64))
    
    # Model metadata
    neuronpedia_compatible = Column(Boolean, default=False)
    exported_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

# Database helper functions
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def create_training_run(config: Any, db: Session) -> TrainingRun:
    """Create a new training run record"""
    training_run = TrainingRun(
        model_name=config.model_name,
        layer_name=config.layer_name,
        d_model=config.d_model,
        d_hidden=config.d_model * config.expansion_factor,
        expansion_factor=config.expansion_factor,
        l1_coefficient=config.l1_coefficient,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        max_samples=config.max_samples,
        dataset_name=config.dataset_name,
        gpu_used=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    db.add(training_run)
    db.commit()
    db.refresh(training_run)
    return training_run

def log_epoch_metrics(training_run_id: uuid.UUID, epoch: int, losses: Dict[str, float], db: Session):
    """Log metrics for a training epoch"""
    metric = EpochMetric(
        training_run_id=training_run_id,
        epoch=epoch,
        total_loss=losses['total'],
        reconstruction_loss=losses['reconstruction'],
        l1_loss=losses['l1'],
        learning_rate=losses.get('lr')
    )
    db.add(metric)
    db.commit()

def update_training_run_completion(training_run: TrainingRun, results: Dict[str, Any], db: Session):
    """Update training run with final results"""
    training_run.completed_at = datetime.utcnow()
    training_run.status = 'completed'
    training_run.final_loss = results.get('final_loss')
    training_run.final_reconstruction_loss = results.get('final_reconstruction_loss')
    training_run.final_l1_loss = results.get('final_l1_loss')
    training_run.active_features = results.get('active_features')
    training_run.total_features = results.get('total_features')
    training_run.sparsity_level = results.get('sparsity_level')
    training_run.training_duration_seconds = results.get('duration_seconds')
    
    db.commit()

def log_discovered_features(training_run_id: uuid.UUID, features: List[Dict[str, Any]], db: Session, batch_size: int = 1000):
    """Log discovered features in batches for efficiency"""
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        feature_objects = []
        
        for feature in batch:
            feature_obj = DiscoveredFeature(
                training_run_id=training_run_id,
                feature_idx=feature['feature_idx'],
                max_activation=feature['max_activation'],
                mean_activation=feature['mean_activation'],
                activation_frequency=feature['activation_frequency'],
                decoder_norm=feature.get('decoder_norm'),
                top_activations=feature.get('top_activations', [])
            )
            feature_objects.append(feature_obj)
        
        db.bulk_save_objects(feature_objects)
        db.commit()
        print(f"  Saved features {i} to {min(i + batch_size, len(features))}")

def log_sae_model(training_run_id: uuid.UUID, model_path: str, file_size: int, checksum: str, db: Session) -> SAEModel:
    """Log SAE model storage information"""
    sae_model = SAEModel(
        training_run_id=training_run_id,
        file_path=model_path,
        file_size_bytes=file_size,
        checksum=checksum,
        neuronpedia_compatible=True,
        exported_at=datetime.utcnow()
    )
    db.add(sae_model)
    db.commit()
    db.refresh(sae_model)
    return sae_model

# Query helper functions
def get_recent_training_runs(db: Session, limit: int = 10) -> List[TrainingRun]:
    """Get recent training runs"""
    return db.query(TrainingRun)\
        .order_by(TrainingRun.created_at.desc())\
        .limit(limit)\
        .all()

def get_best_performing_runs(db: Session, model_name: str, layer_name: str, limit: int = 5) -> List[TrainingRun]:
    """Get best performing runs for a specific model/layer combination"""
    return db.query(TrainingRun)\
        .filter(TrainingRun.model_name == model_name)\
        .filter(TrainingRun.layer_name == layer_name)\
        .filter(TrainingRun.status == 'completed')\
        .order_by(TrainingRun.final_loss)\
        .limit(limit)\
        .all()

def get_active_features_by_run(db: Session, training_run_id: uuid.UUID, min_frequency: float = 0.001) -> List[DiscoveredFeature]:
    """Get active features for a training run"""
    return db.query(DiscoveredFeature)\
        .filter(DiscoveredFeature.training_run_id == training_run_id)\
        .filter(DiscoveredFeature.activation_frequency > min_frequency)\
        .order_by(DiscoveredFeature.activation_frequency.desc())\
        .all()