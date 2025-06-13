-- Create schema for MechInterp Discovery module
CREATE SCHEMA IF NOT EXISTS discovery;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for experiment tracking
CREATE TABLE discovery.training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    layer_name VARCHAR(255) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    
    -- Hyperparameters
    d_model INTEGER NOT NULL,
    d_hidden INTEGER NOT NULL,
    expansion_factor INTEGER NOT NULL,
    l1_coefficient FLOAT NOT NULL,
    learning_rate FLOAT NOT NULL,
    num_epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    max_samples INTEGER,
    
    -- Results
    final_loss FLOAT,
    final_reconstruction_loss FLOAT,
    final_l1_loss FLOAT,
    active_features INTEGER,
    total_features INTEGER,
    sparsity_level FLOAT,
    
    -- Metadata
    gpu_used VARCHAR(255),
    training_duration_seconds INTEGER,
    dataset_name VARCHAR(255),
    dataset_size INTEGER,
    error_message TEXT,
    
    -- Indexes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for common queries
CREATE INDEX idx_training_runs_model_layer ON discovery.training_runs(model_name, layer_name);
CREATE INDEX idx_training_runs_status ON discovery.training_runs(status);
CREATE INDEX idx_training_runs_created ON discovery.training_runs(created_at DESC);

-- Epoch metrics for tracking training progress
CREATE TABLE discovery.epoch_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES discovery.training_runs(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    total_loss FLOAT NOT NULL,
    reconstruction_loss FLOAT NOT NULL,
    l1_loss FLOAT NOT NULL,
    learning_rate FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(training_run_id, epoch)
);

-- Discovered features table
CREATE TABLE discovery.discovered_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES discovery.training_runs(id) ON DELETE CASCADE,
    feature_idx INTEGER NOT NULL,
    
    -- Statistics
    max_activation FLOAT NOT NULL,
    mean_activation FLOAT NOT NULL,
    activation_frequency FLOAT NOT NULL,
    decoder_norm FLOAT,
    
    -- Top activating examples (stored as JSONB)
    top_activations JSONB,
    
    -- Interpretability (to be filled by analysis)
    interpreted_concept VARCHAR(500),
    concept_confidence FLOAT,
    concept_examples JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(training_run_id, feature_idx)
);

-- Create indexes for feature analysis
CREATE INDEX idx_features_training_run ON discovery.discovered_features(training_run_id);
CREATE INDEX idx_features_activation_freq ON discovery.discovered_features(activation_frequency DESC);
CREATE INDEX idx_features_concept ON discovery.discovered_features(interpreted_concept);

-- Model storage references
CREATE TABLE discovery.sae_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    training_run_id UUID NOT NULL REFERENCES discovery.training_runs(id) ON DELETE CASCADE,
    
    -- Storage location
    file_path VARCHAR(500),
    file_size_bytes BIGINT,
    checksum VARCHAR(64),
    
    -- Model metadata
    neuronpedia_compatible BOOLEAN DEFAULT false,
    exported_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_training_runs_updated_at BEFORE UPDATE ON discovery.training_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
