-- Neuronpedia Integration Schema
-- This extends the discovery schema to work with Neuronpedia

-- Enable pgvector extension for Neuronpedia
CREATE EXTENSION IF NOT EXISTS vector;

-- Create public schema tables that Neuronpedia expects
-- (These will be created by Neuronpedia's migrations, but we set up compatibility)

-- Integration view to expose Discovery data to Neuronpedia format
CREATE OR REPLACE VIEW public.mechinterp_saes AS
SELECT 
    tr.id::text as id,
    tr.model_name,
    tr.layer_name,
    tr.d_model,
    tr.d_hidden as d_sae,
    tr.l1_coefficient,
    tr.learning_rate,
    tr.num_epochs,
    tr.final_loss,
    tr.sparsity_level,
    tr.active_features,
    tr.total_features,
    tr.completed_at,
    sm.file_path,
    sm.file_size_bytes,
    -- Neuronpedia compatible metadata
    jsonb_build_object(
        'training_config', jsonb_build_object(
            'expansion_factor', tr.expansion_factor,
            'batch_size', tr.batch_size,
            'max_samples', tr.max_samples,
            'dataset_name', tr.dataset_name
        ),
        'performance', jsonb_build_object(
            'final_reconstruction_loss', tr.final_reconstruction_loss,
            'final_l1_loss', tr.final_l1_loss,
            'sparsity_level', tr.sparsity_level
        )
    ) as metadata
FROM discovery.training_runs tr
LEFT JOIN discovery.sae_models sm ON tr.id = sm.training_run_id
WHERE tr.status = 'completed';

-- Integration view for features
CREATE OR REPLACE VIEW public.mechinterp_features AS
SELECT 
    df.id::text as id,
    df.training_run_id::text,
    df.feature_idx,
    df.max_activation,
    df.mean_activation,
    df.activation_frequency,
    df.decoder_norm,
    df.interpreted_concept,
    df.concept_confidence,
    df.top_activations,
    df.concept_examples,
    -- Add Neuronpedia-compatible fields
    'mechinterp_discovery' as source,
    df.created_at as discovered_at
FROM discovery.discovered_features df;

-- Function to export SAE for Neuronpedia upload
CREATE OR REPLACE FUNCTION public.export_sae_for_neuronpedia(training_run_uuid UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'training_run_id', tr.id,
        'model_name', tr.model_name,
        'layer_name', tr.layer_name,
        'architecture', json_build_object(
            'd_model', tr.d_model,
            'd_sae', tr.d_hidden,
            'expansion_factor', tr.expansion_factor
        ),
        'training', json_build_object(
            'l1_coefficient', tr.l1_coefficient,
            'learning_rate', tr.learning_rate,
            'num_epochs', tr.num_epochs,
            'batch_size', tr.batch_size,
            'dataset_name', tr.dataset_name
        ),
        'performance', json_build_object(
            'final_loss', tr.final_loss,
            'reconstruction_loss', tr.final_reconstruction_loss,
            'l1_loss', tr.final_l1_loss,
            'sparsity_level', tr.sparsity_level,
            'active_features', tr.active_features,
            'total_features', tr.total_features
        ),
        'files', json_build_object(
            'model_path', sm.file_path,
            'file_size', sm.file_size_bytes,
            'checksum', sm.checksum
        ),
        'features_count', (
            SELECT COUNT(*) FROM discovery.discovered_features 
            WHERE training_run_id = tr.id
        )
    ) INTO result
    FROM discovery.training_runs tr
    LEFT JOIN discovery.sae_models sm ON tr.id = sm.training_run_id
    WHERE tr.id = training_run_uuid;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON VIEW public.mechinterp_saes IS 'Neuronpedia-compatible view of MechInterp SAE training runs';
COMMENT ON VIEW public.mechinterp_features IS 'Neuronpedia-compatible view of discovered features';
COMMENT ON FUNCTION public.export_sae_for_neuronpedia IS 'Export SAE metadata in Neuronpedia-compatible format';
