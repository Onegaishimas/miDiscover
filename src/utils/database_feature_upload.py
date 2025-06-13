#!/usr/bin/env python3
"""
Direct database upload of SAE features to Neuronpedia
"""

import json
import psycopg2
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import uuid

def connect_to_database():
    """Connect to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="mechinterp_discovery",
            user="mechinterp",
            password="mechinterp_dev_password"
        )
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def insert_features_to_database(sae_dir: str, model_id: str = "microsoft/phi-2", source_id: str = "16-res-midiscovery"):
    """Insert SAE features directly into Neuronpedia database"""
    
    sae_path = Path(sae_dir)
    
    # Load feature analysis
    with open(sae_path / "feature_analysis.json", 'r') as f:
        analysis = json.load(f)
    
    # Load metadata
    with open(sae_path / "neuronpedia_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    conn = connect_to_database()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        print(f"🚀 Uploading {len(analysis['features'])} features to database...")
        print(f"   Model: {model_id}")
        print(f"   Source: {source_id}")
        
        # Insert features in batches
        batch_size = 1000
        features = analysis['features']
        
        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            
            print(f"   Inserting batch {i//batch_size + 1}/{(len(features) + batch_size - 1)//batch_size}")
            
            for feature in batch:
                feature_id = str(uuid.uuid4())
                feature_index = feature['feature_idx']
                
                # Insert into Neuron table (Neuronpedia's feature table)
                insert_query = """
                INSERT INTO "Neuron" (
                    id, "modelId", "sourceId", index, 
                    "maxActivation", "avgActivation", "activationFreq",
                    "createdAt", "updatedAt"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ("modelId", "sourceId", index) DO UPDATE SET
                    "maxActivation" = EXCLUDED."maxActivation",
                    "avgActivation" = EXCLUDED."avgActivation", 
                    "activationFreq" = EXCLUDED."activationFreq",
                    "updatedAt" = EXCLUDED."updatedAt";
                """
                
                cursor.execute(insert_query, (
                    feature_id,
                    model_id,
                    source_id,
                    feature_index,
                    float(feature['max_activation']),
                    float(feature['mean_activation']),
                    float(feature['activation_frequency']),
                    datetime.now(),
                    datetime.now()
                ))
            
            # Commit each batch
            conn.commit()
        
        print(f"✅ Successfully uploaded {len(features)} features!")
        
        # Verify the upload
        cursor.execute("""
            SELECT COUNT(*) FROM "Neuron" 
            WHERE "modelId" = %s AND "sourceId" = %s
        """, (model_id, source_id))
        
        count = cursor.fetchone()[0]
        print(f"📊 Verification: {count} features found in database")
        
        return True
        
    except Exception as e:
        print(f"❌ Error inserting features: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

def setup_model_and_source(model_id: str = "microsoft/phi-2", source_id: str = "16-res-midiscovery"):
    """Ensure model and source exist in database"""
    
    conn = connect_to_database()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Insert model
        print(f"🔧 Setting up model: {model_id}")
        cursor.execute("""
            INSERT INTO "Model" (id, "displayName", "huggingFaceId", "isPublic", "createdAt", "updatedAt") 
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (model_id, "Phi-2", model_id, True, datetime.now(), datetime.now()))
        
        # Insert source
        print(f"🔧 Setting up source: {source_id}")
        cursor.execute("""
            INSERT INTO "Source" (id, "modelId", name, description, "isPublic", "createdAt", "updatedAt") 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id, "modelId") DO NOTHING;
        """, (source_id, model_id, source_id, "miDiscovery SAE on layer 16 MLP FC2", True, datetime.now(), datetime.now()))
        
        conn.commit()
        print("✅ Model and source setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up model/source: {e}")
        return False
        
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Upload SAE features directly to Neuronpedia database")
    parser.add_argument("sae_dir", help="Directory containing SAE outputs")
    parser.add_argument("--model-id", default="microsoft/phi-2", help="Model ID")
    parser.add_argument("--source-id", default="16-res-midiscovery", help="Source ID")
    
    args = parser.parse_args()
    
    if not Path(args.sae_dir).exists():
        print(f"❌ Directory not found: {args.sae_dir}")
        return
    
    # Check required files
    required_files = ["feature_analysis.json", "neuronpedia_metadata.json"]
    sae_path = Path(args.sae_dir)
    missing_files = [f for f in required_files if not (sae_path / f).exists()]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return
    
    print("🎯 Direct Database Upload to Neuronpedia")
    print("=" * 50)
    
    # Setup model and source
    if not setup_model_and_source(args.model_id, args.source_id):
        print("❌ Failed to setup model and source")
        return
    
    # Upload features
    if insert_features_to_database(args.sae_dir, args.model_id, args.source_id):
        print("\n🎉 Upload completed successfully!")
        print(f"💡 Access your features at: http://localhost:3000/{args.model_id}")
    else:
        print("\n❌ Upload failed")

if __name__ == "__main__":
    main()