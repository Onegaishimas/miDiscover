#!/usr/bin/env python3
"""
Fix sparsity display issues in the database
"""

from db_utils import get_db, TrainingRun

def fix_sparsity_values():
    """Fix sparsity values that are showing as percentages instead of decimals"""
    
    try:
        db = get_db()
        
        # Find all completed runs with unrealistic sparsity values
        problematic_runs = db.query(TrainingRun).filter(
            TrainingRun.status == 'completed',
            TrainingRun.sparsity_level > 100  # Anything over 100% is wrong
        ).all()
        
        print(f"Found {len(problematic_runs)} runs with incorrect sparsity values:")
        
        for run in problematic_runs:
            old_sparsity = run.sparsity_level
            
            # If the value is way too high, it's probably stored as percentage when it should be decimal
            if old_sparsity > 1000:
                # Convert from percentage to decimal (divide by 100)
                new_sparsity = old_sparsity / 100
            elif old_sparsity > 100:
                # If it's just over 100, it might be a calculation error
                # Cap it at 100%
                new_sparsity = 100.0
            else:
                new_sparsity = old_sparsity
            
            print(f"  Run {str(run.id)[:8]}: {old_sparsity:.1f}% → {new_sparsity:.1f}%")
            
            run.sparsity_level = new_sparsity
            db.add(run)
        
        db.commit()
        db.close()
        
        print(f"\n✅ Fixed sparsity values for {len(problematic_runs)} runs")
        
    except Exception as e:
        print(f"❌ Error fixing sparsity values: {e}")
        try:
            db.close()
        except:
            pass

def show_current_sparsity_values():
    """Show current sparsity values to verify they make sense"""
    
    try:
        db = get_db()
        
        recent_runs = db.query(TrainingRun).filter(
            TrainingRun.status == 'completed'
        ).order_by(TrainingRun.started_at.desc()).limit(10).all()
        
        print(f"\n📊 Current Sparsity Values:")
        print(f"{'ID':<8} {'Final Loss':<12} {'Sparsity':<10} {'Active/Total':<15} {'Duration':<8}")
        print("-" * 60)
        
        for run in recent_runs:
            sparsity_str = f"{run.sparsity_level:.1f}%" if run.sparsity_level else "N/A"
            loss_str = f"{run.final_loss:.6f}" if run.final_loss else "N/A"
            active_str = f"{run.active_features}/{run.total_features}" if run.active_features and run.total_features else "N/A"
            duration_str = f"{run.training_duration_seconds}s" if run.training_duration_seconds else "N/A"
            
            print(f"{str(run.id)[:8]:<8} {loss_str:<12} {sparsity_str:<10} {active_str:<15} {duration_str:<8}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error showing sparsity values: {e}")
        try:
            db.close()
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_current_sparsity_values()
    else:
        fix_sparsity_values()
        show_current_sparsity_values()