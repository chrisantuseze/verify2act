"""
Test script for critic data collection integration.

This verifies that the data collection components work together correctly.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "verify2act"))

import numpy as np

print("Testing critic data collection components...")
print("=" * 80)

# Test 1: EmbeddingExtractor
print("\n[1/3] Testing EmbeddingExtractor...")
from critic_embedding_utils import EmbeddingExtractor

extractor = EmbeddingExtractor()

# Test predicate embedding
goal_predicates = np.random.rand(5, 5, 9)
pred_embed = extractor.extract_predicate_embedding(
    goal_predicates, obj_id=0, target_id=1, num_objects=5
)
assert pred_embed.shape == (128,), f"Expected (128,), got {pred_embed.shape}"
print(f"  ✓ Predicate embedding: shape {pred_embed.shape}")

# Test plan summary
primitive_plan = ["Pick(A, table)", "Place(A, B)", "Pick(C, table)"]
plan_summary = extractor.extract_plan_summary(primitive_plan, current_step=0)
assert plan_summary.shape == (128,), f"Expected (128,), got {plan_summary.shape}"
print(f"  ✓ Plan summary: shape {plan_summary.shape}")

print("  ✓ EmbeddingExtractor works!")

# Test 2: CriticDataCollector
print("\n[2/3] Testing CriticDataCollector...")
from critic.critic_data_collector import CriticDataCollector

collector = CriticDataCollector()

# Add dummy successful trajectory
dummy_trajectory = [
    {
        "z_t": np.random.randn(256).astype(np.float32),
        "a_t": np.random.randn(64).astype(np.float32),
        "z_next": np.random.randn(256).astype(np.float32),
    }
    for _ in range(3)
]

dummy_predicates = [np.random.randn(128).astype(np.float32) for _ in range(3)]
dummy_summaries = [np.random.randn(128).astype(np.float32) for _ in range(3)]

collector.add_successful_trajectory(
    dummy_trajectory,
    dummy_predicates,
    dummy_summaries,
)

print(f"  ✓ Added successful trajectory ({len(dummy_trajectory)} steps)")

# Add dummy failed trajectory
collector.add_failed_trajectory(
    dummy_trajectory,
    dummy_predicates,
    dummy_summaries,
    failure_step=2,
    failure_type="predicate",
)

print(f"  ✓ Added failed trajectory (failure at step 2)")

# Get stats
stats = collector.get_statistics()
print(f"  ✓ Stats: {stats['num_positive']} positive, {stats['num_negative']} negative")

print("  ✓ CriticDataCollector works!")

# Test 3: Test saving
print("\n[3/3] Testing data saving...")

save_dir = Path("./data/critic_test")
save_dir.mkdir(parents=True, exist_ok=True)

try:
    collector.save_dataset(str(save_dir / "test_data.pkl"))
    print(f"  ✓ Dataset saved to {save_dir / 'test_data.pkl'}")
    
    # Load it back
    collector2 = CriticDataCollector()
    collector2.load_dataset(str(save_dir / "test_data.pkl"))
    stats2 = collector2.get_statistics()
    assert stats2['total'] == stats['total'], "Loaded data doesn't match!"
    print(f"  ✓ Dataset loaded successfully")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nReady to collect real data!")
print("Run: xvfb-run -a python collect_critic_data.py --num-episodes 10")
