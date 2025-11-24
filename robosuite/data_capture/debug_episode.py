"""Quick debug script to check episode structure"""
import pickle

with open("data/episodes/episode_00000.pkl", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
print("Length:", len(data))

if isinstance(data, tuple):
    data_dict, attrs_dict = data
    print("\nData dict keys:", list(data_dict.keys())[:10])
    print("Attrs dict keys:", list(attrs_dict.keys())[:10])
    
    # Check for block keys
    block_keys = [k for k in data_dict.keys() if k.startswith('block_')]
    print(f"\nBlock keys: {block_keys}")
    
    # Show all keys
    print(f"\nAll data_dict keys: {list(data_dict.keys())}")
