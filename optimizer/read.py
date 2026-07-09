import pickle
import math

pkl_file_path = "optimized_withoutExperimentalPolar.pkl"

with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

def format_value(val):
   
    try:
        num = float(val)
    except (ValueError, TypeError):
        return val

    threshold = 1e6
    if abs(num) > threshold:
        log_val = math.log2(num)
        return f"log2({num:.2e}) = {log_val:.2f}"
    else:
   
        return f"{num:.2f}"

for scheme, nn_dict in data.items():
    print(f"\n【scheme】{scheme}")
    for nn, params in nn_dict.items():
        print(f"  nn = {nn}")
        for k, v in params.items():
            print(f"    {k}: {format_value(v)}")
