import pickle
import math

pkl_file_path = "/home/zhangtong/g6k/CodedDualAttack/OptimizeCodedDualAttack/optimized_withoutExperimentalPolar.pkl"

with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

def format_value(val):
    # 先转为浮点型，兼容高精度大数
    try:
        num = float(val)
    except (ValueError, TypeError):
        return val

    # 阈值：大于 1e6 就转为 log2，可自行调整
    threshold = 1e6
    if abs(num) > threshold:
        log_val = math.log2(num)
        return f"log2({num:.2e}) = {log_val:.2f}"
    else:
        # 普通数保留2位小数
        return f"{num:.2f}"

for scheme, nn_dict in data.items():
    print(f"\n【方案】{scheme}")
    for nn, params in nn_dict.items():
        print(f"  nn = {nn}")
        for k, v in params.items():
            print(f"    {k}: {format_value(v)}")