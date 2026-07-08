import numpy as np
import scipy.stats as stats
from fpylll import IntegerMatrix, BKZ

# RLWE参数
n_lat = 60
n_enu = 2
n_fft = 8
n = n_lat + n_enu + n_fft


# 维度
q = 3329         # 模数
sigma = 2.0      # 高斯噪声标准差
q_half = q // 2  # 1664，用于最短表示转换



def mod_shortest(x, q):
    """将整数x转换为模q下的最短表示 (-q/2 <= x < q/2)"""
    return (x + q_half) % q - q_half

def generate_negacyclic_matrix(row, q):
    """从第一行生成负循环矩阵"""
    n = len(row)
    matrix = np.zeros((n, n), dtype=np.int32)
    
    # 第一行就是输入的row
    matrix[0] = row
    
    # 生成后续行：每一行是上一行右移一位并取负
    for i in range(1, n):
        # 右移一位
        matrix[i, 1:] = matrix[i-1, :-1]
        # 第一个元素是上一行最后一个元素的负值
        matrix[i, 0] = -matrix[i-1, -1]
        
    # 所有元素模q
    matrix = matrix 
    
    return matrix


def construct_special_matrix(n, q):
    

    first_row = np.random.randint(0, q, size=n, dtype=np.int32)
    
    # 从第一行生成负循环矩阵
    A = generate_negacyclic_matrix(first_row, q)
    A = np.vectorize(lambda x: mod_shortest(x, q))(A)
    # 创建全零的矩阵
    M = np.zeros((n + n_lat , n +n_lat), dtype=np.int32)
    
    # 右上角: n×n对角矩阵，对角线为q（转换为最短表示0）
   
    np.fill_diagonal(M[n:, n:], q)
  
    M[:n, n:] = A[:,:n_lat]
    
    np.fill_diagonal(M[:n, :n], 1)
   
    
    return M , A 


def save_to_txt(data, filename, is_M_matrix=False):
    """保存数组到TXT文件，每行前后添加方括号"""
    
    with open(filename, 'w') as f:
        # 处理向量（一维数组）
        if len(data.shape) == 1:
            line = '[' + ' '.join(map(str, data)) + ']'
            f.write(line + '\n')
        
        # 处理矩阵（二维数组）
        else:
            n = data.shape[0]
            rows, cols = data.shape
            for i in range(rows):
                row = data[i]
                line = '[' + ' '.join(map(str, row)) + ']'
                
                
                if is_M_matrix and i == n-1:
                    f.write(line + '\n')
                    line = ']]'
                
                f.write(line + '\n')
    
    print(f"已保存到 {filename}")


def process_file(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file}'")
        return
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        return

    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line.startswith('[') or not line.endswith(']'):
            print(f"警告: 忽略格式不正确的行: '{line}'")
            continue

        content = line[1:-1].strip()
        # 将多个连续空格替换为一个空格
        content = ' '.join(content.split())
        processed_line = f"[{content}]"
        processed_lines.append(processed_line)

    

    try:
        with open(output_file, 'w') as f:
            for line in processed_lines:
                f.write(line + '\n')
            f.write(']]')
        print(f"成功处理文件！结果已保存到 '{output_file}'")
    except Exception as e:
        print(f"错误: 写入文件时发生异常: {e}")



def main():
    target_number = 10000
    """生成嵌入格实例并构造特殊矩阵，保存为TXT文件"""
    print(f"生成嵌入格实例 (n={n}, q={q})...")
    

    input_file = '/home/zhangtong/BLASter/CODE/M.txt'    # 输入文件路径
    output_file = '/home/zhangtong/BLASter/CODE/basis.txt'  # 输出文件路径  

    M , A= construct_special_matrix(n, q)
    A_lat = A[:,:n_lat]
    A_fft = A[:,n-n_fft:]
    A_enu = A[:,n_lat:n-n_fft]
    save_to_txt(A_lat, '/home/zhangtong/BLASter/CODE/A_lat.txt')
    save_to_txt(A_fft, '/home/zhangtong/BLASter/CODE/A_fft.txt')
    save_to_txt(A_enu, '/home/zhangtong/BLASter/CODE/A_enu.txt')
    target = []
    secret = []
    error = []
    for i in range(target_number):
        secret_vec = stats.binom.rvs(n=6, p=0.5, size=n)-3
        error_vec = stats.binom.rvs(n=6, p=0.5, size=n)-3
        b = np.dot(A, secret_vec) + error_vec
        b = np.vectorize(lambda x: mod_shortest(x, q))(b)
        target.append(b)
        secret.append(secret_vec)
        error.append(error_vec)
    target,secret,error = np.array(target),np.array(secret),np.array(error)
    save_to_txt(target, '/home/zhangtong/BLASter/CODE/target.txt')
    save_to_txt(secret, '/home/zhangtong/BLASter/CODE/secret.txt')
    save_to_txt(error, '/home/zhangtong/BLASter/CODE/error.txt')

    save_to_txt(M, '/home/zhangtong/BLASter/CODE/M.txt')
    save_to_txt(A, '/home/zhangtong/BLASter/CODE/A.txt')

    process_file(input_file, output_file)  
    return M

if __name__ == "__main__":
    main()