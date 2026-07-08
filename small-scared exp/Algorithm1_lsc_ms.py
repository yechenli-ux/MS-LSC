import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.linalg import qr
from scipy.special import gamma
from multiprocessing import cpu_count, Pool
from math import ceil, comb, cos, erf, exp, pi, prod, sin, sqrt
from scipy.stats import norm
import time 
from fpylll import IntegerMatrix, BKZ
from functools import partial
from random import random, randint
import scipy
from scipy.integrate import quad
from utilitary import *
from sievers import *
from multiprocessing.pool import ThreadPool
from multiprocessing import Process, Pool,Pipe
from fpylll import IntegerMatrix
from scipy.stats import t ,moment
import scipy
import decoder

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]    

def mod_shortest(x, q):
    """将整数x转换为模q下的最短表示 (-q/2 <= x < q/2)"""
    return (x + q//2) % q - q//2
def np_to_integer_matrix(np_arr):
    np_arr = np.array(np_arr, dtype=int)  # 强制转 int
    rows, cols = np_arr.shape
    M = IntegerMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = int(np_arr[i, j])  # 必须转成 Python int
    return M

def decode_vectors(C_lsc, L_short_vectors, q):
	decoded_vectors = []
	L_norm_lsc_error = []
	e_lsc_vectors = []
	code_vectors = []
	for short_vector in L_short_vectors:
		for maux in C_lsc.decode(mod(short_vector,q)):
			#auxilary_code.check_is_codeword(maux)
			s = np.vectorize(lambda x: mod_shortest(x, q))(minus(maux, tuple(mod(short_vector, q)),q))
			L_norm_lsc_error.append(np.linalg.norm(s))
			e_lsc_vectors.append(s)        
			decoded_vectors.append(tuple(C_lsc.unencode(maux)) )
			code_vectors.append(tuple(maux)) 
	return decoded_vectors, L_norm_lsc_error,e_lsc_vectors, code_vectors

class CentredBinomial:
    """
    Sampler for an integer that is distributed by a binomial with
        (n, p) = (2*eta, 0.5),
    but centred to have an outcome in [-eta, eta].
    """
    def __init__(self, eta=3):
        self.eta = eta

    def support(self):
        """
        Give the interval [l, r] on which the PDF is nonzero.
        """
        return range(-self.eta, self.eta + 1)

    def PDF(self, outcome):
        return 0.25**(self.eta) * comb(2*self.eta, outcome + self.eta)

    def __call__(self):
        return sum(randint(0, 1) for i in range(2*self.eta)) - self.eta

def read_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            # 1. 清理行首尾空白（去掉换行、多余空格）
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 2. 移除当前行所有中括号（不管多少个，直接清干净）
            line_no_brackets = line.replace('[', '').replace(']', '')
            # 3. 按空格分割，过滤空字符（避免括号移除后产生的空值）
            row = [item for item in line_no_brackets.split() if item]
            if row:  # 只添加非空行（过滤最后一行纯括号的情况）
                matrix.append(row)
    first = matrix[0][0]
    dtype = np.int64 if first.isdigit() else np.float64
    return np.array(matrix, dtype=dtype)

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


def gram_schmidt_qr(matrix):
    """
    对整数矩阵进行QR分解，返回正交基(Q)和施密特系数矩阵(R)(以列为向量)
    
    参数:
        matrix: 整数矩阵（二维列表或numpy数组）
    
    返回:
        Q: 正交基矩阵（列向量正交）
        R: 上三角矩阵（施密特系数）
    """
    # 将输入转换为numpy数组（浮点数类型，便于计算）
    A = np.array(matrix, dtype=np.float64)
    
    # 执行QR分解（使用numpy的qr函数，mode='reduced'返回精简形式）
    Q, R = qr(A.T, mode='complete')
    diag_sign = np.sign(np.diag(R))
    # 处理对角元为 0 的情况（避免 0 的符号问题，此处设为 1）
    diag_sign[diag_sign == 0] = 1
    
    # 构造对角矩阵 D（符号矩阵）
    D = np.diag(diag_sign)
    
    # 调整 Q 和 R，使 R 的对角元为正数
    Q_pos = Q @ D  # 新的正交矩阵
    R_pos = D @ R  # 新的上三角矩阵（对角元为正数）
    
    # 输出结果
    return Q_pos, R_pos


def change_basis(basis, vector):
    """
    Puts a dual vector `vector` specified by coefficients into basis `basis`,
    i.e. calculate `vector * basis`.
    :param basis: an IntegerMatrix containing the basis for a lattice.
    :param vector: a vector specifying a lattice vector by its coefficients (in
                   terms of `basis`).
    :returns: the same lattice vector as `vector` but now expressed in the canonical basis.
    """
    return np.dot(vector,basis)

 

def calc_exp_expectation1(a, b, q, mu, sigma2):
    """
    计算无物理约束下 exp(-3π²/q² * ||w1+w2||²) 的期望
    :param a: float, w1的模长
    :param b: float, w2的模长
    :param q: float, 固定常数（如3329）
    :param mu: float, cosθ~N(μ,σ²)的均值
    :param sigma: float, cosθ~N(μ,σ²)的标准差
    :return: float, 期望结果
    """
    term1 = -3 * np.pi**2 * (a**2 + b**2) / q**2
    term2 = -6 * np.pi**2 * a * b * mu / q**2
    term3 = 18 * np.pi**4 * (a*b)**2 * sigma2 / q**4
    return np.exp(term1 - term2 + term3)

def cos_lambdax_expectation_tri(t):
    '计算cos(tx)在x是三角分布的情况下的均值'
    return (2*(1-cos(t))/t**2)
def cos_lambdax_expectation_uniform(t):
    '计算cos(tx)在x是三角分布的情况下的均值'
    return (2*sin(t/2)/t)
def tir_cos_expectation(p,n):
  'p为素数,n为相加的个数'
  term1 = 1
  term2 = cos_lambdax_expectation_tri(2*pi/p)
  term3 = cos_lambdax_expectation_tri(4*pi/p)
  term4 = cos_lambdax_expectation_tri(6*pi/p)
  sum = 20/64*term1+30/64*term2+12/64*term3+2/64*term4
  return sum**n

def uniform_cos_expectation(p,n):
  'p为素数,n为相加的个数'
  term1 = 1
  term2 = cos_lambdax_expectation_uniform(2*pi/p)
  term3 = cos_lambdax_expectation_uniform(4*pi/p)
  term4 = cos_lambdax_expectation_uniform(6*pi/p)
  sum = 20/64*term1+30/64*term2+12/64*term3+2/64*term4
  return sum**n

def compute_cosine(w1,w2):
  cosine = np.dot(w1,w2)/(np.linalg.norm(w1)*np.linalg.norm(w2))
  return cosine


if __name__ == '__main__':
  t1 = time.time()
  beta = 85 #筛法维数
  L = read_matrix('/home/zhangtong/BLASter/CODE/L_sieving_dual.txt')
  B = read_matrix('/home/zhangtong/BLASter/CODE/bkz_output_dual.txt')
  A = read_matrix('/home/zhangtong/BLASter/CODE/A.txt')
  A_fft = read_matrix('/home/zhangtong/BLASter/CODE/A_fft.txt')
  A_lat = read_matrix('/home/zhangtong/BLASter/CODE/A_lat.txt')
  A_enu = read_matrix('/home/zhangtong/BLASter/CODE/A_enu.txt')
  target = read_matrix('/home/zhangtong/BLASter/CODE/target.txt')
  secret= read_matrix('/home/zhangtong/BLASter/CODE/secret.txt')
  error = read_matrix('/home/zhangtong/BLASter/CODE/error.txt')
  C_lsc = read_matrix('/home/zhangtong/BLASter/CODE/C_lsc.txt')
  q = 3329
  p = 128 
  
  #A_fft, A_lat, A_enu = A_fft.T, A_lat.T, A_enu.T
  n,n_lat,n_fft,n_enu =A.shape[1],A_lat.shape[1],A_fft.shape[1],A_enu.shape[1]
  print(f'n={n},n_lat={n_lat},n_fft={n_fft},n_enu={n_enu}')
  k_fft = int(n_fft/2)
  threads = 16
  samples = 5000
  num = 3000
  num = len(L)
  if_decode = False
  #计算高斯启发式
  Q, R = gram_schmidt_qr(B)
  log2 = 0
  for i in range(beta):
      log2 += np.log2(R[i][i])
  log = log2/beta
  gh = np.sqrt(beta/(2*np.pi*np.e))*(2**(log))
  sigma = gh/np.sqrt(2*n)
  print(len(L))
  print(gh)
  print(np.linalg.norm(L[0]),np.linalg.norm(L[-1]))
  

  
  
# -------------------------- 1. 定义核心参数 --------------------------
  def compute_delta(beta):
     s1 = beta/(2*pi*exp(1))
     s2 = (pi*beta)**(1/beta)
     s3 = (s1*s2)**(1/(2*(beta-1)))
     return s3
  def compute_ell(beta_b,beta_s):
     l1 = sqrt(4/3)*compute_delta(beta_s)**(beta_s-1)*compute_delta(beta_b)**(n+n_lat-beta_s)*q**(n_lat/(n+n_lat))
     return l1
  
# -------------------------- 2. 计算yfft和yenum --------------------------
  inner_products = [] 
  secret_dist = CentredBinomial(3)
  with Pool(threads) as pool:
        print("(1/4) Computing y_fft...")
        # 7: Compute y_{j,fft} = x_j^T A_{fft} [Alg. 2, MATZOV].
        y_ffts = pool.map(partial(change_basis, A_fft), L[:,:n])
        
        print("(2/4) Computing y_enum...")
        # 8: Compute y_{j,enum} = x_j^T A_{enum} [Alg. 2, MATZOV].
        y_enums = pool.map(partial(change_basis, A_enu), L[:,:n])
  y_ffts = [vec.astype(np.int64) for vec in y_ffts]


# -------------------------- 3.计算ms对应的码字 --------------------------
  if if_decode:
      option_Clsc = 'change_with_A'
      if option_Clsc == 'change_with_A':
        C_lsc = decoder.Polar_Code(n_fft,k_fft,q)
      else:
        C_lsc = decoder.Polar_code(option_Clsc['code'])
      t2 = time.time()
      print(C_lsc.Mat_Gen_t)


      print(f"耗费{t2-t1}s")
      decoded_vectors, L_norm_lsc_error ,e_lsc_vectors , code_vectors = decode_vectors(C_lsc,y_ffts[:num], q)
   
      t2 = time.time()
      print(f"耗费{t2-t1}s")
      save_to_txt(np.array(decoded_vectors,dtype= int),'/home/zhangtong/BLASter/CODE/decoded_vectors_lsc.txt')
      save_to_txt(np.array(L_norm_lsc_error,dtype= float),'/home/zhangtong/BLASter/CODE/L_norm_lsc_error_lsc.txt')
      save_to_txt(np.array(e_lsc_vectors,dtype= int),'/home/zhangtong/BLASter/CODE/e_lsc_vectors_lsc.txt')
      save_to_txt(np.array(code_vectors,dtype= int),'/home/zhangtong/BLASter/CODE/code_vectors_lsc.txt')
  else:
      decoded_vectors = read_matrix('/home/zhangtong/BLASter/CODE/decoded_vectors_lsc.txt')
      L_norm_lsc_error = read_matrix('/home/zhangtong/BLASter/CODE/L_norm_lsc_error_lsc.txt')
      e_lsc_vectors = read_matrix('/home/zhangtong/BLASter/CODE/e_lsc_vectors_lsc.txt')
      code_vectors = read_matrix('/home/zhangtong/BLASter/CODE/code_vectors_lsc.txt')
  
        # ms_score = 0.0
  def work_uniform(_):
    secret = [secret_dist() for i in range(n)]
    error = [secret_dist() for i in range(n)]
    #target = np.add(np.dot(A,secret), error)
    target = [randint(0, q-1) for i in range(n)]
    table = np.zeros(shape=(p,) * n_fft, dtype=complex)
    random_secret =np.random.randint(0, q, size=n)
    for (j, x_j) in enumerate(L[:num,:n]):
      index = tuple(round(p * x / q) % p for x in y_ffts[j])
      inner_product = np.inner(x_j, target) - np.inner(y_enums[j], secret[n_lat:n-n_fft])
      #inner_product = np.inner(x_j, target) - np.inner(y_enums[j], secret[n_lat:n-n_fft])
      angle = inner_product * 2 * pi / (q)
      table[index] += cos(angle) + sin(angle) * 1.j
    # ms_score += cos(inner_product * 2 * pi / q - np.inner(index, secret[k_enum:k_enum+k_fft]) * 2 * pi / p)
    # Because the dual database is symmetric, the score is always real, so use cos().
    fft_output = np.fft.fftn(table).real
    
    ms_score = fft_output.max()
    #print(f'MS score = {ms_score}')
        # sfft_index = sum((secret[k_enum + i]%p) * p**(k_fft-1-i) for i in range(k_fft))
        # _wr_scores = fft_output.flatten()
        # print(_wr_scores[sfft_index], " vs ", ms_score)
        # assert _wr_scores[sfft_index] == ms_score
        # _wr_scores = np.delete(_wr_scores, sfft_index)
        # _wr_scores.sort()
        # return ms_score, _wr_scores, square_error

    return ms_score
  
  def work_correct(_):
    secret = [secret_dist() for i in range(n)]
    error = [secret_dist() for i in range(n)]
    target = np.add(np.dot(A,secret), error)
    table = np.zeros(shape=(p,) * k_fft, dtype=complex)
    for (j, x_j) in enumerate(L[:num,:n]):
      index = tuple(round(p * x / q) % p for x in decoded_vectors[j])
      inner_product = np.inner(x_j, target) - np.inner(y_enums[j], secret[n_lat:n-n_fft])
      angle = inner_product * 2 * pi / q
      table[index] += cos(angle) + sin(angle) * 1.j
    # ms_score += cos(inner_product * 2 * pi / q - np.inner(index, secret[k_enum:k_enum+k_fft]) * 2 * pi / p)
    # Because the dual database is symmetric, the score is always real, so use cos().
    fft_output = np.fft.fftn(table).real
    
    ms_score = fft_output.max()
    #print(f'MS score = {ms_score}')
        # sfft_index = sum((secret[k_enum + i]%p) * p**(k_fft-1-i) for i in range(k_fft))
        # _wr_scores = fft_output.flatten()
        # print(_wr_scores[sfft_index], " vs ", ms_score)
        # assert _wr_scores[sfft_index] == ms_score
        # _wr_scores = np.delete(_wr_scores, sfft_index)
        # _wr_scores.sort()
        # return ms_score, _wr_scores, square_error
    return ms_score



##-------------------------- 3. 计算随机目标的均值、方差、标准差、中位数 ----------
  '''  
  ms_scores, wr_scores, avg_sq_error = [], [], 0
  with Pool(threads) as pool:
      print("(3/4) Computing ms_scores...")
        # for (_ms_score, __wr_scores, _sq_err) in pool.imap_unordered(work, range(samples)):
      for (_ms_score) in pool.imap_unordered(work_uniform, range(samples)):
            ms_scores.append(_ms_score)
            # wr_scores.append(__wr_scores)
  ms_scores.sort()
  

 
  mean_ip = np.mean(ms_scores)  # 均值
  var_ip = np.var(ms_scores)    # 方差
  std_ip = np.std(ms_scores)    # 标准差
  median_ip = np.median(ms_scores)  # 中位数

  # wr_scores = np.concatenate(wr_scores)

#计算均值方差


  var = num/2
  mean = 0
  std = sqrt(var)
  mean1,var1= normal_max_high_precision(p**n_fft, 0, std)
  std1 = sqrt(var1)


  print("===== 随机目标分数分布的核心统计量 =====")
  print(f"均值：{mean_ip:.8f},{mean:.8f},{mean1:.8f}")
  print(f"方差：{var_ip:.8f},{var:.8f},{var1:.8f}")
  print(f"标准差：{std_ip:.8f},{std:.8f},{std1:.8f}")
  print(f"中位数：{median_ip:.2f}")
  
  '''

  # -------------------------- 4. 计算BDD目标的均值、方差、标准差、中位数 --------------------------
  l_lsc = np.mean(L_norm_lsc_error)
  cos1 = []
  for v in e_lsc_vectors[:num]:
    cosine = compute_cosine(e_lsc_vectors[0],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[1000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[10000],v)
    if cosine < 0.9:      
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[14000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[15000],v)
    if cosine < 0.9:
      cos1.append(cosine) 
    cosine = compute_cosine(e_lsc_vectors[5600],v)
    if cosine < 0.9:
      cos1.append(cosine) 
    cosine = compute_cosine(e_lsc_vectors[17000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[28000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[29000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[27000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[40000],v)
    if cosine < 0.9:
      cos1.append(cosine)
    cosine = compute_cosine(e_lsc_vectors[37000],v)
    if cosine < 0.9:
      cos1.append(cosine)
  print(f'余弦值的均值为{np.mean(cos1)},方差为{np.var(cos1)},最大值为{np.max(cos1)},估计为{1/k_fft}.')  
  r = 1.1*gh
  r = compute_ell(60,85)
  print(1.1*gh,r)
  mean1 = 0
  var1= 0
  for w in L[:num]:
    r =np.linalg.norm(w)
    mean1 += np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/q)**2))*uniform_cos_expectation(p,k_fft)
    var1 += 1/2 + 1/2*np.exp(-2*pi**2*((3/2)*((r/q)**2+(l_lsc/q)**2)*4))*uniform_cos_expectation(p/2,k_fft) - (np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/p)**2))*uniform_cos_expectation(p,k_fft))**2
  
  r = np.linalg.norm(L[-1])
  mean_per = np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/q)**2))*uniform_cos_expectation(p,k_fft)
  mean1 =mean_per*num
  var = var1 
  cov = calc_exp_expectation1(r,r,q,0,1/beta)*calc_exp_expectation1(l_lsc,l_lsc,q,0,np.var(cos1))*tir_cos_expectation(p,k_fft) - mean_per**2
  print(f'mean_per = {mean_per}, cov = {cov}')
  var1 += num*(num-1)*cov
  std1 = np.sqrt(var1)
  std =sqrt(var)
  print(f'mean = {mean1}, var = {var1}, std = {std1}')


  ms_scores1, wr_scores, avg_sq_error = [], [], 0
  '''
  with Pool(threads) as pool:
      print("(4/4) Computing ms_scores...")
        # for (_ms_score, __wr_scores, _sq_err) in pool.imap_unordered(work, range(samples)):
      for (_ms_score) in pool.imap_unordered(work_correct, range(samples)):
            ms_scores1.append(_ms_score)
            # wr_scores.append(__wr_scores)
  ms_scores1.sort()
  '''
  for j,b in enumerate(target):
    cosine = 0
    for i,w in enumerate(L[:num,:n]):
        inner_product = 2*pi*(np.dot(w, b)-np.dot(y_enums[i], secret[j,n_lat:n-n_fft]))/q
        # 取出fft分段密钥向量
        s_fft = secret[j, n - n_fft : n]  # shape (n_fft,)
        # 右乘G.T，完成线性编码映射：s_fft @ G.T
        s_fft_coded = s_fft @ C_lsc # shape (k_fft,)
        # 替换原内积项，用编码后的向量做内积
        vec_round = np.round(p * decoded_vectors[i] / q)
        inner_product -= np.dot(vec_round, s_fft_coded) * 2 * np.pi / p
        #inner_product -= np.dot(np.round(p*decoded_vectors[i]/q), secret[j,n-n_fft:])*2*pi/p
        #print(np.dot(np.round(p*y_ffts[i]/q), secret[j,n-n_fft:])*2*pi/p - np.dot(y_ffts[i], secret[j,n-n_fft:])*2*pi/q , np.dot((p*y_ffts[i]/q-np.round(p*y_ffts[i]/q)), secret[j,n-n_fft:])*2*pi/p )
        cosine += np.cos(inner_product)
    ms_scores1.append(cosine)
    # Sort the scores
  ms_scores1.sort()
  mean_ip1 = np.mean(ms_scores1)  # 均值
  var_ip1 = np.var(ms_scores1)    # 方差
  std_ip1 = np.std(ms_scores1)    # 标准差
  median_ip1 = np.median(ms_scores1)  # 中位数
  # wr_scores = np.concatenate(wr_scores)

  
  

#计算均值方差
  '''
  mean1 = 0
  var1= 0
  for w in L[:num]:
    r =np.linalg.norm(w)
    mean1 += np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/q)**2))*uniform_cos_expectation(p,k_fft)
    var1 += 1/2 + 1/2*np.exp(-2*pi**2*((3/2)*((r/q)**2+(l_lsc/q)**2)*4))*uniform_cos_expectation(p/2,k_fft) - (np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/q)**2))*uniform_cos_expectation(p,k_fft))**2
  
  r = np.linalg.norm(L[-1])
  
  mean_per = np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/q)**2))*uniform_cos_expectation(p,k_fft)
  var = var1 
  cov = calc_exp_expectation1(r,r,q,0,sqrt(1/beta))*calc_exp_expectation1(l_lsc,l_lsc,q,0,np.var(cos1))*tir_cos_expectation(p,k_fft) - mean_per**2
  print(mean_per,cov)
  var1 += num*(num-1)*cov
  std1 = np.sqrt(var1)
  std =sqrt(var)
  '''

  print("===== BDD目标分数分布的核心统计量 =====")
  print(f"均值：{mean_ip1:.8f},{mean_per:.8f},{mean1:.8f}")
  print(f"方差：{var_ip1:.8f},{var:.8f},{var1:.8f}")
  print(f"标准差：{std_ip1:.8f},{std:.8f},{std1:.8f}")
  print(f"中位数：{median_ip1:.2f}")
  t3 = time.time()
  print(f"耗费{t3-t1}s")