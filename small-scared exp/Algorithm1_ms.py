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
import scipy as sp
import decoder

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]    

def mod_shortest(x, q):
 
    return (x + q//2) % q - q//2

def np_to_integer_matrix(np_arr):
    np_arr = np.array(np_arr, dtype=int) 
    rows, cols = np_arr.shape
    M = IntegerMatrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            M[i, j] = int(np_arr[i, j])  
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
           
            line = line.strip()
            if not line:  
                continue
          
            line_no_brackets = line.replace('[', '').replace(']', '')
           
            row = [item for item in line_no_brackets.split() if item]
            if row:  
                matrix.append(row)
    first = matrix[0][0]
    dtype = np.int64 if first.isdigit() else np.float64
    return np.array(matrix, dtype=dtype)

def save_to_txt(data, filename, is_M_matrix=False):
   
    
    with open(filename, 'w') as f:
 
        if len(data.shape) == 1:
            line = '[' + ' '.join(map(str, data)) + ']'
            f.write(line + '\n')
        

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
    
    print(f"saved {filename}")


def gram_schmidt_qr(matrix):
  
  
    A = np.array(matrix, dtype=np.float64)
    
   
    Q, R = qr(A.T, mode='complete')
    diag_sign = np.sign(np.diag(R))
    
    diag_sign[diag_sign == 0] = 1
    

    D = np.diag(diag_sign)
  
    Q_pos = Q @ D 
    R_pos = D @ R  
    
    
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

    
    term1 = -3 * np.pi**2 * (a**2 + b**2) / q**2
    term2 = -6 * np.pi**2 * a * b * mu / q**2
    term3 = 18 * np.pi**4 * (a*b)**2 * sigma2 / q**4
    return np.exp(term1 - term2 + term3)

def cos_lambdax_expectation_tri(t):
    
    return (2*(1-cos(t))/t**2)
def cos_lambdax_expectation_uniform(t):
 
    return (2*sin(t/2)/t)
def tir_cos_expectation(p,n):

  term1 = 1
  term2 = cos_lambdax_expectation_tri(2*pi/p)
  term3 = cos_lambdax_expectation_tri(4*pi/p)
  term4 = cos_lambdax_expectation_tri(6*pi/p)
  sum = 20/64*term1+30/64*term2+12/64*term3+2/64*term4
  return sum**n

def uniform_cos_expectation(p,n):

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
  beta = 85 
  L = read_matrix('/home/zhangtong/BLASter/CODE/L_sieving_dual.txt')
  B = read_matrix('/home/zhangtong/BLASter/CODE/bkz_output_dual.txt')
  A = read_matrix('/home/zhangtong/BLASter/CODE/A.txt')
  A_fft = read_matrix('/home/zhangtong/BLASter/CODE/A_fft.txt')
  A_lat = read_matrix('/home/zhangtong/BLASter/CODE/A_lat.txt')
  A_enu = read_matrix('/home/zhangtong/BLASter/CODE/A_enu.txt')
  target = read_matrix('/home/zhangtong/BLASter/CODE/target.txt')
  secret= read_matrix('/home/zhangtong/BLASter/CODE/secret.txt')
  error = read_matrix('/home/zhangtong/BLASter/CODE/error.txt')
  q = 3329
  p = 128
  
  #A_fft, A_lat, A_enu = A_fft.T, A_lat.T, A_enu.T
  n,n_lat,n_fft,n_enu =A.shape[1],A_lat.shape[1],A_fft.shape[1],A_enu.shape[1]
  print(f'n={n},n_lat={n_lat},n_fft={n_fft},n_enu={n_enu}')
  k_fft = int(n_fft/2)-1
  threads = 16
  samples = 1000
  num = len(L)
  if_decode = True
  Q, R = gram_schmidt_qr(B)
  log2 = 0
  for i in range(beta):
      log2 += np.log2(R[i][i])
  log = log2/beta
  gh = np.sqrt(beta/(2*np.pi*np.e))*(2**(log))


  


  def compute_delta(beta):
     s1 = beta/(2*pi*exp(1))
     s2 = (pi*beta)**(1/beta)
     s3 = (s1*s2)**(1/(2*(beta-1)))
     return s3
  def compute_ell(beta_b,beta_s):
     l1 = sqrt(4/3)*compute_delta(beta_s)**(beta_s-1)*compute_delta(beta_b)**(n+n_lat-beta_s)*q**(n_lat/(n+n_lat))
     return l1
  
  inner_products = [] 
  secret_dist = CentredBinomial(3)
  with Pool(threads) as pool:
        print("(1/4) Computing y_fft...")
        # 7: Compute y_{j,fft} = x_j^T A_{fft} [Alg. 2, MATZOV].
        y_ffts = pool.map(partial(change_basis, A_fft), L[:,:n])
        
        print("(2/4) Computing y_enum...")
        # 8: Compute y_{j,enum} = x_j^T A_{enum} [Alg. 2, MATZOV].
        y_enums = pool.map(partial(change_basis, A_enu), L[:,:n])
  rounded_y_ffts = [np.round((p/q)*vec).astype(np.int64) for vec in y_ffts]

  print("(3/4) Computing Gu...")
  if if_decode:
      option_Clsc = 'change_with_A'
      if option_Clsc == 'change_with_A':
        C_lsc = decoder.Polar_Code(n_fft,k_fft,p)
      else:
        C_lsc = decoder.Polar_code(option_Clsc['code'])
      t2 = time.time()
      print(f"cost{t2-t1}s")
      decoded_vectors, L_norm_lsc_error ,e_lsc_vectors , code_vectors = decode_vectors(C_lsc, rounded_y_ffts[:num], p)
      t2 = time.time()
      print(f"cost{t2-t1}s")
      save_to_txt(np.array(decoded_vectors,dtype= int),'/home/zhangtong/BLASter/CODE/decoded_vectors_ms.txt')
      save_to_txt(np.array(L_norm_lsc_error,dtype= float),'/home/zhangtong/BLASter/CODE/L_norm_lsc_error_ms.txt')
      save_to_txt(np.array(e_lsc_vectors,dtype= int),'/home/zhangtong/BLASter/CODE/e_lsc_vectors_ms.txt')
      save_to_txt(np.array(code_vectors,dtype= int),'/home/zhangtong/BLASter/CODE/code_vectors_ms.txt')
  else:
      decoded_vectors = read_matrix('/home/zhangtong/BLASter/CODE/decoded_vectors_ms.txt')
      L_norm_lsc_error = read_matrix('/home/zhangtong/BLASter/CODE/L_norm_lsc_error_ms.txt')
      e_lsc_vectors = read_matrix('/home/zhangtong/BLASter/CODE/e_lsc_vectors_ms.txt')
      code_vectors = read_matrix('/home/zhangtong/BLASter/CODE/code_vectors_ms.txt')
    

    


  def work_uniform(_):
    secret = [secret_dist() for i in range(n)]
    target = [randint(0, q-1) for i in range(n)]
    table = np.zeros(shape=(p,) * k_fft, dtype=complex)
    for (j, x_j) in enumerate(L[:num,:n]):
      index = tuple(x % p for x in decoded_vectors[j])
      inner_product = np.inner(x_j, target) - np.inner(y_enums[j], secret[n_lat:n-n_fft])
      angle = inner_product * 2 * pi / q
      table[index] += cos(angle) + sin(angle) * 1.j
    fft_output = sp.fft.fftn(table).real

    return fft_output.flatten()

  
  def work_correct(_):
    secret = [secret_dist() for i in range(n)]
    error = [secret_dist() for i in range(n)]
    target = np.add(np.dot(A,secret), error)
    table = np.zeros(shape=(p,) * k_fft, dtype=complex)
    for (j, x_j) in enumerate(L[:num,:n]):
      index = tuple(x % p for x in decoded_vectors[j])
      inner_product = np.inner(x_j, target) - np.inner(y_enums[j], secret[n_lat:n-n_fft])
      angle = inner_product * 2 * pi / q
      table[index] += cos(angle) + sin(angle) * 1.j
    fft_output = sp.fft.fftn(table).real
    ms_score = fft_output.max()
    return ms_score


  l_lsc = np.mean(L_norm_lsc_error)
  r = 1.1*gh
  r = compute_ell(60,85)
  print(1.1*gh,r)
  mean1 = 0
  var1= 0
  for w in L[:num]:
    r =np.linalg.norm(w)
    mean1 += np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/p)**2))*uniform_cos_expectation(p,n_fft)
    var1 += 1/2 + 1/2*np.exp(-2*pi**2*((3/2)*((r/q)**2+(l_lsc/p)**2)*4))*uniform_cos_expectation(p/2,n_fft) - (np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/p)**2))*uniform_cos_expectation(p,n_fft))**2
  
  r = 1.1*gh
  mean_per = np.exp(-2*pi**2*(3/2)*((r/q)**2+(l_lsc/p)**2))*uniform_cos_expectation(p,n_fft)
  mean1 =mean_per*num
  var = var1 
  cov = calc_exp_expectation1(r,r,q,0,1/beta)*calc_exp_expectation1(l_lsc,l_lsc,p,0,1/n_fft)*tir_cos_expectation(p,n_fft) - mean_per**2
  var1 += num*(num-1)*cov
  std1 = np.sqrt(var1)
  std =sqrt(var)
 

  t3 = time.time()
  ms_scores1, wr_scores, avg_sq_error = [], [], 0
  with Pool(threads) as pool:
      print("(4/4) Computing ms_scores...")
        # for (_ms_score, __wr_scores, _sq_err) in pool.imap_unordered(work, range(samples)):
      for (_ms_score) in pool.imap_unordered(work_correct, range(samples)):
            ms_scores1.append(_ms_score)
            # wr_scores.append(__wr_scores)
  ms_scores1.sort()
  t4 = time.time()
  print(f"FFT cost{t4-t3}s")
  ms_scores1.sort()
  '''
  inner_products = []
  for i,b in enumerate(target):
        inner_product = 0
        for j,w in enumerate(L[:num]):
            inner = 1/q*(np.dot(L[j,:n],b) - np.dot(y_enums[j],secret[i,n_lat:n-n_fft]))
            #inner = 1/q*(np.dot(L[j,n:],secret[i,:n_lat])+ np.dot(L[j,:n],e))
            #inner += 1/q*(np.dot(y_fft[j],secret[i,n-n_fft:]))
            inner -= 1/p*((np.dot(code_vectors[j],secret[i,n-n_fft:])))
            #inner += 1/q*((np.dot(e_lsc_vectors[j],secret[i+1,n-n_fft:])))
            #print(1/q*(np.dot(y_fft[j],secret[i,n-n_fft:])))
            #inner_product = inner
            inner_product += np.cos(2*pi*inner)
            #random_vec = np.round(np.random.normal(0, sigma, size=vec_length)).astype(np.int32)
        inner_products.append(inner_product)
  ms_scores1 =  inner_products  
  '''  
  mean_ip1 = np.mean(ms_scores1)  
  var_ip1 = np.var(ms_scores1)    
  std_ip1 = np.std(ms_scores1)    
  median_ip1 = np.median(ms_scores1)  
  # wr_scores = np.concatenate(wr_scores)


  
  print(f"mean：{mean_ip1:.8f},{mean_per:.8f},{mean1:.8f}")
  print(f"var{var_ip1:.8f},{var:.8f},{var1:.8f}")
  print(f"std{std_ip1:.8f},{std:.8f},{std1:.8f}")
  print(f"medium{median_ip1:.2f}")
  t3 = time.time()
  print(f"cost{t3-t1}s")

 
