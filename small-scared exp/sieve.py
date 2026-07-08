from fpylll import IntegerMatrix
from g6k import Siever
import numpy as np
from fpylll import LLL,BKZ
from g6k import SieverParams
from g6k.siever import Siever
from math import ceil,sqrt,log2
from multiprocessing import Pool
from os import cpu_count, path
import sys
from functools import partial, cache
import math
from g6k import SieverParams
#Construct a challenge instance
import re
import numpy as np

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
    return np.array(matrix)

def save_to_txt(data, filename, is_M_matrix=False):
   
    with open(filename, 'w') as f:
    
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


def pad_zero_left(v, target_dim):
   
    v_list = list(v)
    r = len(v_list)
   
    pad_len = target_dim - r
    if pad_len <= 0:
        return v_list  
    
    return [0] * pad_len + v_list


def change_basis(basis, location_back,vector):
	"""
	Puts a vector `vector` specified by coefficients into basis `basis`, i.e.  calculate
	`vector * basis`.
	:param basis: an IntegerMatrix containing the basis for a lattice.
	:param vector: a vector specifying a lattice vector by its coefficients (in terms of `basis`).
	:returns: the same lattice vector as `vector` but now expressed in the canonical basis.
	"""
	n = basis.nrows
	if len(vector)!= n and location_back is True:
		vector1 = pad_zero_left(vector, n)
	else:
		vector1 = vector
	#return vector
	return basis.multiply_left(vector1)


def norm(vector,q):
	s = 0
	for v in vector:
		vq = v%q
		if(abs(vq) <= abs(vq-q)):
			s += vq**2
		else:
			s+= (vq-q)**2
	return math.sqrt(s)

class Generic_Siever_Param:
	def __init__(self, N, beta_0, beta_1):
		self.N = N
		self.beta_0 = beta_0
		self.beta_1 = beta_1

class Progressive_Siever_Param:
	"""
	def __init__(self,_satrat,_satrad):
		self.satrat = _satrat
		self.satrad = _satrad
		self.l = 0
	"""
	def __init__(self,l,generic_siever_params,_satrat):
		N = generic_siever_params.N
		beta_0 = generic_siever_params.beta_0
		beta_1 = generic_siever_params.beta_1
		self.satrat = _satrat
		self.satrad = (2*N/_satrat)**(1/beta_1)
		self.l = l

class Additionnal_Siever_Param:
	def __init__(self,_v = False,_double_db = False,_threads = 1):
		self.v = _v
		self.double_db = _double_db
		self.threads = _threads

class Generic_Siever:
	def __init__(self, generic_siever_params, additional_params):
		self.generic_params = generic_siever_params
		self.add_params = additional_params
	
	def sieve(self,B, bkz=True):
		if(not (self.generic_params.beta_0 == -1)):
			B = BKZ.reduction(B,BKZ.Param(self.generic_params.beta_0))
		#B = BKZ.reduction(B,BKZ.Param(10))
		self.g6k_d = Siever(B, SieverParams(threads=self.add_params.threads, dual_mode=False))
		
		
		#B = BKZ.reduction(B,BKZ.Param(self.generic_params.beta_0))
		self.call(self.g6k_d)
	
		location_back = False
	
		with Pool(self.add_params.threads) as pool:
			#db1 = list(self.g6k_d.itervalues())
			#print(db1[0])
			self.short_vectors = pool.map(partial(change_basis,B,location_back), self.g6k_d.itervalues())

class Progressive_Siever(Generic_Siever):
	#Taken from Accurate Score Prediction
	def __init__(self, generic_siever_params, additional_params, progressive_siever_params):
		Generic_Siever.__init__(self, generic_siever_params, additional_params)
		self.specific_siever_params = progressive_siever_params
	def call(self,g6k):
		self.progressive_sieve(g6k,self.specific_siever_params.l, self.generic_params.beta_1 , saturation_ratio=self.specific_siever_params.satrat, saturation_radius=self.specific_siever_params.satrad, verbose=self.add_params.v,double_db=self.add_params.double_db)

	def progressive_sieve(self,g6k, l,  r, saturation_ratio, saturation_radius, verbose, double_db):
		"""
		Sieve in [l, r) progressively. The g6k object will contain a list of short vectors.
		Taking l > 0, will cause G6K to sieve in a projected sublattice of the full basis.

		:param g6k: Siever object used for sieving
		:param l: integer indicating number of basis vectors to skip at the beginning.
		:param r: integer indicating up to where to sieve.
		:param saturation_ratio: ratio to pass on to G6K (i.e. ratio of lattice vectors in the ball of
		radius sqrt(4/3) that should be in the database).
		:param verbose: boolean indicating whether or not to output progress of sieving.
		"""
		verbose = 20
	
		if verbose:
			print(f"Sieving [{max(l, r-40):3}, {r:3}]", end="", flush=True)
		g6k.initialize_local(l, max(l, r - 40), r)
		g6k(alg="gauss")
		while g6k.l > l:
			# Perform progressive sieving with the `extend_left` operation
			if verbose:
				print(f"\rSieving [{g6k.l:3}, {g6k.r:3}]...", end="", flush=True)
			g6k.extend_left()
			g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")
		with g6k.temp_params(saturation_ratio=saturation_ratio, db_size_factor=6):
			g6k(alg="hk3")
		
		num_dual_vectors = ceil((1 if double_db else 0.9)* saturation_ratio*0.5 * sqrt(4 / 3)**(r - l))
	
		if verbose:
			print("\rSieving is complete! ", flush=True)


		#py_list = [[g6k.M.B[i, j] for j in range(g6k.M.B.ncols)] for i in range(g6k.M.B.nrows)]
		#save_to_txt(np.array(py_list,dtype=int),"/home/zhangtong/BLASter/CODE/g6k_B.txt")
		# Number of dual vectors that is used in a full sieve is (4/3)^{n/2}.
		# Take into account 1/2 since G6K only saves exactly one of the vectors w or -w.
		if len(g6k) < num_dual_vectors:
			print(f"Not enough dual vectors found: {len(g6k)}, expected >= {num_dual_vectors}.")
		assert len(g6k) >= num_dual_vectors
		g6k.resize_db(num_dual_vectors)



def short_vectors(B,beta_0,beta_1,N,l=0,verbose = 20):
	# Options for siever
	double_db = True
	sat_ratio = 0.9
	threads = 8
	generic_param = Generic_Siever_Param(N,beta_0,beta_1)
	additionnal_param = Additionnal_Siever_Param(verbose,double_db,threads)

	# Instantiate siever
	progressive_siever_param = Progressive_Siever_Param(l,generic_param,sat_ratio)
	siever = Progressive_Siever(generic_param,additionnal_param,progressive_siever_param)
	if(verbose != 0):
		print("Begin sieve")
	siever.sieve(B)
	if(verbose != 0):
		print("End sieve")
	return siever.short_vectors


def main():
	file_path = '/home/zhangtong/BLASter/CODE/bkz_output.txt'
	#file_path = '/home/zhangtong/BLASter/CODE/A_fft.txt'
	np_array= np.array(read_matrix(file_path),dtype=int)

	n = np_array.shape[0]
	A = IntegerMatrix.from_matrix(np_array.tolist())
	A = LLL.reduction(A)
	L = short_vectors(A ,-1 ,85,0)

	save_to_txt(np.array(L,dtype=int) ,"/home/zhangtong/BLASter/CODE/L_sieving_dual.txt")
	#save_to_txt(np.array(L,dtype=int) ,"/home/zhangtong/BLASter/CODE/L_sieving1.txt")
	
if __name__ == '__main__':	
    main()