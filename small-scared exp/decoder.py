#import sage.all
from interface_polar import *
#from sage.all import Matrix, vector, GF, LinearCode
import galois
import numpy as np
import copy
import math
class Auxilary_Code:
	def __init__(self,_n,_k,_q):
		self.n = _n
		self.k = _k
		self.q = _q
	def decode(self,x):
		raise NotImplementedError()
	def encode(self,x):
		raise NotImplementedError()
	def unencode(self,x):
		raise NotImplementedError()
	def generator_matrix(self,x):
		raise NotImplementedError()

class Full_Code(Auxilary_Code):
	def __init__(self,_n,_k,_q):
		Auxilary_Code.__init__(self,_n,_k,_q)
	def decode(self,x):
		return [x]
	def dual_syndrome(self,x):
		return x
	def encode(self,x):
		return x
	def unencode(self,x):
		return x
	def generator_matrix(self,x):
		raise NotImplementedError()



def mod_matrix_inverse(mat, p):
    
    n = mat.shape[0]
   
    aug = np.hstack((mat, np.eye(n, dtype=np.int64)))
    
    for i in range(n):
      
        pivot = -1
        for j in range(i, n):
            if math.gcd(int(aug[j, i]), p) == 1:
                pivot = j
                break
        if pivot == -1:
            raise ValueError("not inversed")
        
        
        aug[[i, pivot]] = aug[[pivot, i]]
        
       
        inv_pivot = pow(int(aug[i, i]), -1, p)
        aug[i] = (aug[i] * inv_pivot) % p
        
       
        for j in range(n):
            if j != i and aug[j, i] != 0:
                factor = aug[j, i]
                aug[j] = (aug[j] - factor * aug[i]) % p
    
    
    inv_mat = aug[:, n:]
    return inv_mat.astype(np.int64)

class Polar_Code(Auxilary_Code):
    def __init__(self,_n_or_Polar,_k = None,_q = None,_Filter = False,_filter_dlsc = -1):
        if isinstance(_n_or_Polar, Polar_Code):
            o_polar = _n_or_Polar
            _n = o_polar.n
            _k = o_polar.k
            _q = o_polar.q
            Auxilary_Code.__init__(self,_n,_k,_q)
            self.Filter = o_polar.Filter
            self.filter_dlsc = o_polar.filter_dlsc
            
            self.C_polar_t = polar_copy(o_polar.C_polar_t)
            self.Mat_Gen = copy.deepcopy(o_polar.Mat_Gen)
            self.Mat_Gen_t = copy.deepcopy(self.Mat_Gen).transpose()
            self.mean_error = o_polar.mean_error 
            self.information = copy.deepcopy(o_polar.information)
        else:
            _n = _n_or_Polar
            Auxilary_Code.__init__(self,_n,_k,_q)
            self.Filter = _Filter
            self.filter_dlsc = _filter_dlsc
            print("Generate Polar code (Z_p ring)")
            while True:
                self.C_polar_t = polar_random(self.q,self.n,self.k)
                nb_fail = 0
                while True:
                    _Mat_Gen = []
                    for i in range(self.k):
                        
                        _Mat_Gen.append(list(polar_random_codeword(self.C_polar_t,self.n)))
                   
                    self.Mat_Gen = np.array(_Mat_Gen, dtype=np.int64) % self.q
                   
                    try:
                        mod_matrix_inverse(self.Mat_Gen, self.q)
                        break
                    except ValueError:
                        nb_fail += 1
                    if(nb_fail == 5):
                        break
                if(nb_fail < 5):
                    break
                else:
                    polar_free(self.C_polar_t)
            
            while True:
                I = [int(x) for x in np.random.choice(self.n,self.k,replace=False)]
                I.sort()
                _Mat_Gen_I = [[ M[i] for i in I] for M in _Mat_Gen]
                Mat_Gen_I = np.array(_Mat_Gen_I, dtype=np.int64) % self.q
                
                try:
                    Inv_Mat_Gen_I = mod_matrix_inverse(Mat_Gen_I, self.q)
                    break
                except ValueError:
                    continue
            
          
            self.Mat_Gen = np.matmul(Inv_Mat_Gen_I, self.Mat_Gen) % self.q
            
            self.mean_error = polar_mean_error(self.C_polar_t)
            if(self.Filter and self.filter_dlsc==-1):
                self.filter_dlsc = self.mean_error
            self.information = I
            self.Mat_Gen_t = self.Mat_Gen.transpose()
    
    def __del__(self):
        polar_free(self.C_polar_t)
    
    def decode(self,y):
        if(not self.Filter):
            return [tuple(polar_decode(self.C_polar_t,y))]
        else:
            d = tuple(polar_decode(self.C_polar_t,y))
            n = norm(minus(y,d,self.q),self.q)
            if(n <= self.filter_dlsc):
                return [d]
            else:
                return []
    
    def dual_syndrome(self,_x):
       
        x = np.array(_x, dtype=np.int64)
        result = np.dot(x, self.Mat_Gen_t) % self.q
        return tuple(result)
    
    def unencode(self,c):
        return tuple([c[i] for i in self.information])