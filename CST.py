__author__ = 'Shuyue WANG'

# Description : Create a set of CST parameters from given airfoil
# the coordinates starts at the upper tail, passing through origin, and ends at lower tail.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

N1 = 0.5
N2 = 1.0

class Collecter:

    def __init__(self, filename):
        self.filename =filename

    def data_get(self, filename, col):
        temp = [row.split(',')[col].strip('\n') for row in open(filename,'r',encoding='utf-8')]
        return [float(i) for i in temp[:]]

    def prep(self):
        # read data
        dots_x = self.data_get(self.filename, 0)
        dots_y = self.data_get(self.filename, 1)
        # locate the nose point 
        origin_x = min(dots_x)
        origin_pos = dots_x.index(min(dots_x))
        origin_y = dots_y[origin_pos]
        # determine length
        length = max(dots_x) - min(dots_x)
        # normalize
        ## translate
        dots_x = [float(i) - origin_x for i in dots_x[:]]
        dots_y = [float(i) - origin_y for i in dots_y[:]]
        ## scale
        dots_x = [float(i) / length for i in dots_x[:]]
        dots_y = [float(i) / length for i in dots_y[:]]
        # split upper and lower
        dots_x_up = dots_x[0:origin_pos+1]
        dots_y_up = dots_y[0:origin_pos+1]
        self.dots_x_lo = dots_x[origin_pos:]
        self.dots_y_lo = dots_y[origin_pos:]
        # make up-part from small to big:
        self.dots_x_up = dots_x_up[::-1]
        self.dots_y_up = dots_y_up[::-1]

def gen(p, x,y):
    # more convenient in numpy.array form
    x_np = np.array(x)
    #Class Function
    ClassF = pow(x_np, N1)
    ClassF = ClassF * pow(1-x_np , N2)
    #Shape Function
    K = lambda a, b :  math.factorial(b)/math.factorial(a)/math.factorial(b-a)
    n = len(p) - 1
    ShapeF = np.zeros(len(x_np))
    for i in range(n+1):
        ShapeF = ShapeF + pow(x_np, i)*pow(1-x_np, n-i) * K(i,n) * p[i]
    #summing with tail term
    return ClassF * ShapeF + x_np * y[-1]
        

def error(p,x,y):
    return gen(p,x,y) - y

def main():
    # foil = Collecter('NACA0024.txt')
    foil = Collecter('NLR-7223-62.txt')
    
    foil.prep()
    
    # deal with upper
    ##指定系数初始值，并转为leastsq要求的list
    p0_up = np.random.rand(6)*0.6 + 0.3
    p0_up.tolist()
    ##将翼型数据转为leastsq要求的array
    xx_up = np.array(foil.dots_x_up, dtype = float)
    yy_up = np.array(foil.dots_y_up, dtype = float)
    ##求解最佳系数
    para_up = leastsq(error, p0_up, args=(xx_up, yy_up))
    ##获得对应拟合结果
    fit_up =gen(para_up[0], xx_up, yy_up)
    ##比较得到差距
    difference_up = fit_up - yy_up
    ##获得各子翼型形态
    k = len(para_up[0])
    fit_up_sub = np.zeros(shape=(k,len(xx_up)))
    for j in range(k):
        temp = np.zeros(k)
        temp[j] = para_up[0][j]
        fit_up_sub[j] = gen(temp, xx_up, yy_up)

    #deal with lower
    ##指定系数初始值，并转为leastsq要求的list
    p0_lo = np.random.rand(6)*0.6 + 0.3
    p0_lo.tolist()
    ##将翼型数据转为leastsq要求的array
    xx_lo = np.array(foil.dots_x_lo, dtype = float)
    yy_lo = np.array(foil.dots_y_lo, dtype = float)
    ##求解最佳系数
    para_lo = leastsq(error, p0_lo, args=(xx_lo, yy_lo))
    ##获得对应拟合结果
    fit_lo =gen(para_lo[0], xx_lo, yy_lo)
    ##比较得到差距
    difference_lo = fit_lo - yy_lo
    ##获得各子翼型形态
    k = len(para_lo[0])
    fit_lo_sub = np.zeros(shape=(k,len(xx_lo)))
    for j in range(k):
        temp = np.zeros(k)
        temp[j] = para_lo[0][j]
        fit_lo_sub[j] = gen(temp, xx_lo, yy_lo)


    # make figures 
    ## original figure
    plt.figure(figsize=(20,10),dpi=300)
    plt.axis('off')
    plt.plot(foil.dots_x_up, foil.dots_y_up,'g')
    plt.plot(foil.dots_x_lo, foil.dots_y_lo,'g',)
    plt.savefig('foil NLR-7223-62') 

    ## direct compare
    plt.figure(figsize=(20,10),dpi=300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(foil.dots_x_up, foil.dots_y_up,'r', label='upper surface: original')
    plt.plot(foil.dots_x_lo, foil.dots_y_lo,'r', label='lower surface: original')
    plt.plot(foil.dots_x_up, fit_up,'g:', label='upper surface: fitted')
    plt.plot(foil.dots_x_lo, fit_lo,'g:', label='lower surface: fitted')
    plt.legend(fontsize=20)
    plt.title('NLR-7223-62 original and fitted surface: 6-order Bernstein polynomials',fontsize=30)
    plt.savefig('foil NLR-7223-62 double 6')    


    ## quantified compare
    plt.figure(figsize=(20,10),dpi=300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(foil.dots_x_up, difference_up,'ro', label='upper compared')
    plt.plot(foil.dots_x_lo, difference_lo,'bv', label='lower compared')

    plt.legend(fontsize=20)
    plt.title('NLR-7223-62 difference compared with original surface: 6-order Bernstein polynomials ',fontsize=30)
    plt.savefig('difference 6 Bernstein polynomial coefficients')   

    ## decomposition
    plt.figure(figsize=(16,22),dpi=300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for j in range(len(para_up[0])):
        plt.plot(foil.dots_x_up, fit_up_sub[j],'rx', label='upper surface component No.' + str(j))

    for j in range(len(para_lo[0])):
        plt.plot(foil.dots_x_lo, fit_lo_sub[j],'bo', label='lower surface decomposed No.' + str(j))
    
    plt.legend(fontsize=20)
    plt.title('NLR-7223-62 surface decomposition with 6-order Bernstein polynomials',fontsize=30)
    plt.savefig('decomposition 6')

if __name__=='__main__':
    main()