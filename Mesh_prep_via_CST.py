__author__ = 'Shuyue WANG'

# Description : Create a set of CST parameters from given airfoil by given orders, and redistribute mesh points along curve uniformly.
# the second paragraph from the last can be used to ouput normalized dataset.
# the coordinates starts at the upper tail, passing through origin, and ends at lower tail.
# x- and  y- coordinates are separated by spaces & a comma & spaces


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

N1 = 0.5
N2 = 1.0
N_order_up = 11
N_order_lo = 11
N_mshpnt_up = 300
N_mshpnt_lo = 300


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

def gen_pntwise(p, x_p, y):
    #Class Function
    ClassP = pow(x_p, N1)
    ClassP *= pow(1-x_p , N2)
    #Shape Function
    K = lambda a, b :  math.factorial(b)/math.factorial(a)/math.factorial(b-a)
    n = len(p) - 1
    ShapeP = 0
    for i in range(n+1):
        ShapeP  += pow(x_p, i)*pow(1-x_p, n-i) * K(i,n) * p[i]

    return ClassP * ShapeP + x_p * y[-1]


def main():
    # foil = Collecter('NACA0024.txt')
    foil = Collecter('naca23012adapted.dat')
    # foil = Collecter('NLR-7223-62.txt')
    # foil = Collecter('base3.dat')
    # foil = Collecter('B29TIP_13adapted.txt')
    
    foil.prep()
    
    # deal with upper
    ##指定系数初始值，并转为leastsq要求的list
    p0_up = np.random.rand(N_order_up) 
    p0_up.tolist()
    ##将翼型数据转为leastsq要求的array
    xx_up = np.array(foil.dots_x_up, dtype = float)
    yy_up = np.array(foil.dots_y_up, dtype = float)


    ##求解最佳系数
    para_up = leastsq(error, p0_up, args=(xx_up, yy_up))
    ##获得对应拟合结果
    fit_up =gen(para_up[0], xx_up, yy_up)
    print('the  parameters for upper surface:')
    print(para_up[0])
    ##比较得到差距
    difference_up = fit_up - yy_up
    ##获得各子翼型形态
    k = len(para_up[0])
    fit_up_sub = np.zeros(shape=(k,len(xx_up)))
    for j in range(k):
        temp = np.zeros(k)
        temp[j] = para_up[0][j]
        fit_up_sub[j] = gen(temp, xx_up, yy_up)

    #Length of accumulated curves
    length = sum([math.sqrt((xx_up[i] - xx_up[i-1]) * (xx_up[i] - xx_up[i-1]) + (fit_up[i] - fit_up[i-1]) * (fit_up[i] - fit_up[i-1]) )  for  i  in range(1,len(xx_up))])
    print('uplength:', length)
    N_up_new_tot = N_mshpnt_up
    length_unit = 1.0 * length / N_up_new_tot
    x_new_delta = 0.000001
    x_up_new = [0] * (N_up_new_tot)
    fit_up_morepnts = [0] * (N_up_new_tot)
    for i in range(1,N_up_new_tot):
        temp = 0
        x_up_new[i] = x_up_new[i-1]
        while temp < length_unit:
            x_up_new[i] += x_new_delta
            fit_up_morepnts[i] = gen_pntwise(para_up[0], x_up_new[i], yy_up)
            temp =  math.sqrt( pow((x_up_new[i] - x_up_new[i-1]),2) + pow((fit_up_morepnts[i] - fit_up_morepnts[i-1]),2))
        # print('x:', x_up_new[i])
        # print('y:', fit_up_morepnts[i])
    
    x_up_new += [xx_up[-1]]
    fit_up_morepnts += [yy_up[-1]]
    # print(x_up_new)
    # print(fit_up_morepnts)


    #deal with lower
    ##指定系数初始值，并转为leastsq要求的list
    p0_lo = np.random.rand(N_order_lo) 
    p0_lo.tolist()
    ##将翼型数据转为leastsq要求的array
    xx_lo = np.array(foil.dots_x_lo, dtype = float)
    yy_lo = np.array(foil.dots_y_lo, dtype = float)
    ##求解最佳系数
    para_lo = leastsq(error, p0_lo, args=(xx_lo, yy_lo))
    ##获得对应拟合结果
    fit_lo =gen(para_lo[0], xx_lo, yy_lo)
    print('the  parameters for lower surface:')
    print(para_lo[0])
    ##比较得到差距
    difference_lo = fit_lo - yy_lo
    ##获得各子翼型形态
    k = len(para_lo[0])
    fit_lo_sub = np.zeros(shape=(k,len(xx_lo)))
    for j in range(k):
        temp = np.zeros(k)
        temp[j] = para_lo[0][j]
        fit_lo_sub[j] = gen(temp, xx_lo, yy_lo)
    
    
    #Length of accumulated curves
    length = sum([math.sqrt((xx_lo[i] - xx_lo[i-1]) * (xx_lo[i] - xx_lo[i-1]) + (fit_lo[i] - fit_lo[i-1]) * (fit_lo[i] - fit_lo[i-1]) )  for  i  in range(1,len(xx_lo))])
    print('lolength:', length)
    N_lo_new_tot = N_mshpnt_lo
    length_unit = 1.0 * length / N_lo_new_tot
    x_new_delta = 0.000001
    x_lo_new = [0] * (N_lo_new_tot)
    fit_lo_morepnts = [0] * (N_lo_new_tot)
    for i in range(1,N_lo_new_tot):
        temp = 0
        x_lo_new[i] = x_lo_new[i-1]
        while temp < length_unit:
            x_lo_new[i] += x_new_delta
            fit_lo_morepnts[i] = gen_pntwise(para_lo[0], x_lo_new[i], yy_lo)
            temp =  math.sqrt( pow((x_lo_new[i] - x_lo_new[i-1]),2) + pow((fit_lo_morepnts[i] - fit_lo_morepnts[i-1]),2))
        # print('x:', x_lo_new[i])
        # print('y:', fit_lo_morepnts[i])
    
    x_lo_new += [xx_lo[-1]]
    fit_lo_morepnts += [yy_lo[-1]]
    # print(x_lo_new)
    # print(fit_lo_morepnts)
        
    


    # make figures 
    ## original figure
    plt.figure(figsize=(20,10),dpi=300)
    plt.axis('off')
    plt.plot(foil.dots_x_up, foil.dots_y_up,'g')
    plt.plot(foil.dots_x_lo, foil.dots_y_lo,'g',)
    plt.savefig('foil ') 

    ## direct compare
    plt.figure(figsize=(20,10),dpi=300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(foil.dots_x_up, foil.dots_y_up,'r', label='upper surface: original')
    plt.plot(foil.dots_x_lo, foil.dots_y_lo,'r', label='lower surface: original')
    plt.plot(foil.dots_x_up, fit_up,'g:', label='upper surface: fitted')
    plt.plot(foil.dots_x_lo, fit_lo,'g:', label='lower surface: fitted')
    plt.plot(x_up_new, fit_up_morepnts,'bo', label='upper surface: fitted with more points')
    plt.plot(x_lo_new, fit_lo_morepnts,'bo', label='lower surface: fitted with more points')
    plt.legend(fontsize=20)
    plt.title('original and fitted surface and fitted with more points',fontsize=30)
    plt.savefig('foil double')    


    ## quantified compare
    plt.figure(figsize=(20,10),dpi=300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(foil.dots_x_up, difference_up,'ro', label='upper compared')
    plt.plot(foil.dots_x_lo, difference_lo,'bv', label='lower compared')

    plt.legend(fontsize=20)
    plt.title('difference compared with original surface',fontsize=30)
    plt.savefig('difference')   

    ## decomposition
    plt.figure(figsize=(16,22),dpi=300)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for j in range(len(para_up[0])):
        plt.plot(foil.dots_x_up, fit_up_sub[j],'rx', label='upper surface component No.' + str(j))

    for j in range(len(para_lo[0])):
        plt.plot(foil.dots_x_lo, fit_lo_sub[j],'bo', label='lower surface decomposed No.' + str(j))
    
    plt.legend(fontsize=20)
    plt.title('surface decomposition',fontsize=30)
    plt.savefig('decomposition')


    # # 将数据进行adapted形式的保存（实现从0到1）
    # filename_normalized = 'airfoil_points_adapted.txt'
    # with open(filename_normalized,'w') as f: 
    #     xx_up = xx_up[1:]
    #     xx_up = xx_up[::-1]
    #     yy_up = yy_up[1:]
    #     yy_up = yy_up[::-1]        
    #     for line in  range(len(xx_up)):
    #         f.write(str(round(xx_up[line],6)) + ',' +  str(round(yy_up[line],6)) + "\n")
    #     for line in  range(len(xx_lo)):
    #         f.write(str(round(xx_lo[line],6)) + ',' +  str(round(yy_lo[line],6)) + "\n")


    filename_msh = 'airfoil_points_adapted_added.txt'
    with open(filename_msh,'w') as f: 
        x_up_new = x_up_new[1:]
        x_up_new = x_up_new[::-1]
        fit_up_morepnts = fit_up_morepnts[1:]
        fit_up_morepnts = fit_up_morepnts[::-1]
        for line in  range(len(fit_up_morepnts)):
            f.write(str(round(x_up_new[line],6)) + '  ' +  str(round(fit_up_morepnts[line],6)) + "\n")
        for line in  range(len(fit_lo_morepnts)):
            f.write(str(round(x_lo_new[line],6)) + '  ' +  str(round(fit_lo_morepnts[line],6)) + "\n")

if __name__=='__main__':
    main()