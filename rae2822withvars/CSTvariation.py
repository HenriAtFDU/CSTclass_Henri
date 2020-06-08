__author__ = 'Shuyue WANG'

# Description : Create a set of CST parameters from given airfoil and make variation
# the coordinates starts at the upper tail, passing through origin, and ends at lower tail.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import math

N1 = 0.5
N2 = 1.0


def LHSample( D,bounds,N):
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result

# def TRSample( D,bounds, subzones):
#     '''
#     :param D:参数个数
#     :param bounds:参数对应范围（list）
#     :param subzones:每个维度的层数
#     :return:样本数据
#     '''
#     N = D * subzones
#     result = np.empty([N, D])
#     temp = np.empty([N])
#     d = 1.0 / N

#     for i in range(D):

#         for j in range(N):
#             temp[j] = np.random.uniform(
#                 low=j * d, high=(j + 1) * d, size = 1)[0]

#         np.random.shuffle(temp)

#         for j in range(N):
#             result[j, i] = temp[j]

#     #对样本数据进行拉伸
#     b = np.array(bounds)
#     lower_bounds = b[:,0]
#     upper_bounds = b[:,1]
#     if np.any(lower_bounds > upper_bounds):
#         print('范围出错')
#         return None

#     #   sample * (upper_bound - lower_bound) + lower_bound
#     np.add(np.multiply(result,
#                        (upper_bounds - lower_bounds),
#                        out=result),
#            lower_bounds,
#            out=result)
#     return result



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
     
def gen_firstder(p, x,y):
    # more convenient in numpy.array form
    x_np = np.array(x)
    #Class Function
    ClassF = pow(x_np, N1) * pow(1-x_np , N2)
    x_np[0] = 0.0000001
    x_np[-1] = 1 - 0.0000001

    #Class Function: its derivative
    ClassF_der = N1 * pow(x_np, N1 - 1) * pow(1-x_np , N2) - pow(x_np , N1) * N2 * pow(1 - x_np, N2 - 1)

    #Shape Function
    K = lambda a, b :  math.factorial(b)/math.factorial(a)/math.factorial(b-a)
    n = len(p) - 1
    ShapeF = np.zeros(len(x_np))
    for i in range(n+1):
        ShapeF = ShapeF + pow(x_np, i)*pow(1-x_np, n-i) * K(i,n) * p[i]
    #Shape Function: its derivative
    ShapeF_der = np.zeros(len(x_np))
    for i in range(n+1):
        ShapeF_der = ShapeF_der + ( (-1) * pow(x_np, i) * (n-i) * pow(1- x_np, n-i-1) + pow(1-x_np, n-i) * i * pow(x_np, i-1) ) * K(i,n) * p[i]
    
    #summing with tail term
    return ClassF * ShapeF_der +  ClassF_der * ShapeF + y[-1]


def gen_secondder(p, x,y):
    # more convenient in numpy.array form
    x_np = np.array(x)

    x_np[0] = 0.0000001
    x_np[-1] = 1 - 0.00000001

    #Class Function
    ClassF = pow(x_np, N1) * pow(1-x_np , N2)
    #Class Function: its derivative
    ClassF_der = N1 * pow(x_np, N1 - 1) * pow(1-x_np , N2) - pow(x_np, N1) * N2 * pow(1-x_np, N2 - 1)
    #Class Function: its derivative's derivative
    ClassF_der_der = N1 * (N1 - 1) * pow(x_np, N1-2) * pow(1-x_np, N2) - N1 * pow(x_np, N1-1) * N2 * pow(1-x_np, N2-1) - N1 * pow(x_np, N1 - 1) * N2 * pow(1-x_np, N2-1) + pow(x_np, N1) * N2 * (N2 - 1) * pow(1-x_np, N2 - 2)

    #Shape Function
    K = lambda a, b :  math.factorial(b)/math.factorial(a)/math.factorial(b-a)
    n = len(p) - 1
    ShapeF = np.zeros(len(x_np))
    for i in range(n+1):
        ShapeF = ShapeF + pow(x_np, i)*pow(1-x_np, n-i) * K(i,n) * p[i]
    #Shape Function: its derivative
    ShapeF_der = np.zeros(len(x_np))
    for i in range(n+1):
        ShapeF_der = ShapeF_der + ( (-1) * pow(x_np, i) * (n-i) * pow(1- x_np, n-i-1) + pow(1-x_np, n-i) * i * pow(x_np, i-1) ) * K(i,n) * p[i]
    #Shape Function: its derivative's derivative
    ShapeF_der_der = np.zeros(len(x_np))
    for i in range(n+1):
        ShapeF_der_der = ShapeF_der_der + ( (-1)* pow(x_np, i-1) * (n-i) * pow(1-x_np, n-i-1) ) * K(i,n) * p[i]
        ShapeF_der_der = ShapeF_der_der + ( pow(x_np, i) * (n-i) * (n-i-1) * pow(1-x_np, n-i-2) ) * K(i,n) * p[i]
        ShapeF_der_der = ShapeF_der_der + ( (-1)*(n-i) * pow(1-x_np, n-i-1) * i * pow(x_np, i-1) ) * K(i,n) * p[i]
        ShapeF_der_der = ShapeF_der_der + ( pow(1-x_np, n-i) * i * (i-1) * pow(x_np, i-2) ) * K(i,n) * p[i]

    #summing with tail term
    return ClassF_der * ShapeF_der + ClassF * ShapeF_der_der + ClassF_der_der * ShapeF +  ClassF_der * ShapeF_der


def area(iks, igrak):
    area = 0.0
    for a in range(1,len(iks)):
        delta_iks = iks[a] - iks[a-1]
        area += delta_iks * igrak[a-1]
    return area


def curvature_check_up(cur):
    sign_change = 0
    neg_zone = 0 
    pos_zone = 0
    if (cur[1] < cur[0]):
        return 0
    else:
        status = 'positive'
        for i in range(1,len(cur)):
            if ( (cur[i] < cur[i-1]) and (status == 'positive')):
                sign_change += 1
                status = 'negative'
            elif ( (cur[i] > cur[i-1]) and (status == 'negative')):
                sign_change += 1
                status = 'positive'
        if (sign_change > 1):
            return 0
        else:
            return 1


def monotone_check_up(mon):
    sign_change = 0
    neg_zone = 0 
    pos_zone = 0
    if (mon[1] < mon[0]):
        return 0
    else:
        status = 'positive'
        for i in range(1,len(mon)):
            if ( (mon[i] < mon[i-1]) and (status == 'positive')):
                sign_change += 1
                status = 'negative'
            elif ( (mon[i] > mon[i-1]) and (status == 'negative')):
                sign_change += 1
                status = 'positive'
        if (sign_change > 1):
            return 0
        else:
            return 1

def main():
    foil = Collecter('rae2822nonKutta_points_adapted_added.txt')    
    foil.prep()
  
    para_up_init = [0.12870155, 0.12375994, 0.16686548, 0.11761302, 0.21652734, 0.0979251, 0.28768452, 0.11120358, 0.23844988, 0.17679849, 0.21278639]    #  UP PARAM 
    para_lo_init = [-0.12937721, -0.13069766, -0.16606765, -0.09145432, -0.24817176, -0.14230519,  -0.09567824, -0.136302,   -0.04409382, -0.03108559,  0.06441065] #  LO PARAM 

    
    xx_up = np.array(foil.dots_x_up, dtype = float)
    yy_up = np.array(foil.dots_y_up, dtype = float)
    xx_lo = np.array(foil.dots_x_lo, dtype = float)
    yy_lo = np.array(foil.dots_y_lo, dtype = float)    

    area_upper_init = area(xx_up, yy_up)
    print('the upper area (initial) %10.5f' %(area_upper_init))
    area_lower_init = (-1) *area(xx_lo, yy_lo)
    print('the lower area (initial) %10.5f' %(area_lower_init))
    area_sum_init = area_upper_init + area_lower_init
    print('the total area (initial) %10.5f' %(area_sum_init))
   

    fig1, ax1 = plt.subplots(figsize=(20,10),dpi=300)
    ax1.set_xlabel('x',fontsize=20)
    ax1.set_ylabel('y',fontsize='x-large', fontstyle='oblique')
    # fig1.yticks(fontsize=20)

    para_up = para_up_init
    para_lo = para_lo_init

    fit_up_der_init =gen_firstder(para_up, xx_up, yy_up)
    fit_lo_der_init =gen_firstder(para_lo, xx_lo, yy_lo)
    fit_up_der_der_init =gen_secondder(para_up, xx_up, yy_up)
    fit_lo_der_der_init =gen_secondder(para_lo, xx_lo, yy_lo)
    term = pow(fit_up_der_init , 2)
    fit_up_curvature_init = fit_up_der_der_init/ (1e-5 + pow(1 + term , 1.5))
    term = pow(fit_lo_der_init , 2)
    fit_lo_curvature_init = fit_lo_der_der_init/ (1e-5 + pow(1 + term , 1.5))
    fit_up_curvature_limit1 = fit_up_curvature_init + 10
    fit_up_curvature_limit2 = fit_up_curvature_init - 10
    fit_lo_curvature_limit1 = fit_lo_curvature_init + 10
    fit_lo_curvature_limit2 = fit_lo_curvature_init - 10


    ax1.plot(foil.dots_x_up, fit_up_curvature_init,'bo', label='upper surface: original')
    ax1.plot(foil.dots_x_lo, fit_lo_curvature_init,'g', label='lower surface: original')
    # ax1.plot(foil.dots_x_lo, fit_up_curvature_limit1,'r-', label='upper surface: original')
    # ax1.plot(foil.dots_x_lo, fit_up_curvature_limit2,'ro', label='upper surface: original')
    # ax1.plot(foil.dots_x_lo, fit_lo_curvature_limit1,'g-', label='lower surface: original')
    # ax1.plot(foil.dots_x_lo, fit_lo_curvature_limit2,'go', label='lower surface: original')

    a = curvature_check_up(fit_up_curvature_init)
    if (a == 1):
        print('init curvature normal!')
    else:
        print('init curvature mal!')

    a = monotone_check_up(yy_up)
    if (a == 1):
        print('init montone normal!')
    else:
        print('init monotone mal!')

    # LHS
    D = 4
    N = 500
    bounds = [[0.9,1.1],[0.9,1.1],[0.9,1.1],[0.9,1.1]]
    # bounds = [[0.9,1.1],[0.9,1.1],[0.9,1.1]]
    samples = LHSample(D,bounds,N)
    WXYZ = np.array(samples)
    W = WXYZ[:,0]
    X = WXYZ[:,1]
    Y = WXYZ[:,2]
    Z = WXYZ[:,3]

    fig2,ax2 = plt.subplots(figsize=(20,10),dpi=300)
    ax2.set_xlabel('x',fontsize=30,fontfamily = 'sans-serif',fontstyle='italic')
    ax2.set_ylabel('y',fontsize='x-large', fontstyle='oblique')
    ax2.plot(foil.dots_x_up, foil.dots_y_up,'r', label='upper surface: original')
    ax2.plot(foil.dots_x_lo, foil.dots_y_lo,'r', label='lower surface: original')


    small = 0
    big = 0
    qualified = 0
    badcur = 0
    badmon = 0
    for k in range(N):
        para_up = para_up_init
        para_up[0] = para_up_init[0] * W[k]
        para_up[1] = para_up_init[1] * X[k]
        para_up[2] = para_up_init[2] * Y[k]
        para_up[3] = para_up_init[3] * Z[k]
        para_lo = para_lo_init

        fit_up =gen(para_up, xx_up, yy_up)
        fit_lo =gen(para_lo, xx_lo, yy_lo)

        fit_up_der =gen_firstder(para_up, xx_up, yy_up)
        fit_lo_der =gen_firstder(para_lo, xx_lo, yy_lo)

        fit_up_der_der =gen_secondder(para_up, xx_up, yy_up)
        fit_lo_der_der =gen_secondder(para_lo, xx_lo, yy_lo)        

        term = pow(fit_up_der, 2)
        fit_up_curvature = fit_up_der_der/ (1e-5 + pow(1 + term , 1.5))
        # term = pow(fit_lo_der , 2)
        # fit_lo_curvature = fit_lo_der_der/ (1e-5 + pow(1 + term , 1.5))

        a_cur = curvature_check_up(fit_up_curvature)
        # if (a == 1):
        #     print('normal!')
        # else:
        #     print('mal!')
        a_mon = monotone_check_up(fit_up)



        area_upper = area(xx_up, fit_up)
        # print('the upper area: %10.5f' %(area_upper))
        area_lower = (-1) *area(xx_lo, fit_lo)
        # print('the lower area: %10.5f' %(area_lower))
        area_sum = area_upper + area_lower
        print('the total area: %10.5f' %(area_sum))


        if (area_sum < area_sum_init):
            small += 1
            print('too small sample!')
            ## direct compare of all samples
            ax2.plot(foil.dots_x_up, fit_up,'g:', label='upper surface too small:' )
            ax2.plot(foil.dots_x_lo, fit_lo,'g:', label='lower surface too small:' )

            # ax1.plot(foil.dots_x_up, fit_up_curvature,'g-', label='upper surface:  too small')
            # ax1.plot(foil.dots_x_lo, fit_lo_curvature,'g-', label='lower surface: too small')         

        elif (area_sum > 1.06 * area_sum_init):
            big += 1
            print('too big sample!')
            ## direct compare of all samples
            ax2.plot(foil.dots_x_up, fit_up,'g:', label='upper surface too big:' )
            ax2.plot(foil.dots_x_lo, fit_lo,'g:', label='lower surface too big:' )

            # ax1.plot(foil.dots_x_up, fit_up_curvature,'g-', label='upper surface:  too big')
            # ax1.plot(foil.dots_x_lo, fit_lo_curvature,'g-', label='lower surface: too big')          
               

        elif (a_mon == 0):
            badmon +=1
            print('bad monotone sample!')
            ## direct compare of qualified samples
            ax2.plot(foil.dots_x_up, fit_up,'y:', label='upper surface: bad monotone' )
            ax2.plot(foil.dots_x_lo, fit_lo,'y:', label='lower surface: bad monotone' )            

            ax1.plot(foil.dots_x_up, fit_up_curvature,'y-', label='upper surface: bad monotone')
            # ax1.plot(foil.dots_x_lo, fit_lo_curvature,'r-', label='lower surface: bad curvature')            

        elif (a_cur == 0):
            badcur +=1
            print('bad curvature sample!')
            ## direct compare of qualified samples
            ax2.plot(foil.dots_x_up, fit_up,'b:', label='upper surface: bad curvature' )
            ax2.plot(foil.dots_x_lo, fit_lo,'b:', label='lower surface: bad curvature' )            

            ax1.plot(foil.dots_x_up, fit_up_curvature,'b-', label='upper surface: bad curvature')
            # ax1.plot(foil.dots_x_lo, fit_lo_curvature,'r-', label='lower surface: bad curvature')
        
        else:
            qualified +=1
            print('qualified sample!')
            ## direct compare of qualified samples
            ax2.plot(foil.dots_x_up, fit_up,'r:', label='upper surface: qualified' + str(qualified))
            ax2.plot(foil.dots_x_lo, fit_lo,'r:', label='lower surface: qualified' + str(qualified))            

            ax1.plot(foil.dots_x_up, fit_up_curvature,'r-', label='upper surface: qualified')
            # ax1.plot(foil.dots_x_lo, fit_lo_curvature,'r-', label='lower surface: qualified')

            filename = 'rae2822_sample' + str(qualified) + '.txt'
            with open(filename,'w') as f: 
                for line in  range(1,len(fit_up)):
                    f.write(str(foil.dots_x_up[(-1)*line]) + ' ' +  str(fit_up[(-1)*line]) +    "\n")
                for line in  range(len(fit_lo)):
                    f.write(str(foil.dots_x_lo[line]) + ' ' +  str(fit_lo[line]) + "\n")

    print("too small:", small )
    print("too big:", big )
    print("badmon: ", badmon)
    print("badcur:", badcur)
    print("qualified: ", qualified)

    ax2.set_title('foil variation',fontsize=30)
    fig2.savefig('foil variation')    

    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-100, 15)
    ax1.set_title('foil curvature distribution',fontsize=30)
    fig1.savefig('foil curvature distribution')  


if __name__=='__main__':
    main()