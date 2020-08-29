#!/usr/bin/env python

"""

    PyR² - Calculates coefficients of determination
    Written in 2020 by Lucca M. A. Pellegrini <luccapellegrini@protonmail.com>

    To the extent possible under law, the author(s) have dedicated all
    copyright and related and neighboring rights to this software to the
    public domain worldwide. This software is distributed without any warranty.

    You should have received a copy of the CC0 Public Domain Dedication along
    with this software. If not, see
    <http://creativecommons.org/publicdomain/zero/1.0/>.

"""

from numpy import sum, mean
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt 








plt.style.use('default')  # coloquei o tema do plot como padrao 
rcParams['figure.figsize'] = 12, 6 #tamanho da figura

data = pd.read_csv('resistividade.txt', sep='\s+',header=None,comment='@')

data = pd.DataFrame(data)  # vai ler o arquivo usando panda no formato Data frame
x = data[0] # escolhi a primeira coluna como x
y = data[1] # segunda coluna como y







from pylab import rcParams

plt.style.use('default')

plt.rcParams['font.family'] = 'Palatino'
#plt.rcParams['font.serif'] = 'Sans'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#rc('font',**{'family':'serif','serif':['Sans']


SMALL_SIZE = 18  # 14 before
MEDIUM_SIZE = 24 # 18 BEFORE
BIGGER_SIZE =  24 #it was 18 before

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  #




plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{siunitx}')


rcParams['figure.figsize'] = 12, 6






def r_squared(x, y, popt, f):
    """Finds R²

    Parameters:
        x = array of x values
        y = array of y values
        popt = array of optimal values for parameters (first element
        returned by scipy.optimize.curve_fit. Check documentation)
        f = fitted function

    Returns:
        r2 = Coefficient of determination (R²)

    R² can be found using the mean (ȳ), the total sum of squares (SSₜₒₜ)
    & the residual sum of squares (SSᵣₑₛ). They're defined as:

             1   n
        ȳ = ---  Σ yᵢ
             n  i=1

        SSₜₒₜ = Σᵢ (yᵢ - ȳ)²

        SSᵣₑₛ = Σᵢ (yᵢ - fᵢ)²

                  SSᵣₑₛ
        R² = 1 - -------
                  SSₜₒₜ

    Sources:
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>

    See also:
    <https://docs.scipy.org/doc/>
    """
    res = y - f(x, *popt)
    ss_r = sum(res**2)
    ss_t = sum((y - mean(y))**2)

    r2 = 1 - (ss_r / ss_t)
    return r2

# - Here is an example - #
from numpy import sin, linspace, sqrt, diag
from pylab import *

# Suppose we have some experimental data in the form of x and y values:
# x = np.array([ 0.        ,  0.20408163,  0.40816327,  0.6122449 ,  0.81632653,
 #              1.02040816,  1.2244898 ,  1.42857143,  1.63265306,  1.83673469,
  #             2.04081633,  2.24489796,  2.44897959,  2.65306122,  2.85714286,
  #             3.06122449,  3.26530612,  3.46938776,  3.67346939,  3.87755102,
  #             4.08163265,  4.28571429,  4.48979592,  4.69387755,  4.89795918,
  #             5.10204082,  5.30612245,  5.51020408,  5.71428571,  5.91836735,
  #             6.12244898,  6.32653061,  6.53061224,  6.73469388,  6.93877551,
  #             7.14285714,  7.34693878,  7.55102041,  7.75510204,  7.95918367,
  #             8.16326531,  8.36734694,  8.57142857,  8.7755102 ,  8.97959184,
   #            9.18367347,  9.3877551 ,  9.59183673,  9.79591837, 10.        ])
#"""
#y = np.array([ -1.64845425,   7.19558714,   2.00102495,   6.06702304,
 #             -15.27756523,  22.08882723,  -3.63629021,   1.0622148 ,
  #             -3.70225705,  19.5679423 ,  27.48150934,  32.5276587 ,
   #             2.85825521,   9.35228703,  29.92399819,  27.64103531,
    #           30.78532271,  53.53330932,  30.77858214,  41.772589  ,
     #          56.12053544,  67.99483104,  59.46815296,  68.41795649,
      #         73.74473895,  81.55511625,  76.13443532,  90.19565609,
       #        93.38427864, 100.19719293, 126.47173659, 127.16205096,
        #      120.0345198 , 145.52675677, 148.45670638, 150.02927224,
         #     151.1687063 , 177.28836837, 182.04301953, 231.02125416,
          #    218.54344709, 206.16065363, 242.89689171, 230.98071537,
           #   246.22926094, 281.94785511, 284.41776556, 293.66792671,
            #  294.48106707, 315.83682544                            ])

# we can see them here as a scatter plot
#xlabel("x")
#ylabel("y")
#scatter(x, y, marker = '.')
#show()

# and we know they should obey a given function
# in this case, suppose it's y = f(x) = ax² + b + c sin(x)
def f(x, a, b, c):
    y =  a*x**2+ b + c*x 
    return y

# we'd like to know what a, b and c are.
# thankfully, we can use SciPy
from scipy import optimize

# the scipy.optimize module has a function curve_fit
# it takes your data and your function as parameters
# and returns two arrays, 'popt' and 'pcov'
# where 'popt' contains our values for a, b and c.
popt, pcov = optimize.curve_fit(f, x, y)
a, b, c = popt
sa, sb, sc = sqrt(diag(pcov))
print(f"a = {a}, b = {b}, c = {c}")
print(f"σa = {sa}, σb = {sb}, σc = {sc}")

# So we now have values for a, b and c, but how accurate are they?
# this is where R² comes in. It is the proportion of the variance in the
# dependent variable that is predictable from the independent variables.
# <https://en.wikipedia.org/wiki/Coefficient_of_determination>

# So, to find R², we use r_squared
R2 = r_squared(x, y, popt, f)
print(f"R² = {R2}")

# As we can see, R² is around 0.98788, which tells us that the values we
# got for a, b and c fit the data quite well

# We can demonstrate that by plotting the function we got and our data.
xs = linspace(0, 0.6, 512)
ys = f(xs, a, b, c)


ax = plt.subplot(111) #criei o subplot

ax.spines['top'].set_visible(False) 
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()


plt.plot(x, y, color="darkslategrey",marker='.',linestyle='',linewidth=2, markersize=10) #lw was 1.2 before

#\SI{1,09715247}{\micro\ohm}
plt.ylabel(r'Microstrain [$\mu \Si{\ohm}$]')
#xlabel("x")
#ylabel("y")
plot(xs, ys, color='maroon',linewidth=2,label="reta da regressão linear")
#scatter(x, y, marker = '.')
#plt.ylabel(r'Magnetização (NB)')
#plt.xlabel(r'Campo / Gauss ')
#plt.ylabel(' Resistência \, (ohm) ')
plt.ylabel(r' Resistência\,($ \Omega $)')
plt.xlabel(r' Distância\,(m) ')
plt.title(r"Resistência versus distância ao longo de um fio")
plt.legend(loc=4)
plt.savefig("rVl2.png", dpi=600, bbox_inches='tight')

show()
