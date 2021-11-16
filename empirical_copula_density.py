'''
Calculate and draw empirical copula density with two correlated time series
..moduleauthor:: Shanshan Wang
..Email: shanshan.wang@uni-due.de
..Date: Nov. 16
'''
# ----------------------------------------------------------------------------
# Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import csv

# ----------------------------------------------------------------------------
# functions

def sampling_two_corr_time_series(mean,cov,n):
    """
    e.g.,
    mean = (1, 2) # a list of the mean values of two time series
    cov = [[1, 1], [1, 2]] # 2-d array of diagonal covariances
    n is the length of each time series
    """
    val=np.random.multivariate_normal(mean,cov,n)
    x0=val[:,0].T
    y0=val[:,1].T
    x=x0.tolist()
    y=y0.tolist()
    z=pd.DataFrame(list(zip(x,y)),columns=['x','y'])
    return x, y, z

def draw_histograms_of_two_series(x,y,n_bins):
    # histograms of the data
    fig, axes=plt.subplots(1,2, figsize=(15,4))
    sns.histplot(data=x, bins=n_bins,ax=axes[0],stat='density').set(title='probability density distribution of x', xlabel='x',ylabel='pdf')
    sns.histplot(data=y, bins=n_bins,ax=axes[1],stat='density').set(title='probability density distribution of y', xlabel='y',ylabel='pdf')
    plt.savefig('hist_of_two_series.png',dpi=300, transparent=False, format='png', bbox_inches='tight')
    plt.close(fig)

def draw_joint_distribution_of_two_series(z):
    # joint distribution of x and y
    sns.jointplot(data=z, kind="scatter", x="x", y="y")
    
def qrank_data(x):
    # quantiles of ranks of variables x and y
    rx=ss.rankdata(x)
    qx=(rx-0.5)/len(x)
    return qx

def calc_emp_copula_density(qx,qy,nx,ny):
    # calculate empirical copula density
    xmin=0
    xmax=1
    ymin=0
    ymax=1
    cop_dens_emp=np.histogram2d(qx, qy, bins=(nx, ny), range=[[xmin, xmax], [ymin, ymax]],density=True)    # with density=True, normalize quantiles qx and qy
    return cop_dens_emp[0]

def draw_heatmap(matrix):
    # draw a two dimensional array in a heatmap
    fig=plt.figure(figsize=(8,6))
    nx=ny=len(matrix)
    xticklist=range(0,nx,2)
    xticklabels=[format(xt/nx,'.2f') for xt in xticklist]
    yticklist=range(0,ny,2)
    yticklabels=[format(yt/ny,'.2f') for yt in yticklist]
    sns.heatmap(matrix,cmap='jet').set(title='Empirical copula density', xlabel='Quantile(x)',ylabel='Quantile(y)', xticks=xticklist,yticks=yticklist, xticklabels=xticklabels, yticklabels=yticklabels);
    plt.savefig('heatmap_emp_cop_den.png',dpi=300, transparent=False, format='png', bbox_inches='tight')
    plt.close(fig)

def draw_surface(matrix):
    # draw the empirical copula density in a surface plot
    nx=ny=len(matrix)
    X, Y = np.meshgrid(range(nx), range(ny)) 
    fig = plt.figure(figsize=(12,8))
    ax = plt.axes(projection='3d')
    mycmap = plt.get_cmap('jet')
    surf1=ax.plot_surface(X, Y, matrix, cmap = mycmap)
    xticklist=range(0,nx,2)
    xticklabels=[format(xt/nx,'.2f') for xt in xticklist]
    yticklist=range(0,ny,2)
    yticklabels=[format(yt/ny,'.2f') for yt in yticklist]
    plt.xlabel('\n\n Quantile(x)')
    plt.ylabel('\n\n Quantile(y)')
    plt.xticks(xticklist,xticklabels,rotation=45)
    plt.yticks(yticklist,yticklabels,rotation=135)
    ax.set_zlabel('Empirical copula density')
    fig.colorbar(surf1, ax=ax, shrink=0.3, aspect=8)
    plt.savefig('surface_emp_cop_den.png',dpi=300, transparent=False, format='png', bbox_inches='tight')
    plt.close(fig)
    
def draw_bar3d(matrix):
    # draw copula density in 3-dimensional bar chart
    # Construct arrays for the anchor positions of the nx*ny bars.
    nx=ny=len(matrix)
    xpos, ypos = np.meshgrid(range(nx), range(ny))
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    # Construct arrays with the dimensions for the nx*ny bars.
    dx = dy = 1 * np.ones_like(zpos)
    dz = matrix.ravel()
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(projection='3d')
    colors = plt.cm.jet(matrix.flatten()/float(matrix.max()))
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=colors)
    xticklist=range(0,nx,2)
    xticklabels=[format(xt/nx,'.2f') for xt in xticklist]
    yticklist=range(0,ny,2)
    yticklabels=[format(yt/ny,'.2f') for yt in yticklist]
    plt.xlabel('\n\n Quantile(x)')
    plt.ylabel('\n\n Quantile(y)')
    plt.xticks(xticklist,xticklabels,rotation=45)
    plt.yticks(yticklist,yticklabels,rotation=135)
    ax.set_zlabel('Empirical copula density')
    plt.savefig('bar3d_emp_cop_den.png',dpi=300, transparent=False, format='png', bbox_inches='tight')
    plt.close(fig)

def main():
    """The main function of the script.
    The main function is used to test the functions in the script.
    :return: None.
    """
    # input parameters
    mean = (1, 2)
    cov = [[1, 1], [1, 2]] # diagonal covariance
    n=10000
    n_bins=50
    nx=30 # number of bins for qx
    ny=30 # number of bins for qy
    # calcualte and plot
    x,y,z=sampling_two_corr_time_series(mean,cov,n)
    draw_histograms_of_two_series(x,y,n_bins)
    draw_joint_distribution_of_two_series(z)
    qx=qrank_data(x)
    qy=qrank_data(y)
    emp_cop_den=calc_emp_copula_density(qx,qy,nx,ny)
    # draw empirical copula density in three ways
    draw_heatmap(emp_cop_den)
    draw_surface(emp_cop_den)
    draw_bar3d(emp_cop_den)
    # save data
    df_emp_cop_den= pd.DataFrame(emp_cop_den)
    df_emp_cop_den.to_csv('emp_cop_den.csv', index=False, header=False)
    
    return None

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
