#! /usr/bin/env python
# The MIT License (MIT)
#
# Copyright (c) 2015, EPFL Reconfigurable Robotics Laboratory,
#                     Philip Moseley, philip.moseley@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse, time, sys
import numpy as np
import curve_fit as CF
import utility as U
import model_utility as MU
from matplotlib import pyplot,axes
import scipy.interpolate as interp


#--------------------------------------------------------------------------------
# Create data plots.
#   ax    = two axes handles to plot on.
#   data  = dictionary of experimental datasets.
#   popt  = optimal parameters to calculate fit lines.
#   paperBOOL = it True, prepare an image suitable for a paper.
#--------------------------------------------------------------------------------
def create_data_plots(ax,data,popt,xlim=[],paperBOOL=False):
    NP = 2000
    fvals = np.empty([NP,2])
    M0 = popt[0]
    t0 = np.abs(popt[1])

    # Plot the experimental and fitted data.
    if not paperBOOL: ax[0].set_title('Shear Modulus vs Time')
    ax[0].plot(data[:,0]+t0,data[:,1],label='Experimental Data')
    if xlim==[]:
        x0,x1 = ax[0].get_xlim()
        xlim = [0.0, 1.5*x1]
    fvals[:,0] = np.linspace(xlim[0],xlim[1],NP)
    fvals[:,1] = prony(fvals[:,0]-t0,*popt) # This offsets the t0 in the prony function, giving us back the regular values.
    ax[0].plot(fvals[:,0],fvals[:,1],'r',label='Prony Fit')
    # Plot the dimensionless data and fit.
    # ax[1].plot(data[:,0],data[:,1]/M0,'k-',label='Dimensionless Experimental Data')
    # ax[1].plot(fvals[:,0],fvals[:,1]/M0,'r-',label='Dimensionless Fit')

    ax[0].set_adjustable('box-forced')
    ax[1].set_adjustable('box-forced')
    # Set the plot limits.
    ylim = [min(fvals[:,1]),max(fvals[:,1])]
    ylim[0] = ylim[0] - 0.1*np.diff(ylim)
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim[0]/ylim[1],1.0)
    # Set the aspect ratios to ensure the two plots line up.
    if not paperBOOL:
        ratio = 1.0
        ax[0].set_aspect(ratio*np.diff(xlim)/np.diff(ax[0].get_ylim()))
        ax[1].set_aspect(ratio*np.diff(xlim)/np.diff(ax[1].get_ylim()))
    # ax[0].set_xscale('log')
    ax[0].grid()
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(r'Shear Modulus $\left({}^N\!/{}_{mm^2}\right)$')
    ax[1].set_ylabel(r'Dimensionless Shear Modulus $\left({}^G\!/{}_{G_0}\right)$')
    if paperBOOL: ax[0].legend(loc='upper right',frameon=False,framealpha=0)
    pyplot.tight_layout()

#--------------------------------------------------------------------------------
# Create error plot.
#   ax    = axis handle to plot on.
#   data  = dictionary of experimental datasets.
#   popt  = optimal parameters to calculate fit lines.
#   R2    = overall R^2 fit error.
#--------------------------------------------------------------------------------
def create_error_plot(ax,data,popt,R2):
    M0 = popt[0]
    t0 = np.abs(popt[1])
    ax.set_title('Relative Errors ($R^2$='+str(R2)+')')
    if not len(data)==0:
        fvals = prony(data[:,0],*popt)
        err = 100.0 * (data[:,1]-fvals)/data[:,1]
        ax.plot(data[:,0],err,'.b',label='Errors')
        ax.grid()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (%)')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0])) # Square axis.

    # Calculate the equilibrium modulus.
    pterms = np.reshape(popt[2:],((len(popt)-2)/2,2))
    pdescr = str(popt[2:])+'\nG_inst = '+str(M0)+'\nG_equil = '+str(M0-np.sum(pterms[:,0]))+'\nt0 = '+str(np.abs(popt[1]))
    pyplot.figtext(0.5,0.02,pdescr,ha='center')
    pyplot.tight_layout()


#--------------------------------------------------------------------------------
# Prony function, M(t) = M_equil + sum(M_i e^(-t/tau_i))
# Abaqus implements as: M(t) = M0 - sum(M_i*(1-e^(-t/tau_i)))
#   t     = time series.
#   M0    = instantaneous modulus.
#   args  = mi and taui, number depends on number of terms.
#--------------------------------------------------------------------------------
def prony(t,M0,t0,*args):
    Mt = np.ones(t.shape) * M0
    t0 = np.abs(t0)
    for i in range(0,len(args),2):
        Mt = Mt - args[i]*(1.0-np.exp(-(t+t0)/args[i+1]))
    return Mt

#--------------------------------------------------------------------------------
# Calculate the parameters to fit a model to the experimental data.
#   data       = experimental dataset.
#   descr      = first part of filename.
#   terms      = number of terms to use in the Prony series.
#   poisson    = Poisson's ratio of material.
#   num_points = number of random points to try for optimization start.
#   ext        = image extension (with .).
#--------------------------------------------------------------------------------
def calc_params(data,descr,terms,poisson,num_points,ext):
    maxitr = 100    # Default = 100
    nparams = 2*terms+2
    print '---------------------------------------------------------'
    print ' Calculating',str(terms)+'-term fit with',num_points,'guess(es)'
    print '---------------------------------------------------------'
    S_best = -1.0e8
    iname = descr+'--'+str(terms)+'terms'+ext
    title = 'Prony Series '+str(terms)+'-term Fit'
    for i in range(num_points):
        p0 = np.longdouble(np.random.random_sample(nparams))
        p0[range(3,nparams,2)] = p0[range(3,nparams,2)] * 5.0
        print '\tStarting point '+str(i)
        try:
            if poisson==0.5:
                popt = CF.curve_fit1_basinhopping(prony,data[:,0],data[:,1],p0,maxitr)
            else:
                # TODO - how to do the bulk fitting? Completely seperately?
                U.print_error("Bulk fitting not yet supported, use nu=0.5",True)
                popt = CF.curve_fit2(prony,data[:,0],data[:,1],
                                     prony,data[:,0],data[:,2],
                                     p0,maxitr)
        except Exception:
            print '\t\tERROR:',sys.exc_info()[1]
            continue
        S = calculate_rsquared(data,poisson,popt)
        print '\t\tRsquared: ',S
        if S<=S_best: continue
        if (popt[range(2,nparams,2)]/popt[0] > 1.0).any():
            print '\t\t** Good result, but invalid for Abaqus. Continuing... **'
            continue
        S_best = S
        print '\t\tM0: ',popt[0]
        print '\t\t** New Best Result. Updating Plots **'
        # Plot results.
        ax = []
        fig = pyplot.figure(figsize=(10,6))
        ax.append(fig.add_subplot(121))
        ax.append(ax[0].twinx())
        ax.append(fig.add_subplot(122))
        create_data_plots(ax,data,popt)
        create_error_plot(ax[2],data,popt,S)
        pyplot.suptitle(title,fontweight="bold")
        pyplot.savefig(iname)
        pyplot.close()
        MU.write_viscomatfile(descr,popt[0],popt[2:])
        params = np.concatenate((popt,[S_best]))
    if S_best!=-1.0e8:
        print '\n\tBest-fit Rsquared:',params[-1]
    else:
        U.print_error("No suitable fit found.",False)
        return np.zeros(nparams+1)
    print '\n\n'
    return params


#--------------------------------------------------------------------------------
# Calculate Rsquared.
#--------------------------------------------------------------------------------
def calculate_rsquared(data,poisson,popt):
    if len(data)==0: return 0.0
    ydata = data[:,1]
    yfit  = prony(data[:,0],*popt)
    if poisson!=0.5:
        ydata = np.hstack((ydata,data[:,2]))
        yfit = np.hstack((yfit, prony(data[:,0],*popt)))
    return CF.rsquared(ydata,yfit)


#--------------------------------------------------------------------------------
# Main.
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    tinit = time.time()
    # Handle user input.
    parser = argparse.ArgumentParser(description="Fit the given dataset to a viscoelastic Prony series.",
                                     epilog="Example: calc_viscoelastic_parameters.py 0.5 newdata uni.txt")
    parser.add_argument("-p","--points",type=int,default=15,help="Number of random starting points (default 30).")
    parser.add_argument("--datapoints",type=int,default=500,help="The target number of datapoints to fit from each dataset (default 500).")
    parser.add_argument("--format",choices=['png','eps'],help="Image format, default is png.")
    parser.add_argument("--terms",nargs=2,type=int,default=[2,5],help="Range of terms to compare in Prony expansion, default is 3-9 (inclusive).")
    parser.add_argument("poisson",type=float,help="Poisson's ratio.")
    parser.add_argument("descr",help="Additional descriptive term to add to output file titles.")
    parser.add_argument("datafile",help="Dataset to fit, consisting of (time, shear mod, bulk mod) columns.")
    args = parser.parse_args()
    if args.format: fmt = '.'+args.format
    else:           fmt = '.png'
    # 'suppress' disables scientific notation for small numbers.
    np.set_printoptions(precision=4,linewidth=130,suppress=True)
    # np.seterr(all='raise')
    pyplot.rc('savefig',dpi=300)
    pyplot.rc('font',size=8)
    pyplot.rc('mathtext',default='regular') # Don't use italics for mathmode.

    # Read in the given datasets.
    print '--------------------------------------------------------------------'
    print ' Importing dataset...'
    data = np.loadtxt(args.datafile,comments='#',delimiter=',',dtype=np.longdouble)
    print '  Imported',data.shape[0],'datapoints.'

    # dataF = interp.UnivariateSpline(data[:,0], data[:,1], k=3)
    # dataS = np.zeros((args.datapoints,2))
    # dataS[:,0] = np.linspace(min(data[:,0]),max(data[:,0]),args.datapoints)
    # dataS[:,1] = dataF(dataS[:,0])
    # data = dataS
    # print '** Smoothed points, now',data.shape[0],'points. **'

    # Calculate optimal parameters for several lengths of Prony expansions.
    params = dict()
    for i in range(args.terms[0],args.terms[1]+1):
        params[i] = calc_params(data,args.descr,i,args.poisson,args.points,fmt)

    print '--------------------------------------------------------------------'
    print ' Results for Prony fits to Shear Modulus (G).'
    print '--------------------------------------------------------------------'
    np.set_printoptions(suppress=False)
    for t,p in params.iteritems():
        R2 = p[-1]
        M0 = p[0]
        t0 = p[1]
        pronyM = np.reshape(p[2:-1],(t,2))
        pronyM = pronyM[pronyM[:,1].argsort()]      # Sort the rows by time.
        pronyM_nd = np.zeros(pronyM.shape)
        pronyM_nd[:,0] = pronyM[:,0] / M0
        pronyM_nd[:,1] = pronyM[:,1]
        Me = M0-np.sum(pronyM[:,0])
        print '* Parameters for '+str(t)+'-term fit.'
        print 'Rsquared:',R2
        print "G_instantaneous:  {: 12f}\t\t\t(dimless {: 12f})".format(M0,M0/M0)
        print "G_equilibrium:    {: 12f}\t\t\t(dimless {: 12f})".format(Me,Me/M0)
        print "t_0 (seconds):    {: 12f}".format(t0)
        print 'Parameters (G_i, tau_i):\t\tDimensionless Parameters (g_i, tau_i):'
        for r in range(pronyM.shape[0]):
            print "  {: 12f},  {: 12f}\t\t\t{: 12f},  {: 12f}".format(pronyM[r,0],pronyM[r,1],pronyM_nd[r,0],pronyM_nd[r,1])
        print
    print 'TOTAL TIME ELAPSED: ',U.time_elapsed(tinit)
