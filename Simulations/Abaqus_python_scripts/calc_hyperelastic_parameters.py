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

import argparse, time, csv, sys, inspect
import scipy.optimize as opt
import numpy as np
import curve_fit as CF
import utility as U
import model_utility as MU
from matplotlib import pyplot,axes
from models import *


#--------------------------------------------------------------------------------
# Create plots.
#   model = module defining the model to fit.
#   data  = dictionary of experimental datasets.
#   popt  = optimal parameters to calculate fit lines.
#   R2    = overall R^2 fit error.
#   D     = volumetric parameters.
#   iname = image name.
#   title = image title.
#   xlim  = x-axis limits (strain).
#   ylim  = y-axis limits (stress).
#   ylimerr = y-axis limits for error plot (percent).
#   paperBOOL = if True, prepare an image for the paper.
#--------------------------------------------------------------------------------
def create_plots(model,data,popt,R2,D,iname,title,xlim=[],ylim=[],ylimerr=[],paperBOOL=False):
    if not paperBOOL:
        fig,ax = pyplot.subplots(1,2,figsize=(10,6))
        ax[0].set_title('Stress-Strain Curves')
        plot_data(ax[0],model,data,popt,xlim,ylim)
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()
        ax[0].set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0])) # Square axis.

        ax[1].set_title('Relative Errors ($R^2$='+str(R2)+')')
        if not len(data)==0:
            plot_err(ax[1],model,data,popt)
            xlim = ax[1].get_xlim()
            ylim = ax[1].get_ylim()
            ax[1].set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0])) # Square axis.
        else:
            ax[1].set_aspect(1)
        pyplot.suptitle(title,fontweight="bold")
        pyplot.figtext(0.5,0.02,model.params()+'\n'+str(popt)+'\nD: '+str(D),ha='center')
    else:
        fig,ax = pyplot.subplots(1,1)
        plot_data(ax,model,data,popt,xlim,ylim)
    pyplot.tight_layout()
    pyplot.savefig(iname)
    pyplot.close()


#--------------------------------------------------------------------------------
# Plot a dataset.
#   axis  = pyplot figure axis to plot on.
#   model = module defining the model to fit.
#   data  = dictionary of experimental datasets.
#   popt  = optimal parameters to calculate fit lines.
#--------------------------------------------------------------------------------
def plot_data(axis,model,data,popt=[],xlim=[],ylim=[]):
    # Plot the datasets.
    if 'u' in data:
        axis.plot(data['u'][:,0],data['u'][:,1],color='aqua',label='Uniaxial Data')
    if 'b' in data:
        axis.plot(data['b'][:,0],data['b'][:,1],color='lime',label='Biaxial Data')
    if 'p' in data:
        axis.plot(data['p'][:,0],data['p'][:,1],color='red',label='Planar Data')
    if 'v' in data:
        axis.plot(data['v'][:,0],data['v'][:,1],color='fuchsia',label='Volumetric Data')

    # Find the limits.
    if len(xlim)==0 or xlim[0]==xlim[1]:
        xlim = list(axis.get_xlim())
        xlim[1] = 1.2*xlim[1]
    if len(ylim)==0 or ylim[0]==ylim[1]:
        ylim = list(axis.get_ylim())
        ylim[1] = 1.2*ylim[1]

    if len(popt)!=0:
        # Calculate the fitted lines.
        NP = 2000
        keys = ['u','b','p']
        fits = dict()
        for i,key in enumerate(keys):
            fvals = np.empty([NP,2])
            fvals[:,0] = np.linspace(xlim[0],xlim[1],NP)
            fvals[:,1] = MU.get_function(model,key)(fvals[:,0],*popt)  # splat operator.
            fits[key] = fvals
        # Plot the fited lines.
        axis.plot(fits['u'][:,0],fits['u'][:,1],'--',color='blue',label='Uniaxial Fit')
        axis.plot(fits['b'][:,0],fits['b'][:,1],'--',color='green',label='Biaxial Fit')
        axis.plot(fits['p'][:,0],fits['p'][:,1],'--',color='maroon',label='Planar Fit')
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
    axis.grid()
    axis.set_xlabel('Engineering Strain')
    axis.set_ylabel(r'Nominal Stress $\left({}^N\!/{}_{mm^2}\right)$')
    L = axis.legend(loc='best',frameon=False,framealpha=0)
    for legobj in L.legendHandles: legobj.set_linewidth(3.0)


#--------------------------------------------------------------------------------
# Plot errors.
#   axis  = pyplot figure axis to plot on.
#   model = module defining the model to fit.
#   data  = dictionary of experimental datasets.
#   popt  = optimal parameters to calculate error against.
#--------------------------------------------------------------------------------
def plot_err(axis,model,data,popt,ylim=[]):
    CAP = 1000      # Error values capped at CAP, or numpy may crash on saving plots.
    MIN = 1.0e-5    # Min value for division.
    YLIM = 25       # Limit values for y-axis.

    # Note: the 1.0*data line is important, otherwise we just get a ref to data.
    if 'u' in data:
        S = data['u'][:,1] - MU.get_function(model,'u')(data['u'][:,0],*popt)
        d = 1.0*data['u'][:,1]  # See note.
        np.copyto(d,MIN,where=d<MIN)
        err = 100.0 * (S / d)
        np.copyto(err,CAP,where=err>CAP)
        axis.plot(data['u'][:,0], err, '.b', label='Uniaxial Errors')
    if 'b' in data:
        S = data['b'][:,1] - MU.get_function(model,'b')(data['b'][:,0],*popt)
        d = 1.0*data['b'][:,1]  # See note.
        np.copyto(d,MIN,where=d<MIN)
        err = 100.0 * (S / d)
        np.copyto(err,CAP,where=err>CAP)
        axis.plot(data['b'][:,0], err, '.g', label='Biaxial Errors')
    if 'p' in data:
        S = data['p'][:,1] - MU.get_function(model,'p')(data['p'][:,0],*popt)
        d = 1.0*data['p'][:,1]  # See note.
        np.copyto(d,MIN,where=d<MIN)
        err = 100.0 * (S / d)
        np.copyto(err,CAP,where=err>CAP)
        axis.plot(data['p'][:,0], err, '.r', label='Planar Errors')

    # Set the axis limits.
    if len(ylim)==0 or ylim[0]==ylim[1]:
        ylim = axis.get_ylim()
        ylim = [ylim[0] if ylim[0]>-YLIM else -YLIM,
                ylim[1] if ylim[1]<YLIM else YLIM]
    axis.set_ylim(ylim)
    axis.grid()
    axis.set_xlabel('Engineering Strain')
    axis.set_ylabel('Error (%)')
    axis.legend(loc='best',frameon=False,framealpha=0)


#--------------------------------------------------------------------------------
# Calculate the parameters to fit a model to the experimental data.
#   data       = dictionary of experimental datasets.
#   descr      = first part of filename.
#   model      = module defining the model to fit.
#   method     = method to use for minimization (eg, 'cg')
#   poisson    = Poisson's ratio of material.
#   t          = type of fit to perform (eg, 'u' for fitting uniaxial data)
#   num_points = number of random points to try for optimization start.
#   params     = dictionary to store results and residual.
#   ext        = image extension (with .).
#   xlim       = x-axis limits (strain).
#   ylim       = y-axis limits (stress).
#   ylimerr    = y-axis limits for error plot (percent).
#--------------------------------------------------------------------------------
def calc_params(data,descr,model,method,poisson,t,num_points,params,ext,xlim=[],ylim=[],ylimerr=[]):
    maxitr = 5000   # Default = 100*(len(data[t][:,0])+1)
    nparams = len(inspect.getargspec(model.stressU)[0])-1
    print '---------------------------------------------------------'
    print ' Calculating fit to',MU.get_name(t),'data'
    print '  with',num_points,'guess(es)'
    print '  using the \''+method+'\' fitting method'
    print '---------------------------------------------------------'
    S_best = -1.0e8
    iname = descr+'--'+model.name()+'--'+MU.get_name(t)+ext
    title = model.pname()+' fit to '+MU.get_name(t)+' Data'
    for i in range(num_points):
        p0 = np.longdouble(6.0*np.random.random_sample(nparams)-1.0)
        print '\tStarting point '+str(i)+':\t',p0
        try:
            if len(t)==1:
                mfun = MU.get_function(model,t)
                popt = CF.curve_fit1(mfun,data[t][:,0],data[t][:,1],method,p0,maxitr)
            elif len(t)==2:
                mfun0 = MU.get_function(model,t[0])
                mfun1 = MU.get_function(model,t[1])
                popt = CF.curve_fit2(mfun0,data[t[0]][:,0],data[t[0]][:,1],
                                     mfun1,data[t[1]][:,0],data[t[1]][:,1],
                                     method,p0,maxitr)
            elif len(t)==3:
                mfun0 = MU.get_function(model,t[0])
                mfun1 = MU.get_function(model,t[1])
                mfun2 = MU.get_function(model,t[2])
                popt = CF.curve_fit3(mfun0,data[t[0]][:,0],data[t[0]][:,1],
                                     mfun1,data[t[1]][:,0],data[t[1]][:,1],
                                     mfun2,data[t[2]][:,0],data[t[2]][:,1],
                                     method,p0,maxitr)
            else: print_error('Invalid fit.',True)
        except Exception:
            print '\t\tERROR:',sys.exc_info()[1]
            continue
        # TODO - check if u*a>0 for each pair. Implement as a check function in model?
        # TODO - check if sum(ui*ai) = 2*shear_modulus
        # TODO - test Drucker stability, as in Abaqus.
        # TODO - always iterate towards best global residual? Not exactly what the user is asking for...
        S = calculate_rsquared(data,model,popt,t)
        print '\t\tFinal:   ',popt
        print '\t\tLocal Rsquared: ',S
        if S<=S_best: continue
        S_best = S
        S_global = calculate_rsquared(data,model,popt,'ubp')
        print '\t\tGlobal Rsquared:',S_global
        print '\t\t** New Best Result. Updating Plots **'
        params[t] = np.append(popt,[S_best,S_global])

        # Plot.
        D = model.compressibility(poisson,*params[t][:-2])
        create_plots(model,data,popt,S,D,iname,title+'\n'+args.uniaxial+'\n'+args.biaxial+'\n'+args.planar,xlim,ylim,ylimerr)
        MU.write_matfile(model,descr,t,params[t][:-2],D)
    if S_best!=-1.0e8:
        print '\n\tBest-fit LRsquared:',params[t][-2]
        print '\tBest-fit GRsquared:',params[t][-1]
    else:
        U.print_error("No suitable fit found.",False)
    print '\n\n'

#--------------------------------------------------------------------------------
# Calculate Rsquared.
#--------------------------------------------------------------------------------
def calculate_rsquared(data,model,popt,t):
    if len(data)==0: return 0.0
    for char in t:
        if not char in data: continue
        ydata = data[char][:,1]
        yfit  = MU.get_function(model,char)(data[char][:,0],*popt)
        if 'yfits' in locals():
            ydatas = np.hstack((ydatas,ydata))
            yfits = np.hstack((yfits, yfit))
        else:
            ydatas = ydata
            yfits  = yfit
    return CF.rsquared(ydatas,yfits)


#--------------------------------------------------------------------------------
# Main.
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    tinit = time.time()
    # Handle user input.
    # TODO - support running multiple material models?
    parser = argparse.ArgumentParser(description="Fit the given datasets to a material model. For missing datasets, enter 'none'.",
                                     epilog="Example: calc_hyperelastic_parameters.py -m cg 0.48 ogden6 test uni.dat bi.dat plan.dat")
    parser.add_argument("-m","--method",default='tnc',help="Method to use for fitting, method=help to see list.")
    parser.add_argument("-p","--points",type=int,default=15,help="Number of random starting points (default 15).")
    parser.add_argument("--discard",type=float,help="Discard datapoints below a certain strain.")
    parser.add_argument("--datapoints",type=int,default=1000,help="The target number of datapoints to fit from each dataset (default 1000).")
    parser.add_argument("poisson",type=float,help="Poisson's ratio.")
    parser.add_argument("model",help="Name of model to fit, use model 'help' to see list.")
    parser.add_argument("descr",help="Additional descriptive term to add to output file titles.")
    parser.add_argument("uniaxial",nargs='?',default='none',help="Optional uniaxial dataset to fit, columns of stress and strain (default 'none').")
    parser.add_argument("biaxial",nargs='?',default='none',help="Optional biaxial dataset to fit, columns of stress and strain (default 'none').")
    parser.add_argument("planar",nargs='?',default='none',help="Optional planar dataset to fit, columns of stress and strain (default 'none').")
    parser.add_argument("volumetric",nargs='?',default='none',help="Optional volumetric dataset to fit (default 'none').")
    parser.add_argument("--xlim",nargs=2,type=float,default=[0.0,0.0],help="Min,Max values on xscale (AUTO if undefined).")
    parser.add_argument("--ylim",nargs=2,type=float,default=[0.0,0.0],help="Min,Max values on yscale (AUTO if undefined).")
    parser.add_argument("--ylimerr",nargs=2,type=float,default=[0.0,0.0],help="Min,Max values on yscale for error plots (AUTO if undefined).")
    parser.add_argument("--eps",action='store_true',help="Create eps plots instead of png.")
    args = parser.parse_args()
    ext = '.png' if not args.eps else '.eps'
    # 'suppress' disables scientific notation for small numbers.
    np.set_printoptions(precision=4,linewidth=130,suppress=True)
    np.seterr(all='raise')
    pyplot.rc('savefig',dpi=300)
    pyplot.rc('font',size=8)
    pyplot.rc('mathtext',default='regular') # Don't use italics for mathmode.

    # Error checking on the given method.
    min_opts = ('nelder-mead','powell','cg','bfgs','newton-cg','l-bfgs-b','tnc', 'cobyla','slsqp','dogleg','trust-ncg')
    if args.method.lower() in min_opts: method=args.method
    else:
        U.print_error('Invalid minimization method: '+args.method,False)
        print 'Options are:',min_opts
        exit(1)

    # Find the model module.
    model = MU.get_model(args.model)
    if not args.volumetric.lower()=='none': U.print_error('Volumetric fitting is unsupported.',True)

    # Read in the given datasets.
    data = dict()
    print '--------------------------------------------------------------------'
    print ' Importing datasets...'
    MU.import_dataset(args.uniaxial,data,'u')
    MU.import_dataset(args.biaxial,data,'b')
    MU.import_dataset(args.planar,data,'p')
    for key in 'ubp':
        if not key in data: continue
        # Discard datapoints if requested.
        if args.discard:
            data[key] = data[key][data[key][:,0]>=args.discard,:]
            data[key] = np.vstack(([0.0,0.0],data[key]))
            print '** Discarded points below',args.discard,'strain, now ',data[key].shape[0],MU.get_name(key),'points. **'
        # TODO - use smoothing here, instead of sampling.
        if data[key].shape[0]>args.datapoints:
            data[key] = data[key][range(0,data[key].shape[0],int(data[key].shape[0]/args.datapoints)),:]
            print '** Smoothed points, now ',data[key].shape[0],MU.get_name(key),'points. **'
    MU.import_dataset(args.volumetric,data,'v')

    # Fit three datasets together.
    params = dict()
    if 'u' in data and 'b' in data and 'p' in data:
        calc_params(data,args.descr,model,method,args.poisson,'ubp',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)

    # Fit two datasets together.
    if len(params)==0:
        if 'u' in data and 'b' in data: calc_params(data,args.descr,model,method,args.poisson,'ub',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)
        if 'b' in data and 'p' in data: calc_params(data,args.descr,model,method,args.poisson,'bp',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)
        if 'p' in data and 'u' in data: calc_params(data,args.descr,model,method,args.poisson,'pu',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)

    # Fit the datasets individually.
    if len(params)==0:
        if 'u' in data: calc_params(data,args.descr,model,method,args.poisson,'u',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)
        if 'b' in data: calc_params(data,args.descr,model,method,args.poisson,'b',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)
        if 'p' in data: calc_params(data,args.descr,model,method,args.poisson,'p',args.points,params,ext,args.xlim,args.ylim,args.ylimerr)

    # TODO - Drucker stability check?

    print '--------------------------------------------------------------------'
    print ' Results for',model.pname(),'fits.'
    print ' Format:',model.params()
    print ' Model Description:',model.descr()
    print '--------------------------------------------------------------------'
    np.set_printoptions(suppress=False)
    for t,p in params.iteritems():
        print '* Parameters for fit to',MU.get_name(t),'data.'
        print 'LRsquared:',p[-2]
        print 'GRsquared:',p[-1]
        print 'P:',p[:-2]
        print 'D:',model.compressibility(args.poisson,*p[:-2])
        print
    print 'TOTAL TIME ELAPSED: ',U.time_elapsed(tinit)
