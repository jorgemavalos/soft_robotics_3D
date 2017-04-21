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

import matplotlib
matplotlib.use('Agg')                   # Matplotlib uses the png renderer for the images. Needed for cluster.
import argparse, time, os, textwrap
import os.path as path
import numpy as np
import scipy.optimize as opt
import scipy.interpolate as interp
import utility as U
import run_tests as RT
import curve_fit as CH
import calc_hyperelastic_parameters as CHP
from matplotlib import pyplot,axes
import model_utility as MU


#--------------------------------------------------------------------------------
# Minimization function.
#   act      = Type of actuator, [lin,bnd,both].
#   test     = Actuator test type, [U,F,UF].
#   cmd      = type of brute-force test to run.
#   expdata  = dictionary of imported experimental data.
#   matdata  = dictionary of imported material data (uniaxial, biaxial, planar data).
#   testfunc = function to run test.
#   *popt    = optimization parameters for the model.
#--------------------------------------------------------------------------------
def calc_fit(act,test,cmd,expdata,matdata,testfunc,*popt):
    fig,ax = pyplot.subplots(1,2,figsize=(10,6))
    print '\n********************************************************************'
    print '** Testing with parameters: ',popt
    print '********************************************************************'
    # Get the simulation results.
    if not hasattr(calc_fit, "iter"): calc_fit.iter=0
    simdata = testfunc((calc_fit.iter,)+popt)
    if len(simdata)==0:
        print '\n********************************************************************'
        print ' Abaqus failed, we return an average error of 1000.0'
        print '********************************************************************'
        calc_fit.iter = calc_fit.iter + 1
        return 1000.0

    # Loop over multiple simulation results, if necessary.
    NP = 50
    errs = dict()
    cm = pyplot.get_cmap('jet')
    colors = [cm(float(i)/(2*len(simdata))) for i in range(2*len(simdata))]
    cidx = 0
    for T,sdata in simdata.iteritems():
        # Plot the simulation data.
        ax[0].plot(1000.0*sdata[:,0],sdata[:,1],'o-',color=colors[cidx],label='Sim '+T)

        # Interpolate the simulation results to remove zigzags and evenly space the points.
        # This is important because otherwise we may weight solutions very heavily at one point
        # in one solution, and completely differently in another.
        u,idx = np.unique(sdata[:,0],return_index=True)
        try:
            sim_smooth = np.zeros((NP,2))
            # sim_smooth[:,0] = np.linspace(min(sdata[idx,0]),max(sdata[idx,0]),NP)
            xvals = 1.0 - (np.logspace(0,np.log10(NP+1.0),NP,endpoint=True)-1.0)/(NP+1.0)
            sim_smooth[:,0] = min(sdata[idx,0]) + (max(sdata[idx,0])-min(sdata[idx,0]))*xvals
            dataF = interp.UnivariateSpline(sdata[idx,0],sdata[idx,1],k=3,s=0)
            sim_smooth[:,1] = dataF(sim_smooth[:,0])
            ax[0].plot(1000.0*sim_smooth[:,0],sim_smooth[:,1],'.--',color=colors[cidx],label='Sim '+T+' (Smoothed)')
        except:
            print '** Failed to smooth simulation data. Proceding with untouched data.'
            sim_smooth = 1.0*sdata[idx,:]

        # Interpolate the experimental results to find comparable data.
        edata = expdata[T]
        ax[0].plot(1000.0*edata[:,0],edata[:,1],'-',color=colors[cidx+1],label='Exp '+T)
        dataF = interp.UnivariateSpline(edata[:,0], edata[:,1],k=3,s=0)
        exp_smooth = np.zeros(sim_smooth.shape)
        exp_smooth[:,0] = sim_smooth[:,0]
        exp_smooth[:,1] = dataF(exp_smooth[:,0])
        ax[0].plot(1000.0*exp_smooth[:,0],exp_smooth[:,1],'.--',color=colors[cidx+1],label='Exp '+T+' (Smoothed)')

        # Make the comparison.
        # Try to handle a case where Abaqus failed early, but we have partial results.
        if max(sdata[:,0])<0.666*max(edata[:,0]):
            errs[T] = 1000.0 - 1000.0*max(sdata[:,0])/max(edata[:,0])
            print '\n********************************************************************'
            print ' Abaqus only reached',max(sdata[:,0]),' (out of '+str(max(edata[:,0]))+')!'
            print '********************************************************************'
        else:
            # errs[T] = np.linalg.norm((exp_smooth[:,1]-sim_smooth[:,1])/max(edata[:,1]))
            errs[T] = 1.0 - CH.rsquared(exp_smooth,sim_smooth)
        cidx = cidx+2

    # Plot the model uniaxial, biaxial, and planar curves.
    if cmd!='vis':
        for t in 'ubp':
            if not t in matdata: continue
            dataF = interp.interp1d(matdata[t][:,0], matdata[t][:,1], kind='nearest', bounds_error=False, fill_value=0.0)
            vals = np.zeros((NP,3))
            vals[:,0] = np.linspace(min(matdata[t][:,0]),max(matdata[t][:,0]),NP)
            vals[:,1] = dataF(vals[:,0])
            vals[:,2] = MU.get_function(model,t)(vals[:,0],*popt)  # splat operator.
            # errs['mat_'+t] = np.linalg.norm((vals[:,1]-vals[:,2])/max(matdata[t][:,1]))
            errs['mat_'+t] = 1.0 - CH.rsquared(vals[:,1],vals[:,2])
        CHP.plot_data(ax[1],model,matdata,popt,[0,4],[0,.25])
        ax[1].set_aspect(np.diff(ax[1].get_xlim())/np.diff(ax[1].get_ylim())) # Square axis.


    # Plot settings.
    if cmd=='mat':
        ax[0].set_xlabel('Engineering Strain')
        ax[0].set_ylabel(r'Nominal Stress $\left({}^N\!/{}_{mm^2}\right)$')
    elif cmd=='act' or cmd=='vis':
        ax[0].set_xlabel('Internal Pressure (kPa)')
        if len(simdata)>1: ax[0].set_ylabel('Displacement (mm) or Force (N)')
        else:
            if 'linU' in simdata or 'bndU' in simdata:   ax[0].set_ylabel('Displacement (mm)')
            elif 'linF' in simdata or 'bndF' in simdata: ax[0].set_ylabel('Force (N)')
    ax[0].grid()
    # Plot with the same axis every time.
    if not hasattr(calc_fit, "xlim"):
        calc_fit.xlim = ax[0].get_xlim()
        calc_fit.ylim = ax[0].get_ylim()
    ax[0].set_xlim(calc_fit.xlim)
    ax[0].set_ylim(calc_fit.ylim)
    ax[0].set_aspect(np.diff(calc_fit.xlim)/np.diff(calc_fit.ylim)) # Square axis.
    # ax[0].set_aspect(abs(xlim[1]-xlim[0])/abs(ylim[1]-ylim[0])) # Square axis.
    pyplot.suptitle('\n'.join(textwrap.wrap(str(popt),100))+'\nitr='+str(calc_fit.iter)+'\nerrs='+str(errs)+'\nerr_sum='+str(sum(errs.values())))
    L = ax[0].legend(loc='best',frameon=False,framealpha=0)
    for legobj in L.legendHandles: legobj.set_linewidth(2.0)
    pyplot.tight_layout()
    pyplot.savefig('results--'+str(calc_fit.iter)+'.png')
    pyplot.savefig('results--latest.png')
    pyplot.close()
    print '\n********************************************************************'
    print ' Calculated sum-of-squares error:',errs,'=',sum(errs.values())
    print '********************************************************************'
    calc_fit.iter = calc_fit.iter + 1
    return sum(errs.values())

#--------------------------------------------------------------------------------
# Create and run an Abaqus test for MAT.
#   geomcmd = Abaqus command to create test.
#   A       = cross-sectional area.
#   L       = sample length in testing direction.
#   model   = module defining the model to fit.
#   density = material density to use.
#   i       = iteration number.
#   *popt   = optimization parameters for the model.
#--------------------------------------------------------------------------------
def run_MAT(geomcmd,A,L,model,density,i,*popt):
    PROCS = 4
    matfile = MU.write_matfile(model,str(i),'up',popt,[],density)
    U.run_cmd(geomcmd+[matfile])
    RT.run_abq('test_hyper.inp','.',PROCS,str(i))
    results = dict()
    if RT.run_post('test_hyper.inp','.',str(i)):
        try:
            R = np.loadtxt(path.join('test_hyper-'+str(i),'data.rpt'),skiprows=4)
        except Exception:
            print '********************************************************************'
            print ' Failed to load data.rpt file, skipping dataset.'
            print '********************************************************************'
            return dict()
        if len(R.shape)==1: return dict()
        R = R[:,[2,1]]              # Length, Force.
        R[:,0] = R[:,0] / (0.5*L)   # Calculate engineering strain (with symmetry).
        R[:,1] = R[:,1] / A         # Calculate nominal stress.
        R = R[abs(R[:,1])<100.0]    # Filter out results from blow-ups.
        results['mat'] = R
    return dict()

#--------------------------------------------------------------------------------
# Create and run a multiple Abaqus tests for ACT.
#   act     = Type of actuator, [lin,bnd,both].
#   test    = Actuator test type, [U,F,UF].
#   geomcmd = Abaqus command to create test.
#   model   = module defining the model to fit.
#   density = material density to use.
#   i       = iteration number.
#   *popt   = optimization parameters for the model.
#--------------------------------------------------------------------------------
def run_multiACT(act,test,geomcmd,model,density,i,*popt):
    results = dict()
    gcmd = geomcmd
    # Insert the parameters into the geometry creation command and run the tests.
    if act=='lin' or act=='both':
        if test=='U' or test=='UF': # Run the displacement tests.
            gcmd[4] = 'test_hyper-linU.cae'
            gcmd[6] = 'lin'
            gcmd[7] = 'U'
            results['linU'] = run_ACT('lin','U',geomcmd,model,density,i,*popt)
            if results['linU']==[]: return dict()
        if test=='F' or test=='UF': # Run the force tests.
            geomcmd[4] = 'test_hyper-linF.cae'
            geomcmd[6] = 'lin'
            geomcmd[7] = 'F'
            results['linF'] = run_ACT('lin','F',geomcmd,model,density,i,*popt)
            if results['linF']==[]: return dict()
    if act=='bnd' or act=='both':
        if test=='U' or test=='UF': # Run the displacement tests.
            geomcmd[4] = 'test_hyper-bndU.cae'
            geomcmd[6] = 'bnd'
            geomcmd[7] = 'U'
            results['bndU'] = run_ACT('bnd','U',geomcmd,model,density,i,*popt)
            if results['bndU']==[]: return dict()
        if test=='F' or test=='UF': # Run the force tests.
            geomcmd[4] = 'test_hyper-bndF.cae'
            geomcmd[6] = 'bnd'
            geomcmd[7] = 'F'
            results['bndF'] = run_ACT('bnd','F',geomcmd,model,density,i,*popt)
            if results['bndF']==[]: return dict()
    return results

#--------------------------------------------------------------------------------
# Create and run an Abaqus test for ACT.
#   act     = Type of actuator, [lin,bnd].
#   test    = Actuator test type, [U,F].
#   geomcmd = Abaqus command to create test.
#   model   = module defining the model to fit.
#   density = material density to use.
#   i       = iteration number.
#   *popt   = optimization parameters for the model.
#--------------------------------------------------------------------------------
def run_ACT(act,test,geomcmd,model,density,i,*popt):
    PROCS = 8
    geomcmd[20] = MU.write_matfile(model,str(i),'up',popt,[],density)
    U.run_cmd(geomcmd)
    RT.run_abq('test_hyper-'+act+test+'.inp','.',PROCS,str(i))
    if RT.run_post('test_hyper-'+act+test+'.inp','.',str(i)):
        try:
            results = np.loadtxt(path.join('test_hyper-'+act+test+'-'+str(i),'data.rpt'),skiprows=4)
        except Exception:
            print '********************************************************************'
            print ' Failed to load data.rpt file, skipping dataset.'
            print '********************************************************************'
            return []
        if len(results.shape)==1: return []
        if act=='lin' and test=='U':
            results = results[:,[1,4]]              # Pressure, Z-displacement.
            results[:,1] = -1.0*results[:,1]
        elif act=='bnd' and test=='U':
            L = results[0,-1] - results[0,-2]       # Initial length of the actuator.
            angles = np.degrees(np.arctan(-results[:,3]/(L-results[:,4])))
            angles[angels<0.0] = angles[angles<0.0] + 180
            results = results[:,[1,2]]
            results[:,1] = angles
        elif act=='lin' and test=='F': results = results[:,[1,4]]  # Pressure, Z-Force.
        elif act=='bnd' and test=='F': results = results[:,[1,3]]  # Pressure, Y-Force.
        results = results[abs(results[:,1])<100.0]  # Filter out results from blow-ups.
        return results
    return []

#--------------------------------------------------------------------------------
# Create and run an Abaqus test for VIS.
#   act     = Type of actuator, [lin,bnd].
#   test    = Actuator test type, [U,F].
#   geomcmd = Abaqus command to create test.
#   i       = iteration number.
#   *popt   = optimization parameters for the model.
#--------------------------------------------------------------------------------
def run_VIS(act,test,geomcmd,i,*popt):
    PROCS = 8
    geomcmd[21] = MU.write_viscomatfile(str(i),popt[0],popt[1:],True)
    U.run_cmd(geomcmd)
    RT.run_abq('test_visco.inp','.',PROCS,str(i))
    if RT.run_post('test_visco.inp','.',str(i)):
        try:
            results = np.loadtxt(path.join('test_visco-'+str(i),'data.rpt'),skiprows=4)
        except Exception:
            print '********************************************************************'
            print ' Failed to load data.rpt file, skipping dataset.'
            print '********************************************************************'
            return []
        if len(results.shape)==1: return []
        if test=='U':
            results = results[:,[1,4]]              # Pressure, Z-displacement.
            results[:,1] = -1.0*results[:,1]
        elif act=='bnd' and test=='U':
            L = results[0,-1] - results[0,-2]       # Initial length of the actuator.
            angles = np.degrees(np.arctan(-results[:,3]/(L-results[:,4])))
            angles[angels<0.0] = angles[angles<0.0] + 180
            results = results[:,[1,2]]
            results[:,1] = angles
        elif act=='lin' and test=='F': results = results[:,[1,4]]  # Pressure, Z-Force.
        elif act=='bnd' and test=='F': results = results[:,[1,3]]  # Pressure, Y-Force.
        results = results[abs(results[:,1])<100.0]  # Filter out results from blow-ups.
        return results
    return []

#--------------------------------------------------------------------------------
# Main.
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    tinit = time.time()
    # Handle user input.
    parser = argparse.ArgumentParser(
            description="Attempt to optimize hyperelastic material parameters based on additional datasets.",
            epilog="Example: optimize_hyperelastic_parameters.py act ./bf_test lin UF dataset.csv matfile.mat 200 4")
    parser.add_argument("--method",default='nelder-mead',
            choices=['nelder-mead','powell','cg','bfgs','newton-cg','l-bfgs-b','tnc', 'cobyla','slsqp','dogleg','trust-ncg'],
            help="Method to use for fitting, default is Nelder-Mead.")
    subparsers = parser.add_subparsers(dest='cmd',help='Choose a subcommand.')

    parser_MAT = subparsers.add_parser('mat',help='Optimize a material testing curve (see mat --help).')
    parser_MAT.add_argument("dirname",help="Name of directory to create, where output will be stored.")
    parser_MAT.add_argument("dim",nargs=3,help="Depth, width, and length of sample (mm). Length will be divided by 2 for symmetry.")
    parser_MAT.add_argument("matfile",help="File containing initial hyperelastic material properties.")
    parser_MAT.add_argument("fittingdata",help='Actuator dataset(s) to fit, consisting of two columns: stress,strain.')
    parser_ACT.add_argument('--uniaxial_data',default='none',help='Optional uniaxial data to include in fitting process.')
    parser_ACT.add_argument('--biaxial_data',default='none',help='Optional biaxial data to include in fitting process.')
    parser_ACT.add_argument('--planar_data',default='none',help='Optional planar data to include in fitting process.')

    parser_ACT = subparsers.add_parser('act',help='Optimize an actuator result curve (see act --help).')
    parser_ACT.add_argument("dirname",help="Name of directory to create, where output will be stored.")
    parser_ACT.add_argument('actuator',choices=['lin','bnd','both'],help='Type(s) of actuator to create and test.')
    parser_ACT.add_argument('test',choices=['U','F','UF'],help='Type of test to run: U, F, or both U and F (UF).')
    parser_ACT.add_argument('matfile',help='File containing hyperelastic material properties.')
    parser_ACT.add_argument('time',help='Internal simulation time (try 200 for quasistatic).')
    parser_ACT.add_argument('num_chambers',help='Number of chambers.')
    parser_ACT.add_argument("fittingdata",nargs='+',help='Actuator dataset(s) to fit, consisting of two columns: '
            'pressure, and force or displacement. Comma seperated, comments as #.')
    parser_ACT.add_argument('--mesh_size','-m',default='2.0',metavar='S',help='Approximate size of mesh elements (default 2.0).')
    parser_ACT.add_argument('--maxnuminc',default='10000',help='Maximum number of increments (default 10000).')
    parser_ACT.add_argument('--chamber',nargs=3,default=['6.0','6.0','2.0'],metavar=('H','W','L'),
            help='Height, width, and length of a chamber (default \'6.0 6.0 2.0\').')
    parser_ACT.add_argument('--inlet',nargs=2,default=['2.0','2.0'],metavar=('H','W'),
            help='Height and width of inlet tunnel (default \'2.0 2.0\').')
    parser_ACT.add_argument('--wall',default='3.0',help='Wall thickness (default 3.0).')
    parser_ACT.add_argument('--uniaxial_data',default='none',help='Optional uniaxial data to include in fitting process.')
    parser_ACT.add_argument('--biaxial_data',default='none',help='Optional biaxial data to include in fitting process.')
    parser_ACT.add_argument('--planar_data',default='none',help='Optional planar data to include in fitting process.')

    parser_VIS = subparsers.add_parser('vis',help='Optimize an actuator result curve with viscoelasticity (see vis --help).')
    parser_VIS.add_argument("dirname",help="Name of directory to create, where output will be stored.")
    parser_VIS.add_argument('actuator',choices=['lin','bnd','both'],help='Type(s) of actuator to create and test.')
    parser_VIS.add_argument('test',choices=['U','F','UF'],help='Type of test to run: U, F, or both U and F (UF).')
    parser_VIS.add_argument('matfile',help='File containing hyperelastic material properties.')
    parser_VIS.add_argument('visco',help='File containing initial viscoelastic material properties.')
    parser_VIS.add_argument('time',help='Internal simulation time (try 200 for quasistatic).')
    parser_VIS.add_argument('num_chambers',help='Number of chambers.')
    parser_VIS.add_argument("fittingdata",nargs='+',help='Actuator dataset(s) to fit, consisting of two columns: '
            'pressure, and force or displacement. Comma seperated, comments as #.')
    parser_VIS.add_argument('--mesh_size','-m',default='0.8',metavar='S',help='Approximate size of mesh elements (default 0.8).')
    parser_VIS.add_argument('--chamber',nargs=3,default=['6.0','6.0','2.0'],metavar=('H','W','L'),
            help='Height, width, and length of a chamber (default \'6.0 6.0 2.0\').')
    parser_VIS.add_argument('--inlet',nargs=2,default=['2.0','2.0'],metavar=('H','W'),
            help='Height and width of inlet tunnel (default \'2.0 2.0\').')
    parser_VIS.add_argument('--wall',default='3.0',help='Wall thickness (default 3.0).')
    args = parser.parse_args()
    np.set_printoptions(precision=4,linewidth=130,suppress=True) # Disable scientific notation for small numbers.
    pyplot.rc('savefig',dpi=300)
    pyplot.rc('font',size=8)
    pyplot.rc('mathtext',default='regular') # Don't use italics for mathmode.

    #--------------------------------------------------------------------------------
    # Read in the given datasets.
    #--------------------------------------------------------------------------------
    print '--------------------------------------------------------------------'
    print ' Importing datasets...'
    data = dict()
    matdata = dict()
    if args.cmd=='mat':
        data['mat'] = np.loadtxt(args.fittingdata,comments='#',delimiter=',')
        data['mat'] = data[:,[1,0]] # Swap columns.
        print '  Imported',data['mat'].shape[0],'points from',args.fittingdata
        if args.uniaxial_data!='none': MU.import_dataset(args.uniaxial_data,matdata,'u')
        if args.biaxial_data!='none':  MU.import_dataset(args.biaxial_data,matdata,'b')
        if args.planar_data!='none':   MU.import_dataset(args.planar_data,matdata,'p')
    else:
        data = dict()
        pressure = 1.0e10
        for i,fname in enumerate(args.fittingdata):
            T = args.actuator + args.test[i]
            data[T] = np.loadtxt(fname,comments='#',delimiter=',')
            pressure = min(pressure,max(data[T][:,0]))
            print '  Imported',data[T].shape[0],'points from',fname,'for test type:',T
        if args.uniaxial_data!='none': MU.import_dataset(args.uniaxial_data,matdata,'u')
        if args.biaxial_data!='none':  MU.import_dataset(args.biaxial_data,matdata,'b')
        if args.planar_data!='none':   MU.import_dataset(args.planar_data,matdata,'p')

    #--------------------------------------------------------------------------------
    # Prepare variables.
    #--------------------------------------------------------------------------------
    if args.cmd=='mat':
        # Initial parameters.
        mat = MU.read_matfile(args.matfile)
        model = MU.get_model(mat['model'] + (str(mat['order']) if 'order' in mat else ''))
        popt = mat['params']
        strain = max(data['mat'][:,0])
        # Other variables.
        act = ''
        test = ''
        A = float(args.dim[0]) * float(args.dim[1])
        L = float(args.dim[2])     # Divided by two later, in abq_hypertest.py
        script = path.abspath(U.find_file('python/abq_hypertest.py'))
        geomcmd = ['abaqus','cae','noGUI='+script,'--',
                   args.dim[0],args.dim[1],args.dim[2],
                   str(strain),'1.0']
        testfunc = lambda Rargs: run_MAT(geomcmd,A,L,model,mat['density'],*Rargs)
    elif args.cmd=='act':
        # Initial parameters.
        mat = MU.read_matfile(args.matfile)
        model = MU.get_model(mat['model'] + (str(mat['order']) if 'order' in mat else ''))
        popt = mat['params']
        # Other variables.
        act = args.actuator
        test = args.test
        script = path.abspath(U.find_file('python/abq_create_geom.py'))
        geomcmd = ['abaqus', 'cae', 'noGUI='+script,'--',
                   'test_hyper-'+act+test+'.cae', 'noale', act,
                   test, args.mesh_size, str(pressure),
                   args.time, args.num_chambers, '1wall',
                   args.inlet[0], args.inlet[1],
                   args.chamber[0], args.chamber[1], args.chamber[2],
                   args.wall, '1.0', 'MATFILE', 'none', '0.0', args.maxnuminc]
        testfunc = lambda Rargs: run_multiACT(act,test,geomcmd,model,mat['density'],*Rargs)
    elif args.cmd=='vis':
        # Initial parameters.
        vmat = MU.read_viscomatfile(args.visco)
        terms = len(vmat['g'])
        popt = [vmat['G0']]
        for t in range(terms): popt = popt + [vmat['g'][t],vmat['tau'][t]]
        # Other variables.
        act = args.actuator
        test = args.test
        script = path.abspath(U.find_file('python/abq_create_geom.py'))
        matfile = path.abspath(args.matfile)
        geomcmd = ['abaqus', 'cae', 'noGUI='+script,'--',
                   'test_visco.cae', 'noale', act,
                   test, args.mesh_size, str(pressure),
                   args.time, args.num_chambers, '1wall',
                   args.inlet[0], args.inlet[1],
                   args.chamber[0], args.chamber[1], args.chamber[2],
                   args.wall, '1.0', matfile, 'VISCOFILE', '0.0', args.maxnuminc]
        testfunc = lambda Rargs: run_VIS(act,test,geomcmd,*Rargs)

    #--------------------------------------------------------------------------------
    # Change directory.
    #--------------------------------------------------------------------------------
    if path.exists(args.dirname): U.print_error('Output directory exists.',True)
    os.makedirs(args.dirname)
    rootdir = os.getcwd()
    os.chdir(args.dirname)

    #--------------------------------------------------------------------------------
    # Calculate optimal parameters.
    #--------------------------------------------------------------------------------
    print '---------------------------------------------------------'
    print ' Calculating parameters...'
    maxitr = 20000
    minfunc = lambda Largs: calc_fit(act,test,args.cmd,data,matdata,testfunc,*Largs)
    OR = opt.minimize(minfunc,popt,method=args.method,options={'maxiter':maxitr,'eps':0.005})
    # kwargs = {"method": args.method}
    # OR = opt.basinhopping(minfunc,popt,niter=maxitr,minimizer_kwargs=kwargs,
            # T=0.1,stepsize=0.01,interval=5)
    if not OR.success: print '\t\tWARNING: '+OR.message
    popt = OR.x


    np.set_printoptions(suppress=False)
    print '\n\n--------------------------------------------------------------------'
    print ' Final results.'
    print '--------------------------------------------------------------------'
    print 'Parameters:\n',popt
    print 'TOTAL TIME ELAPSED: ',U.time_elapsed(tinit)
    os.chdir(rootdir)
