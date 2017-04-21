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
from matplotlib import pyplot,axes


#--------------------------------------------------------------------------------
# Calculate the number of chambers from a float.
# nc_float = floating point representation.
# init_nc  = number of chambers given in first iteration.
#--------------------------------------------------------------------------------
def calc_nc(nc_float, init_nc):
    # In addition to rounding to the nearest int (0.5 rounds up),
    # we can also increase the sensitivity of the nc parameter during
    # optimization. Essentially we multiply the delta by 10.
    # nc = init_nc + int(np.rint(10.0*(nc_float-init_nc)))
    nc = int(np.rint(nc_float))
    if nc<=1.0: return 1
    return nc

#--------------------------------------------------------------------------------
# Minimization function.
#   OF         = output file, for recording status.
#   cmd        = type of brute-force test to run.
#   expdata    = dictionary of imported experimental data.
#   init_nc    = number of chambers given in first iteration.
#   interpBOOL = define whether to interpolate the experimental datapoints.
#   optfunc    = function to run test.
#   *popt      = optimization parameters.
#--------------------------------------------------------------------------------
def calc_fit(OF,cmd,expdata,init_nc,interpBOOL,optfunc,*popt):
    fig,axis = pyplot.subplots(1,1)
    popt_display = np.abs(list(popt))
    popt_display[0] = calc_nc(popt[0],init_nc)
    print '\n********************************************************************'
    print '** Testing with parameters: ',popt_display
    print '********************************************************************'
    # Get the simulation results.
    if not hasattr(calc_fit, "iter"): calc_fit.iter=0
    simdata = optfunc((calc_fit.iter,)+popt)
    if len(simdata)==0:
        print '\n********************************************************************'
        print ' Abaqus failed, we return an average error of 1000.0'
        print '********************************************************************'
        calc_fit.iter = calc_fit.iter + 1
        pyplot.close()
        return 1000.0

    # Loop over multiple simulation results, if necessary.
    NP = 50
    errs = dict()
    cm = pyplot.get_cmap('jet')
    colors = [cm(float(i)/(2*len(simdata))) for i in range(2*len(simdata))]
    cidx = 0
    for T,sdata in simdata.iteritems():
        if T in expdata: edata=expdata[T]
        elif T.endswith('--strain'):
            if 'strain' in expdata: edata=expdata['strain']
            else: continue
        else: U.print_error('Failed to find comparison data for '+T,True)

        # Plot the simulation data.
        axis.plot(1000.0*sdata[:,0],sdata[:,1],'o-',color=colors[cidx],label='Sim '+T)

        # Interpolate the simulation results to remove zigzags and evenly space the points.
        # This is important because otherwise we may weight solutions very heavily at one point
        # in one solution, and completely differently in another.
        u,idx = np.unique(sdata[:,0],return_index=True)
        try:
            if interpBOOL:
                sim_smooth = np.zeros((NP,2))
                sim_smooth[:,0] = np.linspace(min(sdata[idx,0]),max(sdata[idx,0]),NP)
                # xvals = 1.0 - (np.logspace(0,np.log10(NP+1.0),NP,endpoint=True)-1.0)/(NP+1.0)
                # sim_smooth[:,0] = min(sdata[idx,0]) + (max(sdata[idx,0])-min(sdata[idx,0]))*xvals
            else:
                sim_smooth = np.zeros(edata.shape)
                sim_smooth[:,0] = edata[:,0]
            dataF = interp.UnivariateSpline(sdata[idx,0],sdata[idx,1],k=3,s=0)
            sim_smooth[:,1] = dataF(sim_smooth[:,0])
            # Remove any extrapolations, replace with the max calculated value.
            sim_smooth[sim_smooth[:,0]>max(sdata[:,0]),1] = max(sdata[:,1])
            axis.plot(1000.0*sim_smooth[:,0],sim_smooth[:,1],'.--',color=colors[cidx],label='Sim '+T+' (Smoothed)')
        except:
            print '** Failed to smooth simulation data. Proceding with untouched data.'
            sim_smooth = 1.0*sdata[idx,:]
            interpBOOL = True

        # Interpolate the experimental results to find comparable data.
        axis.plot(1000.0*edata[:,0],edata[:,1],'-',color=colors[cidx+1],label='Exp '+T)
        if interpBOOL:
            try:
                dataF = interp.UnivariateSpline(edata[:,0], edata[:,1],k=3,s=0)
                exp_smooth = np.zeros(sim_smooth.shape)
                exp_smooth[:,0] = sim_smooth[:,0]
                exp_smooth[:,1] = dataF(exp_smooth[:,0])
            except:
                U.print_error('Failed to fit a spline to experimental data, consider using the '
                              '--no_interpolate option or adding more datapoints.',True)
        else: exp_smooth = 1.0*edata
        axis.plot(1000.0*exp_smooth[:,0],exp_smooth[:,1],'.--',color=colors[cidx+1],label='Exp '+T+' (Smoothed)')

        # Make the comparison.
        # Try to handle a case where Abaqus failed early, but we have partial results.
        if max(sdata[:,0])<0.666*max(edata[:,0]):
            errs[T] = 1000.0 - 1000.0*max(sdata[:,0])/max(edata[:,0])
            print '\n********************************************************************'
            print ' Abaqus only reached',max(sdata[:,0]),' (out of '+str(max(edata[:,0]))+')!'
            print '********************************************************************'
        else:
            # For errors on max strain, no penalty for being too low.
            if T.endswith('--strain'):
                if max(exp_smooth[:,1])>=max(sim_smooth[:,1]): errs[T]=0.0
                else: errs[T] = 1.0 - CH.rsquared(exp_smooth,sim_smooth)
            else:
                errs[T] = 1.0 - CH.rsquared(exp_smooth,sim_smooth)
        cidx = cidx+2


    # Record to file.
    if calc_fit.iter==0:
        OF.write('# Each row shows the iteration #, the '+str(len(popt))+' parameters,\n')
        OF.write('# and then the '+str(len(errs.keys()))+' errors: '+str(errs.keys())+'\n')
    record_string = str(calc_fit.iter)+','
    for p in popt_display: record_string = record_string + str(p) + ','
    for err in errs.values(): record_string = record_string + str(err) + ','
    OF.write(record_string[:-1]+'\n')

    # Plot settings.
    if cmd=='act':
        axis.set_xlabel('Internal Pressure (kPa)')
        if len(simdata)>1: axis.set_ylabel('Displacement (mm) or Force (N)')
        else:
            if 'linU' in simdata or 'bndU' in simdata:   axis.set_ylabel('Displacement (mm)')
            elif 'linF' in simdata or 'bndF' in simdata: axis.set_ylabel('Force (N)')
    axis.grid()
    # Plot with the same axis every time.
    if not hasattr(calc_fit, "xlim"):
        calc_fit.xlim = axis.get_xlim()
        calc_fit.ylim = axis.get_ylim()
    axis.set_xlim(calc_fit.xlim)
    axis.set_ylim(calc_fit.ylim)
    axis.set_aspect(np.diff(calc_fit.xlim)/np.diff(calc_fit.ylim)) # Square axis.
    err_str = ', '.join("{:}:{:g}".format(key,val) for (key,val) in errs.iteritems())
    title = axis.set_title('\n'.join(textwrap.wrap(str(popt_display),100))+'\nitr='+str(calc_fit.iter)+'\nerrs=['+err_str+']\nerr_sum='+str(sum(errs.values())))
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    L = axis.legend(loc='best',frameon=False,framealpha=0)
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
# Create and run a multiple Abaqus tests for ACT.
#   act           = Type of actuator, [lin,bnd].
#   test          = Actuator test type, [U,F,UF].
#   dist_to_force = distance to the blocked force plate.
#   geomcmd       = Abaqus command to create test geometry.
#   pressures     = max pressure to run for each test.
#   init_nc       = number of chambers given in first iteration.
#   i             = iteration number.
#   *popt         = optimization parameters.
#--------------------------------------------------------------------------------
def run_multiACT(act,test,dist_to_force,geomcmd,pressures,init_nc,i,*popt):
    results = dict()
    gcmd = geomcmd
    # Insert the parameters into the geometry creation command.
    if act=='lin':
        nc = calc_nc(popt[0],init_nc)
        inlet_H =   abs(popt[1])
        inlet_W =   abs(popt[1])
        chamber_H = abs(popt[2])
        chamber_W = abs(popt[2])
        chamber_D = abs(popt[3])
        wall =      abs(popt[4])
    elif act=='bnd':
        nc = calc_nc(popt[0],init_nc)
        inlet_H =   abs(popt[1])
        inlet_W =   abs(popt[2])
        chamber_H = abs(popt[3])
        chamber_W = abs(popt[4])
        chamber_D = abs(popt[5])
        wall =      abs(popt[6])

    # Print dimensions.
    L = nc*chamber_D + (nc+1)*wall
    W = chamber_W + 2*wall
    H = chamber_H + 2*wall
    print 'Actuator Dimensions (L,W,H): ',L,W,H
    # Check some dimensions for physicality.
    # We explicitly check this one because sometimes the mesher will succeed at creating a mesh,
    # but the mesh will be missing pieces and the run will be meaningless.
    if inlet_H>=chamber_H or inlet_W>=chamber_W:
        print 'Inlet larger than chamber; non-physical.'    # Mesher can't properly handle this.
        return dict()
    # Assign parameters.
    gcmd[11] = str(nc)
    gcmd[13] = str(inlet_H)
    gcmd[14] = str(inlet_W)
    gcmd[15] = str(chamber_H)
    gcmd[16] = str(chamber_W)
    gcmd[17] = str(chamber_D)
    gcmd[18] = str(wall)
    if geomcmd[8]=='AUTO':  # Try to calculate a reasonable mesh refinement.
        gcmd[8] = str(min(inlet_H,inlet_W,chamber_D,min(chamber_H,chamber_W)/3.0,wall/2.0))

    # Run the displacement tests.
    if test=='U' or test=='UF':
        gcmd[4] = 'test_geom-'+act+'U.cae'  # Output Abaqus database.
        gcmd[7] = 'U'                       # Type of test to run.
        gcmd[9] = str(pressures[0])         # Pressure.
        results.update(run_ACT(act,'U',dist_to_force,gcmd,i))
        if not act+'U' in results: return dict()
    # Run the force tests.
    if test=='F' or test=='UF':
        gcmd[4] = 'test_geom-'+act+'F.cae'  # Output Abaqus database.
        gcmd[7] = 'F'                       # Type of test to run.
        gcmd[9] = str(pressures[-1])        # Pressure.
        results.update(run_ACT(act,'F',dist_to_force,gcmd,i))
        if not act+'F' in results: return dict()
    return results

#--------------------------------------------------------------------------------
# Create and run an Abaqus test for ACT.
#   act           = Type of actuator, [lin,bnd].
#   test          = Actuator test type, [U,F,UF].
#   dist_to_force = distance to the blocked force plate.
#   geomcmd       = Abaqus command to create test.
#   i             = iteration number.
#--------------------------------------------------------------------------------
def run_ACT(act,test,dist_to_force,geomcmd,i):
    PROCS = 8
    try: U.run_cmd_screen(geomcmd)                              # Create the test.
    except Exception:
        U.print_error('Failed to create geometry! Non-physical dimensions?',False)
        return dict()
    results = dict()
    RT.run_abq('test_geom-'+act+test+'.inp','.',PROCS,str(i))   # Run the test.
    if RT.run_post('test_geom-'+act+test+'.inp','.',str(i)):    # Postprocess the test.
        try:
            data = np.loadtxt(path.join('test_geom-'+act+test+'-'+str(i),'data.rpt'),skiprows=4)
        except Exception:
            U.print_error('Failed to load data.rpt file, skipping dataset.',False)
            return dict()
        if len(data.shape)==1: return dict()
        if act=='lin' and test=='U':
            data = data[:,[1,4]]              # Pressure, Z-displacement.
            data[:,1] = -1.0*data[:,1]
        elif act=='bnd' and test=='U':
            L = data[0,-1] - data[0,-2]       # Initial length of the actuator.
            angles = np.degrees(np.arctan(-data[:,3]/(L-data[:,4])))
            angles[angels<0.0] = angles[angles<0.0] + 180
            data = data[:,[1,2]]
            data[:,1] = angles
        elif act=='lin' and test=='F': data= data[:,[1,4]]  # Pressure, Z-Force.
        elif act=='bnd' and test=='F':
            # Need to calculate the force perpendicular to the plate.
            alpha = np.radians(dist_to_force)
            beta = np.radians(90 - dist_to_force)
            f_normal = np.abs(data[:,3]*np.cos(alpha)) + np.abs(data[:,4]*np.cos(beta))
            data = data[:,[1,2]]
            data[:,1] = f_normal     # Pressure, Normal-Force.
        results[act+test] = data
        try:
            # strains are time (same indices as pressure), nominal strain.
            strains = np.loadtxt(path.join('test_geom-'+act+test+'-'+str(i),'data-strain.csv'),delimiter=',')
            strains[:,0] = data[:,0]
            results[act+test+'--strain'] = strains
        except: pass
    return results


#--------------------------------------------------------------------------------
# Main.
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    tinit = time.time()
    # Handle user input.
    parser = argparse.ArgumentParser(
            description="Optimize geometric parameters based on desired performance characteristics.",
            epilog="Example: optimize_geometric_parameters.py")
    parser.add_argument("--method",default='cobyla',
            choices=['nelder-mead','powell','cg','bfgs','newton-cg','l-bfgs-b','tnc', 'cobyla','slsqp','dogleg','trust-ncg'],
            help="Method to use for optimization, default is cobyla.")
    parser.add_argument('--basinhopping',action='store_true',help='Also use a basin-hopping/simulated-annealing algorithm.')
    subparsers = parser.add_subparsers(dest='cmd',help='Choose a geometry type to optimize.')

    parser_ACT = subparsers.add_parser('act',help='Optimize parameters for an actuator (see act --help).')
    parser_ACT.add_argument("dirname",help="Name of directory to create, where output will be stored.")
    parser_ACT.add_argument('actuator',choices=['lin','bnd'],help='Type of actuator to create and test.')
    parser_ACT.add_argument('test',choices=['U','F','UF'],help='Type of test to run: U, F, or both U and F (UF).')
    parser_ACT.add_argument('matfile',help='File containing hyperelastic material properties.')
    parser_ACT.add_argument('time',type=float,help='Internal simulation time (try 200 for quasistatic).')
    parser_ACT.add_argument('fittingdata',nargs='+',help='Actuator dataset(s) to fit, consisting of two columns: '
            'pressure, and force or displacement. Comma seperated, comments as #.')
    parser_ACT.add_argument('--no_interpolate',action='store_true',help='Do not interpolate fitting data; compare with those datapoints exactly.')
    parser_ACT.add_argument('--dist_to_force',type=float,default=0.0,help='Distance (or angle for bending actuators) to the blocked force plate in blocked-force tests (default 0.0).')
    parser_ACT.add_argument('--maxnuminc',type=int,default=100000,help='Maximum number of increments (default 100000).')
    parser_ACT.add_argument('--num_chambers','-nc',type=int,default=4,help='Number of chambers (default 4).')
    parser_ACT.add_argument('--mesh_size','-m',default='AUTO',metavar='S',help='Approximate size of mesh elements, default (\'AUTO\') changes with each iteration.')
    parser_ACT.add_argument('--chamber',type=float,nargs=3,default=[6.0,6.0,2.0],metavar=('H','W','L'),
            help='Height, width, and length of a chamber (default \'6.0 6.0 2.0\').')
    parser_ACT.add_argument('--inlet',type=float,nargs=2,default=[2.0,2.0],metavar=('H','W'),
            help='Height and width of inlet tunnel (default \'2.0 2.0\').')
    parser_ACT.add_argument('--wall',type=float,default=3.0,help='Wall thickness (default 3.0).')
    parser_ACT.add_argument('--length_constraints',nargs=2,type=float,default=[0,0],metavar=('MIN','MAX'),
            help='Optional constraints for initial length of actuator (default=no constraints).')
    parser_ACT.add_argument('--width_constraints',nargs=2,type=float,default=[0,0],metavar=('MIN','MAX'),
            help='Optional constraints for initial width of actuator (default=no constraints).')
    parser_ACT.add_argument('--max_strain',type=float,help='Maximum strain (default=none).')
    # TODO - add option for random starting point.

    args = parser.parse_args()
    np.set_printoptions(precision=4,linewidth=130,suppress=True) # Disable scientific notation for small numbers.
    pyplot.rc('savefig',dpi=300)
    pyplot.rc('font',size=8)
    pyplot.rc('mathtext',default='regular') # Don't use italics for mathmode.

    #--------------------------------------------------------------------------------
    # Read in the given datasets.
    #--------------------------------------------------------------------------------
    if len(args.test)!=len(args.fittingdata):
        U.print_error('Number of fittingdata files must be equal to the number of tests.',True)
    print '--------------------------------------------------------------------'
    print ' Importing data file(s)...'
    data = dict()
    pressures = []
    for i,fname in enumerate(args.fittingdata):
        T = args.actuator + args.test[i]
        data[T] = np.loadtxt(fname,comments='#',delimiter=',')
        pressures.append(max(data[T][:,0]))
        print '  Imported',data[T].shape[0],'points from',fname,'for test type:',T
    if args.max_strain:
        data['strain'] = np.array([[0.0,0.0],[min(pressures),args.max_strain]])

    #--------------------------------------------------------------------------------
    # Prepare variables.
    #--------------------------------------------------------------------------------
    constraints = list()
    if args.cmd=='act':
        # Initial parameters.
        if args.actuator=='lin':
            parameter_descr = ['num_chambers', 'inlet_height', 'chamber_height', 'chamber_depth', 'wall_thickness']
            # Chambers and inlet will be square.
            popt = [args.num_chambers,  # 0 - Number of chambers.
                    args.inlet[0],      # 1 - Inlet height.
                    args.chamber[0],    # 2 - Chamber height.
                    args.chamber[2],    # 3 - Chamber depth.
                    args.wall,          # 4 - Wall thickness.
                    ]
            # Minimum dimensions, using abs because sometimes simulated annealing start points
            # may be negative, and it's difficult to get back to positives.
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[1])-0.5})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[2])-0.5})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[3])-1.0})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[4])-2.0})
            # Chambers must be larger than inlets.
            constraints.append({'type': 'ineq', 'fun': lambda P: P[2]-P[1]})
            # Length constraints.
            if np.diff(args.length_constraints)!=0.0:
                Lmax = lambda P: args.length_constraints[1] - calc_nc(P[0],popt[0])*P[3] - (calc_nc(P[0],popt[0])+1)*P[4]
                Lmin = lambda P: calc_nc(P[0],popt[0])*P[3] + (calc_nc(P[0],popt[0])+1)*P[4] - args.length_constraints[0]
                constraints.append({'type': 'ineq', 'fun': Lmax})
                constraints.append({'type': 'ineq', 'fun': Lmin})
            # Width constraints.
            if np.diff(args.width_constraints)!=0.0:
                Wmax = lambda P: args.width_constraints[1] - P[2] - 2.0*P[4]
                Wmin = lambda P: P[2] + 2.0*P[4] - args.width_constraints[0]
                constraints.append({'type': 'ineq', 'fun': Wmax})
                constraints.append({'type': 'ineq', 'fun': Wmin})
        elif args.actuator=='bnd':
            parameter_descr = ['num_chambers', 'inlet_height', 'inlet_width', 'chamber_height',
                               'chamber_width', 'chamber_depth', 'wall_thickness']
            # Chambers will not necessarily be square.
            popt = [args.num_chambers,  # 0 - Number of chambers.
                    args.inlet[0],      # 1 - Inlet height.
                    args.inlet[1],      # 2 - Inlet width.
                    args.chamber[0],    # 3 - Chamber height.
                    args.chamber[1],    # 4 - Chamber width.
                    args.chamber[2],    # 5 - Chamber depth.
                    args.wall,          # 6 - Wall thickness.
                    ]
            # Minimum dimensions, using abs because sometimes simulated annealing start points
            # may be negative, and it's difficult to get back to positives.
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[1])-0.5})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[2])-0.5})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[3])-0.5})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[4])-0.5})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[5])-1.0})
            constraints.append({'type': 'ineq', 'fun': lambda P: abs(P[6])-2.0})
            # Chambers must be larger than inlets.
            constraints.append({'type': 'ineq', 'fun': lambda P: P[3]-P[1]})
            constraints.append({'type': 'ineq', 'fun': lambda P: P[4]-P[2]})
            # Length constraints.
            if np.diff(args.length_constraints)!=0.0:
                Lmax = lambda P: args.length_constraints[1] - calc_nc(P[0],popt[0])*P[5] - (calc_nc(P[0],popt[0])+1)*P[6]
                Lmin = lambda P: calc_nc(P[0],popt[0])*P[5] + (calc_nc(P[0],popt[0])+1)*P[6] - args.length_constraints[0]
                constraints.append({'type': 'ineq', 'fun': Lmax})
                constraints.append({'type': 'ineq', 'fun': Lmin})
            # Width constraints.
            if np.diff(args.width_constraints)!=0.0:
                Wmax = lambda P: args.width_constraints[1] - P[4] - 2.0*P[6]
                Wmin = lambda P: P[4] + 2.0*P[6] - args.width_constraints[0]
                constraints.append({'type': 'ineq', 'fun': Wmax})
                constraints.append({'type': 'ineq', 'fun': Wmin})
        # Other variables.
        act = args.actuator
        matfile = path.abspath(args.matfile)
        script = path.abspath(U.find_file('python/abq_create_geom.py'))
        geomcmd = ['abaqus', 'cae', 'noGUI='+script,'--',
                   'test_geo-'+act+args.test+'.cae', 'noale', act,
                   args.test, args.mesh_size, '0.05',
                   str(args.time), str(args.num_chambers), '1wall',
                   str(args.inlet[0]), str(args.inlet[1]),
                   str(args.chamber[0]), str(args.chamber[1]), str(args.chamber[2]),
                   str(args.wall), '1.0', matfile, 'none',
                   str(args.dist_to_force), str(args.maxnuminc)]
        # This lambda function will end up taking simply the iteration and the optimization parameters.
        optfunc = lambda Rargs: run_multiACT(act,args.test,args.dist_to_force,geomcmd,pressures,
                                             args.num_chambers,*Rargs)

    #--------------------------------------------------------------------------------
    # Change directory.
    #--------------------------------------------------------------------------------
    if path.exists(args.dirname): U.print_error('Output directory exists.',True)
    os.makedirs(args.dirname)
    rootdir = os.getcwd()
    os.chdir(args.dirname)
    records_file = open('records.csv','w',1)    # 1 refers to a line-wise file buffer.
    records_file.write('# Parameter description: '+str(parameter_descr)+'\n')

    #--------------------------------------------------------------------------------
    # Calculate optimal parameters.
    #--------------------------------------------------------------------------------
    print '---------------------------------------------------------'
    print ' Calculating parameters...'
    print ' Parameters are: ',parameter_descr
    interpBOOL = not args.no_interpolate
    minfunc = lambda Largs: calc_fit(records_file,args.cmd,data,args.num_chambers,interpBOOL,optfunc,*Largs)
    if not args.basinhopping:
        maxitr = 20000
        OR = opt.minimize(minfunc,popt,method=args.method,options={'maxiter':maxitr},
                constraints=tuple(constraints))
    else:
        # Stepsize is the critical value here, setting the size of the random displacements
        # to search for new basins. Interval sets how often this is updated.
        basin_itr = 100
        minimizer_itr = 35 # We use a lower maxitr to avoid the long tail.
        kwargs = {'method':args.method,'constraints':tuple(constraints),'options':{'maxiter':minimizer_itr}}
        OR = opt.basinhopping(minfunc,popt,niter=basin_itr,minimizer_kwargs=kwargs,
                T=0.5,stepsize=5.0,interval=5)
    if not OR.success: print '\t\tWARNING: '+OR.message
    records_file.close()
    popt = np.abs(list(OR.x))
    if args.cmd=='act': popt[0] = calc_nc(popt[0],args.num_chambers)


    np.set_printoptions(suppress=False)
    print '\n\n--------------------------------------------------------------------'
    print ' Final results.'
    print '--------------------------------------------------------------------'
    print 'Parameter description:',parameter_descr
    print 'Parameters:\n',popt
    print 'TOTAL TIME ELAPSED: ',U.time_elapsed(tinit)
    os.chdir(rootdir)
