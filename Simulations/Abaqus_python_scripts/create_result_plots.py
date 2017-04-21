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

import argparse, sys, os, shutil, time
import os.path as path
import numpy as np
import utility
from matplotlib import pyplot,axes



#--------------------------------------------------------------------------------
# Create energy plots from extracted FEM data files, only called from run_tests.
#   axis      = axis handle of figure.
#   rpt       = rpt file to plot.
#--------------------------------------------------------------------------------
def plot_fem_energy(axis,rpt):
    try:
        data = np.loadtxt(rpt,skiprows=3)
    except:
        utility.print_error('Problem reading .rpt file: '+rpt,False)
        return
    if len(data.shape)==1:
        print '\tNot enough datapoints, skipping file.'
        return
    axis.plot(data[:,0], data[:,1], 'ko-', label='Internal Energy', markersize=3)
    axis.plot(data[:,0], data[:,2], 'rs-', label='Kinetic Energy', markersize=3)
    axis.set_xlabel('Time (meaningless units)')
    axis.set_ylabel('Energy')



#--------------------------------------------------------------------------------
# Create the plots from extracted FEM data files.
#   axis      = axis handle of figure.
#   rpt       = rpt file to plot.
#   test      = type of test to analyse.
#   color     = choose a color for all lines from this datafile.
#   labelpref = prefix for labels.
#   fulllabel = full label.
#   dist_to_force = distance to the blocked force plate, if applicable.
#--------------------------------------------------------------------------------
def plot_fem_data(axis,rpt,test='unknown',color='',labelpref='',fulllabel='',dist_to_force=0):
    # Load the file.
    try:
        data = np.loadtxt(rpt,skiprows=3)
    except:
        try:
            data = np.loadtxt(rpt,skiprows=4)
        except:
            utility.print_error('Problem reading .rpt file: '+rpt,False)
            return
    if len(data.shape)==1:
        print '\tNot enough datapoints, skipping file.'
        return

    # Get the initial length of the actuator.
    L = data[0,-1] - data[0,-2]
    if test=='unknown':
        # Columns are: X (time), Pressure, Fx, Fy, Fz, Zmin, Zmax
        #    or maybe: X (time), Pressure, Ux, Uy, Uz, Zmin, Zmax
        if data.shape[1]==7:
            labels = ('$X$','$Y$','$Z$')
            ylabel = 'Value - Blocked Force (N) or Displacement (mm)'
            title = 'Todo'
        # Columns are: X (time), Fz, Uz
        elif data.shape[1]==3:
            axis.set_xlabel('Displacement (mm)')
            axis.set_ylabel('Force (N/mm2)')
            axis.set_title('Displacement versus Force')
            axis.grid()
            axis.plot(data[:,2], data[:,1], label='$F_z$')
            return
        # Columns are X (time), Pressure, Ux, Uy, Uz
        elif data.shape[1]==5:
            labels = ('$X$','$Y$','$Z$')
            ylabel = 'Displacement (mm)'
            title = 'Bubble Displacement'
        else:
            utility.print_error('Didn\'t recognize data format for plotting.',False)
            print data
            return
        if axis.get_ylabel()=='':
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid()
        elif axis.get_ylabel()!=ylabel:
            utility.print_error('Multiple plot types. Plotting anyway...',False)
    else:
        if test=='linU':
            data[:,2] = -data[:,4]
            labels = ('$U_z$',)
        elif test=='linF':
            data[:,2] = data[:,4]
            labels = ('$F_z$',)
        elif test=='bndU':
            # Calculate the "bending angle". Assumes positive z-displacement is a shorter actuator.
            data[:,2] = np.degrees(np.arctan(-data[:,3]/(L-data[:,4])))
            data[data[:,2]<0.0,2] = data[data[:,2]<0.0,2] + 180
            labels = ('angle',)
        elif test=='bndF':
            alpha = np.radians(dist_to_force)
            beta = np.radians(90 - dist_to_force)
            f_normal = np.abs(data[:,3]*np.cos(alpha)) + np.abs(data[:,4]*np.cos(beta))
            data[:,2] = f_normal     # Pressure, Normal-Force.
            labels = ('$F_y$',)
        elif test=='bubU':
            data[:,2] = data[:,4]
            labels = ('$U_z$',)
        else:
            utility.print_error('Invalid test type in plot_fem_data.',True)
    # Set the plot properties and create the plot.
    # axis.set_xlabel(r'Internal Pressure $\left({}^N\!/{}_{mm^2}\right)$')
    data[:,1] = data[:,1] * 1000.0
    axis.set_xlabel('Internal Pressure (kPa)')
    linetypes = ['o-','s-','^-']
    for i in range(len(labels)):
        if not fulllabel:
            if labelpref=='': label = labels[i]
            else:             label = labelpref + ' - ' + labels[i]
        else:
            label = fulllabel
        if color=='':
            axis.plot(data[:,1], data[:,2+i], linetypes[i], label=label)
        else:
            axis.plot(data[:,1], data[:,2+i], linetypes[i], color=color, label=label, markersize=3, markeredgecolor=color)
            # ,linewidth=3)


#--------------------------------------------------------------------------------
# Create the plots from extracted experimental data files.
#   axis      = axis handle of figure.
#   fname     = file to plot.
#   test      = type of test to analyse.
#   color     = choose a color for all lines from this datafile.
#   labelpref = prefix for labels.
#   fulllabel = full label.
#--------------------------------------------------------------------------------
def plot_exp_data(axis, fname, test, color='', labelpref='', fulllabel=''):
    # Load the file.
    try:
        if   fname.endswith('.csv'): alldata = np.loadtxt(fname,delimiter=',')
        elif fname.endswith('.txt'): alldata = np.loadtxt(fname)
        else: raise('Unsupported file format.')
    except:
        utility.print_error('Problem reading experimental data file: '+fname,True)

    if test=='linU':
        try:
            T = path.splitext(fname)[0]
            T = float(T[T.rindex('-t')+2:])
            npress = alldata.shape[1]/2
            data = np.zeros([npress,2])
            for i in range(npress):
                # Note that we skip the first row, since it contains the pressure data.
                row = 0
                col = 2*i
                for r in range(alldata.shape[0]):
                    if alldata[r,col]>=T: break
                    row = r
                data[i,:] = alldata[0,col]
                data[i,1] = alldata[row,col+1]
                descr = '$U_z(t='+str(alldata[row,col])+'s)$'
        except Exception:
            utility.print_error('Problem determining cutoff time from filename.\n'
                    'Filename should be in this format: <fname>-t<time>.csv',False)
            print 'Assuming no cutoff, printing all data.'
            data = np.zeros([alldata.shape[0],2])
            data[:,0] = alldata[:,0]
            data[:,1] = alldata[:,1]
            descr = '$U_z$'
        line = 'o--'
    elif test=='linF' or test=='bndF':
        # Extract the unique pressures into a column of an array.
        pressures = np.unique(alldata[:,2])
        data = np.zeros((pressures.shape[0],3))
        data[:,0] = pressures
        # This extracts the last experimental datapoint at each pressure.
        for i in range(alldata.shape[0]):
            P = alldata[i,2]
            idx = np.where(pressures==P)
            data[idx,1] = alldata[i,1]
            data[idx,2] = alldata[i,0]
        # Legend label includes the time of the data sampled.
        # We take the average time, not including for the 0 pressure.
        tavg = np.around(np.average(data[1:,2]),2)
        descr = '$F_z(t='+str(tavg)+'s)$'
        line = 'o--'
    elif test=='bndU':
        data = np.zeros([alldata.shape[0],2])
        data[:,0] = alldata[:,0]
        data[:,1] = alldata[:,1]
        descr = 'angle'
        line = '--'
    elif test=='bubU':
        data = alldata
        descr = '$U_z$'
        line = '--'
    else:
        utility.print_error('Invalid test type in plot_exp_data.',True)

    # Set the plot properties and create the plot.
    # axis.set_xlabel(r'Internal Pressure $\left({}^N\!/{}_{mm^2}\right)$')
    data[:,0] = data[:,0] * 1000.0
    axis.set_xlabel('Internal Pressure (kPa)')
    if not fulllabel:
        if labelpref=='': label = labels[i]
        else:             label = labelpref + ' - ' + descr
    else:
        label = fulllabel
    if color=='': axis.plot(data[:,0], data[:,1], line, label=label)
    else:         axis.plot(data[:,0], data[:,1], line, color=color, label=label)
    # Print the array for use with brute-forcing.
    # np.set_printoptions(threshold='nan')
    # print data



#--------------------------------------------------------------------------------
# Main.
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    tinit = time.time()
    # Handle user input.
    parser = argparse.ArgumentParser(description='Create plots of simulation results (eg displacements or blocked forces) versus pressure. Image can be png or eps (or others).',
                                     epilog='Example: create_result_plots.py --sort --ylim 0 10 linU linear-U.png ./test_mat/*-U-*/data.rpt ../../data/*.csv')
    parser.add_argument('test',choices=['linF','linU','bndF','bndU','bubF','bubU','unknown'],help='Type of test to analyse.')
    parser.add_argument('img',help='Output image to create.')
    parser.add_argument('data',nargs='+',help='Data file(s) to plot. RPT files are assumed to be Abaqus FEM output, CSV or TXT files are assumed to be experimental data.')
    parser.add_argument('--sort',action='store_true',help='Numerically sort inputs from smallest to largest.')
    parser.add_argument('--sort_reverse',action='store_true',help='Numerically sort inputs from largest to smallest.')
    parser.add_argument('--xlim',nargs=2,type=float,metavar=('MIN','MAX'),help='Optional min and max for x-axis.')
    parser.add_argument('--ylim',nargs=2,type=float,metavar=('MIN','MAX'),help='Optional min and max for y-axis.')
    parser.add_argument('--title',help='Optional title for plot.')
    parser.add_argument("--paper",action='store_true',help="Create plots designed for the paper, rather than for general use.")
    parser.add_argument("--labels",nargs='+',help="Labels for lines (all FEM labels first, then experimental labels).")
    parser.add_argument('--dist_to_force',type=float,default=0.0,help='Distance (or angle for bending actuators) to the blocked force plate in blocked-force tests (default 0.0).')
    args = parser.parse_args()
    # pyplot.rc('savefig',dpi=300)
    # pyplot.rc('font',size=8)
    pyplot.rc('mathtext',default='regular') # Don't use italics for mathmode.
    if args.labels and len(args.labels)<len(args.data):
        utility.print_error('Number of given labels must be the same as the number of given files.',True)


    # Separate the data files into FEM and EXP.
    femdata = []
    expdata = []
    for df in args.data:
        if df.endswith('.rpt'): femdata.append(df)
        else:                   expdata.append(df)

    # Try to sort the data.
    if args.sort or args.sort_reverse:
        try:
            from natsort import natsort
            femdata = natsort(femdata)
            expdata = natsort(expdata)
            if args.sort_reverse:
                femdata = list(reversed(femdata))
                expdata = list(reversed(expdata))
        except:
            print 'WARNING: no natsort module found, sorting not available.'

    # Prepare the plots.
    fig,ax = pyplot.subplots()
    cm = pyplot.get_cmap('gist_rainbow')
    # cm = pyplot.get_cmap('jet')
    if len(femdata)==len(expdata):
        colors = [cm(float(i)/len(femdata)) for i in range(len(femdata))]
    else:
        colors = [cm(float(i)/len(args.data)) for i in range(len(args.data))]

    # TODO - remove longest common prefix from label strings.
    # Plot the FEM data.
    for i in range(len(femdata)):
        print 'Reading FEM data for:',femdata[i]
        if args.labels: plot_fem_data(ax, femdata[i], args.test, colors[i], '', args.labels[i], args.dist_to_force)
        else:           plot_fem_data(ax, femdata[i], args.test, colors[i], femdata[i], args.dist_to_force)
    # Plot the EXP data.
    for i in range(len(expdata)):
        print 'Reading EXP data for:',expdata[i]
        if len(femdata)==len(expdata): color=colors[i]
        else:                          color=colors[len(femdata)+i]
        if args.labels: plot_exp_data(ax, expdata[i], args.test, color, '', args.labels[i+len(femdata)])
        else:           plot_exp_data(ax, expdata[i], args.test, color, path.splitext(expdata[i])[0])

    # Write the plot to disk.
    if args.test=='linU':
        ylabel = 'Displacement (mm)'
        title = 'Displacement vs. Pressure'
    elif args.test=='linF':
        ylabel = 'Blocked Force (N)'
        title = 'Blocked Force vs. Pressure'
    elif args.test=='bndU':
        ylabel = 'Bending Angle ($^\circ$)'
        title = 'Bending Angle vs. Pressure'
    elif args.test=='bndF':
        ylabel = 'Blocked Force (N)'
        title = 'Blocked Force vs. Pressure'
    elif args.test=='bubU':
        ylabel = 'Displacement (mm)'
        title = 'Displacement vs. Pressure'
    ax.set_ylabel(ylabel)
    ax.grid()
    if args.xlim: ax.set_xlim(args.xlim)
    if args.ylim: ax.set_ylim(args.ylim)
    if args.paper:
        lgd = ax.legend(loc='upper left',frameon=False,framealpha=0)
        for legobj in lgd.legendHandles: legobj.set_linewidth(2.0)
    else:
        if args.title: ax.set_title(args.title)
        else:          ax.set_title(title)
        lgd = ax.legend(loc=2,bbox_to_anchor=(1,1),frameon=False,framealpha=0)
        for legobj in lgd.legendHandles: legobj.set_linewidth(2.0)
    print 'Writing plot to:',args.img
    pyplot.savefig(args.img,bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyplot.close()
