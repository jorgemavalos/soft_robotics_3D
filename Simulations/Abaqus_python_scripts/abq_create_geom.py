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

# Use this script to create Abaqus geometry for a SPA.
# Note that you probably never want to run this script directly;
# instead prefer to use the create_geom.py wrapper script.
# Run using: abaqus cae noGUI=abq_create_geom.py -- --help
#            Output will be written to the abaqus.rpy file.
# Note that if you're editing this file, you may want to run the following command in ABQ
# in order to get more useful history in the rpy file:
#   session.journalOptions.setValues(replayGeometry=COORDINATE,recoverGeometry=COORDINATE)

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import sys, pprint, os, shutil
sys.path.append('/usr/lib/python2.7/')
import argparse
import inspect, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import model_utility as MU


#--------------------------------------------------------------------------------
# Return the model given a string model name.
#--------------------------------------------------------------------------------
def get_abq_model(mname):
    mn = mname.lower()
    if mn=='ab':    return ARRUDA_BOYCE
    if mn=='mr':    return MOONEY_RIVLIN
    if mn=='neoh':  return NEO_HOOKE
    if mn=='ogden': return OGDEN
    if mn=='poly':  return POLYNOMIAL
    if mn=='rpoly': return REDUCED_POLYNOMIAL
    if mn=='vdw':   return VAN_DER_WAALS
    if mn=='yeoh':  return YEOH
    print 'ERROR: invalid model name \''+mname+'\''
    exit(1)

#--------------------------------------------------------------------------------
# Main.
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    # Handle user input. We can't do any optional arguments, because we'd have
    # to find them in the sys.argv, which includes a bunch of Abaqus junk.
    # Also, Abaqus won't let us pass "--xxx=xxx" style arguments anyway.
    NARGS = 20
    run_cmd = 'abaqus cae noGUI=abq_create_geom.py -- '
    if '--help' in sys.argv: sys.argv = [run_cmd,'--help']
    else: sys.argv = [run_cmd]+sys.argv[len(sys.argv)-NARGS:]
    parser = argparse.ArgumentParser(
            description='Create Abaqus geometry for a SPA.',
            epilog='Example: '+run_cmd+'problem1.cae ale lin U 2.0 0.05 200 4 1wall 2.0 2.0 6.0 6.0 2.0 3.0 1.0 ecoflex30.mat visfile.mat 0.0')
    parser.add_argument('cae',help='Output Abaqus database file to create.')
    parser.add_argument('ale',choices=['ale','noale'],help='Whether or not to create an ALE mesh.')
    parser.add_argument('actuator',choices=['lin','bnd'],help='Type of actuator to create.')
    parser.add_argument('test',choices=['U','F'],help='Type of test to run.')
    parser.add_argument('mesh_size',type=float,help='Approximate size of mesh elements.')
    parser.add_argument('pressure',type=float,help='Pressure in chamber, probably in N/mm2. 0.1=100kPa. If not converging, try really small (0.01) values here.')
    parser.add_argument('time',type=float,help='Time to end simulation (s).')
    parser.add_argument('num_chambers',type=int,help='Number of chambers.')
    parser.add_argument('cspace',choices=['1wall','2wall'],help='Spacing between chambers.')
    parser.add_argument('inlet_H',type=float,help='Height of inlet tunnel (divided by two for linear actuators).')
    parser.add_argument('inlet_W',type=float,help='Width of inlet tunnel (will be divided by two for symmetry).')
    parser.add_argument('chamber_H',type=float,help='Height of chamber (divided by two for linear actuators).')
    parser.add_argument('chamber_W',type=float,help='Width of chamber (will be divided by two for symmetry).')
    parser.add_argument('chamber_L',type=float,help='Length of chamber.')
    parser.add_argument('wall',type=float,help='Wall thickness.')
    parser.add_argument('mass_scaling',type=float,help='Mass scaling factor (1.0 to disable).')
    parser.add_argument('matfile',help='File containing material properties.')
    parser.add_argument('visfile',help='File containing viscoelastic material properties (or \'none\'.')
    parser.add_argument('dist_to_force',type=float,help='Distance to blocked force plate in blocked-force tests.')
    parser.add_argument('maxnuminc',type=int,help='Maximum number of increments.')
    args = parser.parse_args()
    if not args.cae.endswith('.cae'):
        print 'WARNING: incorrect inputs. Found arguments:'
        print sys.argv
        exit(1)
    #--------------------------------------------------------------------------------
    # Read the material file.
    #--------------------------------------------------------------------------------
    mat = MU.read_matfile(args.matfile)
    vis = MU.read_viscomatfile(args.visfile) if args.visfile!='none' else dict()

    #--------------------------------------------------------------------------------
    # Calculate dimensions, assuming mirror-symmetry in x.
    #   height = Y dimension.
    #   width  = X dimension.
    #   length = Z dimension.
    #--------------------------------------------------------------------------------
    quadelms = True                                                     # Always use quadratic elements.
    ALE = True if args.ale=='ale' else False                            # Arbitrary Lagrangian-Eulerian.
    nc = args.num_chambers                                              # Number of air chambers.
    tube_H = 2.0 / (2.0 if args.actuator=='lin' else 1.0)               # Height of inlet tube (symmetry if lin).
    tube_W = 2.0 / 2.0                                                  # Width of inlet tube (always symmetry).
    inlet_H = args.inlet_H / (2.0 if args.actuator=='lin' else 1.0)     # Height of inlet tunnel between chambers (symmetry if lin).
    inlet_W = args.inlet_W / 2.0                                        # Width of inlet tunnel between chambers (always symmetry).
    chamber_H = args.chamber_H / (2.0 if args.actuator=='lin' else 1.0) # Height of chamber (symmetry if lin).
    chamber_W = args.chamber_W / 2.0                                    # Width of chamber (always symmetry).
    chamber_L = args.chamber_L                                          # Depth of chamber.
    wall = args.wall                                                    # Thickness of chamber walls.
    cwall = 2.0*wall if args.cspace=='2wall' else wall                  # Wall thickness between chambers.
    # offset = chamber_L+wall if args.actuator=='bnd' and args.test=='F' else 0.0   # Constrained region for bndF actuators.
    offset = 5.0 if args.actuator=='bnd' and args.test=='F' else 0.0   # Constrained region for bndF actuators.
    body_H = wall + chamber_H                                           # Height of cross-section.
    body_W = wall + chamber_W                                           # Width of cross-section.
    body_L = nc*chamber_L + 2.0*wall + (nc-1.0)*cwall + offset          # Length of SPA.
    time = args.time
    mass_scaling = args.mass_scaling
    dist_to_force = args.dist_to_force
    film_thickness = 0.5


    #--------------------------------------------------------------------------------
    # Open the viewport.
    #--------------------------------------------------------------------------------
    session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=300, height=200)
    executeOnCaeStartup()
    Mdb()

    #--------------------------------------------------------------------------------
    # Create the part.
    #--------------------------------------------------------------------------------
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__',sheetSize=200.0)
    s.setPrimaryObject(option=STANDALONE)

    # Extrude a solid rectangle for body (assuming mirror-symmetry in x).
    #   O = origin
    #   H = body_H
    #   W = body_W
    #   L = body_L
    #
    #               +-----W-----+
    #             /           / |
    #           L           L   H
    #         /           /     |
    #       +-----W-----+       +
    #       |           |     /     Y
    #       |           H   L       ^    -Z
    # SYMM  INLET       | /         |   /
    #       O- - - - - -+           | /
    #           SYMM                O-----> X
    #
    s.rectangle(point1=(0.0, 0.0), point2=(body_W, body_H))
    p = mdb.models['Model-1'].Part(name='SPA', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['SPA']
    p.BaseSolidExtrude(sketch=s, depth=body_L)
    s.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']

    # Extrude a cut for the inlet tunnel.
    #   O = origin
    #   H = body_H      h = inlet_H
    #   L = body_L      l = wall
    #
    #         /                      /
    #       +----------L-----------+
    #       |                      |
    #       |                      | /
    #       |                      |
    #       H            +----l--- H
    #       |            h   CUT   |
    #       |            |         |
    #       +- - - - - - - - - - - O
    #
    # Cut for inlet tube (where the tube is actually inserted).
    f, e = p.faces, p.edges
    face = f.findAt(((0.0, 0.1, 0.1),),)[0]
    edge = e.findAt(((0.0, body_H-0.1, body_L),),)[0]
    t = p.MakeSketchTransform(sketchPlane=face, sketchUpEdge=edge,
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, body_L))
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=45.6, gridSpacing=1.14, transform=t)
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.rectangle(point1=(0.0, 0.0), point2=(-wall-offset, tube_H))
    p.CutExtrude(sketchPlane=face, sketchUpEdge=edge, sketchPlaneSide=SIDE1,
        sketchOrientation=RIGHT, sketch=s1, depth=tube_W, flipExtrudeDirection=OFF)
    s1.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']
    # Cut for air inlet tunnel, which runs between chambers.
    face = f.findAt(((0.0, 0.1, 0.1),),)[0]
    edge = e.findAt(((0.0, body_H-0.1, body_L),),)[0]
    t = p.MakeSketchTransform(sketchPlane=face, sketchUpEdge=edge,
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, body_L))
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=45.6, gridSpacing=1.14, transform=t)
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.rectangle(point1=(-wall-offset, 0.0), point2=(-body_L+wall, inlet_H))
    p.CutExtrude(sketchPlane=face, sketchUpEdge=edge, sketchPlaneSide=SIDE1,
        sketchOrientation=RIGHT, sketch=s1, depth=inlet_W, flipExtrudeDirection=OFF)
    s1.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']

    # Extrude cuts for the air chambers.
    #   O = sketch origin   @ = global origin
    #   H = body_H          h = inlet_H         x = chamber_L
    #   L = body_L          l = wall            y = chamber_H
    #
    #         /                      /
    #       +----------L-----------+
    #       |                      |
    #       |        +-x-+         | /
    #       |        |   |         |            Y               Y
    #       |        | C +----l--- |            ^     X         ^
    #       H        y U h         H            |   /           |
    #       |        | T |         |            | /             |
    #       @- - - - + - + - - - - O            @-----> Z       O----> X
    #
    face = f.findAt(((0.0, 0.1, 0.1),),)[0]
    edge = e.findAt(((0.0, body_H-0.1, body_L),),)[0]
    t = p.MakeSketchTransform(sketchPlane=face, sketchUpEdge=edge,
        sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, body_L))
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=45.6, gridSpacing=1.14, transform=t)
    s.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
    for i in range(nc):
        xoff = offset + wall + i*(cwall+chamber_L)
        s.rectangle(point1=(-xoff,0.0), point2=(-xoff-chamber_L, chamber_H))

    p.CutExtrude(sketchPlane=face, sketchUpEdge=edge, sketchPlaneSide=SIDE1,
        sketchOrientation=RIGHT, sketch=s, depth=chamber_W, flipExtrudeDirection=OFF)
    s.unsetPrimaryObject()
    del mdb.models['Model-1'].sketches['__profile__']

    # Create a film to cover exposed areas for bending actuators.
    if args.actuator=='bnd':
        face = f.findAt(coordinates=(0.1, 0.1, 0.0))
        edge = e.findAt(coordinates=(0.0, 0.1, 0.0))
        t = p.MakeSketchTransform(sketchPlane=face, sketchUpEdge=edge,
            sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(0.0, 0.0, 0.0))
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=8.48, gridSpacing=0.21, transform=t)
        s.setPrimaryObject(option=SUPERIMPOSE)
        p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
        s.Line(point1=(-chamber_W, 0.0), point2=(0.0, 0.0))
        p.ShellExtrude(sketchPlane=face, sketchUpEdge=edge, sketchPlaneSide=SIDE1,
                sketchOrientation=RIGHT, sketch=s, depth=body_L, flipExtrudeDirection=ON)
        del mdb.models['Model-1'].sketches['__profile__']

    #--------------------------------------------------------------------------------
    # Define some variables.
    #--------------------------------------------------------------------------------
    d = p.datums
    c = p.cells
    f = p.faces
    e = p.edges
    v = p.vertices

    #--------------------------------------------------------------------------------
    # Create sets and surfaces. See diagram above for global origin and directions.
    #--------------------------------------------------------------------------------
    p.Set(cells=c, name='ALL')
    # Symmetry boundary condition in width.
    faces = f.findAt(((0.0, 0.1, 0.1),),)
    p.Set(faces=faces, name='SYMM_W')
    # Symmetry boundary condition in height (for linear actuators).
    if args.actuator=='lin':
        faces = f.findAt(((0.1, 0.0, 0.1),),)
        p.Set(faces=faces, name='SYMM_H')
    # Pressure set, for measuring purposes.
    zoff = body_L-offset-wall-(nc-1)*(cwall+chamber_L)
    faces = f.findAt(((chamber_W-0.1, chamber_H-0.1, zoff-chamber_L),),) # chamber-left
    p.Set(faces=faces, name='PRESSURE')
    if args.test=='U':
        # Surface for measuring displacements (same for linear or bending).
        faces = f.findAt(((0.1, 0.1, 0.0),),)   # back side.
        p.Surface(side1Faces=faces, name='MEASURED')
    elif args.test=='F':
        # Surface for blocked force contact.
        if args.actuator=='lin':
            faces = f.findAt(((0.1, 0.1, 0.0),),)   # left side.
            p.Surface(side1Faces=faces, name='BLOCKED')
        if args.actuator=='bnd':
            faces1 = f.findAt(((0.1, 0.1, 0.0), ))     # left side.
                              # ((0.1, 0.0, 0.1), ))     # bottom side.
            # faces2 = f.findAt(((0.1, 0.0, wall+0.1),)) # membrane.
            # p.Surface(side1Faces=faces1, side2Faces=faces2, name='BLOCKED')
            p.Surface(side1Faces=faces1, name='BLOCKED')

    #--------------------------------------------------------------------------------
    # Create some partitions.
    #--------------------------------------------------------------------------------
    # Split everything into 2 large sections; top, middle.
    pickedCells = c.findAt(((0.1,0.1,0.1),))
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=chamber_H)
    p.PartitionCellByDatumPlane(datumPlane=d.values()[-1], cells=pickedCells)
    # Split the middle section along the back of the chambers.
    pickedCells = c.findAt(((0.1,0.1,0.1), ))
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=chamber_W)
    p.PartitionCellByDatumPlane(datumPlane=d.values()[-1], cells=pickedCells)
    # Partition the membrane for the tube BC, if necessary.
    if args.actuator=='bnd':
        if args.test=='U':   cut = body_L-wall
        elif args.test=='F': cut = body_L-offset
        pickedFaces = f.findAt(((0.1, 0.0, wall+0.1), ))    # Membrane.
        p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=cut)
        p.PartitionFaceByDatumPlane(datumPlane=d.values()[-1], faces=pickedFaces)

    #--------------------------------------------------------------------------------
    # Additional sets and surfaces after partitioning.
    #--------------------------------------------------------------------------------
    # Tube (fixed) boundary condition.
    faces = f.findAt(((body_W-0.1, 0.1,           body_L),),        # external face 1
                     ((tube_W+0.1, 0.1,           body_L),),        # external face 2
                     ((0.1,        chamber_H+0.1, body_L),),)       # external face 3
    if args.actuator=='lin':
        faces = faces + f.findAt(((tube_W/2.0, tube_H, body_L-0.1),),    # inlet-top
                                 ((tube_W,     0.1,    body_L-0.1),))    # inlet-side
    p.Set(faces=faces, name='TUBE')

    # Pressure boundary condition, in chambers and connecting tubes.
    for i in range(nc):
        zoff = body_L-offset-wall-i*(cwall+chamber_L)
        faces = f.findAt(((chamber_W-0.1, chamber_H,     zoff-0.1),),        # chamber-top
                         ((chamber_W-0.1, chamber_H-0.1, zoff-chamber_L),),  # chamber-left
                         ((chamber_W-0.1, chamber_H-0.1, zoff),),            # chamber-right
                         ((chamber_W,     chamber_H-0.1, zoff-0.1),),)       # chamber-back
        if i==0: allfaces = faces
        else:
            faces2 = f.findAt(((0.1,     inlet_H,     zoff+0.1),),           # inlet-top
                              ((inlet_W, inlet_H-0.1, zoff+0.1),),)          # inlet-side
            allfaces = allfaces + faces + faces2
    p.Surface(side1Faces=allfaces, name='PRESSURE')

    # If it's a actuator experiencing blocked force, set additional BCs.
    if args.test=='F':
        if args.actuator=='lin':
            # linF test has cups on each end of the actuator that extend 5mm along the length,
            # constraining expansion in the off-axis directions.
            p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=5.0)
            p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=body_L-5.0)
            pickedCells = c.findAt(((0.1,        body_H-0.1, body_L/2.0), ),     # Top block.
                                   ((body_W-0.1, 0.1,        body_L/2.0), ))     # Back block (behind chambers).
            p.PartitionCellByDatumPlane(datumPlane=d.values()[-1], cells=pickedCells)
            pickedCells = c.findAt(((0.1,        body_H-0.1, body_L/2.0), ),     # Top block.
                                   ((body_W-0.1, 0.1,        body_L/2.0), ))     # Back block (behind chambers).
            p.PartitionCellByDatumPlane(datumPlane=d.values()[-2], cells=pickedCells)
            # Create a single set for both cups.
            if dist_to_force==0.0:
                faces = f.findAt(((0.1,    body_H,     0.1),),          # Left Cup - Top.
                                 ((body_W, body_H-0.1, 0.1),),          # Left Cup - Back Top.
                                 ((body_W, 0.1,        0.1),),          # Left Cup - Back Bottom.
                                 ((0.1,    body_H,     body_L-0.1),),   # Right Cup - Top.
                                 ((body_W, body_H-0.1, body_L-0.1),),   # Right Cup - Back Top.
                                 ((body_W, 0.1,        body_L-0.1),),)  # Right Cup - Back Bottom.
            else:
                faces = f.findAt(((0.1,    body_H,     body_L-0.1),),   # Right Cup - Top.
                                 ((body_W, body_H-0.1, body_L-0.1),),   # Right Cup - Back Top.
                                 ((body_W, 0.1,        body_L-0.1),),)  # Right Cup - Back Bottom.
            p.Set(faces=faces, name='BC_CUPS')
        elif args.actuator=='bnd':
            # bndF test has a large cup on the inlet end that extends "offset" along the length,
            # constraining expansion in the off-axis directions.
            p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=body_L-offset)
            pickedCells = c.findAt(((0.1,        body_H-0.1, body_L/2.0), ),    # Top block.
                                   ((0.1,        tube_H+0.1, body_L-0.1), ),    # Chamber block.
                                   ((body_W-0.1, 0.1,        body_L/2.0), ))    # Back block (behind chambers).
            p.PartitionCellByDatumPlane(datumPlane=d.values()[-1], cells=pickedCells)
            # Create another partition for contact with the cup, if necessary.
            p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=0.75*(body_L-offset))
            pickedCells = c.findAt(((0.1,        body_H-0.1, body_L-offset-0.1), ),    # Top block.
                                   ((body_W-0.1, 0.1,        body_L-offset-0.1), ))    # Back block (behind chambers).
            p.PartitionCellByDatumPlane(datumPlane=d.values()[-1], cells=pickedCells)
            # Create a set for the cup.
            faces = f.findAt(((0.1,        body_H,     body_L-0.1),),   # Top.
                             ((body_W,     body_H-0.1, body_L-0.1),),   # Back Top.
                             ((body_W,     0.1,        body_L-0.1),),   # Back Bottom.
                             ((0.1,        0.0,        body_L-0.1),),   # Membrane.
                             ((body_W-0.1, 0.0,        body_L-0.1),),   # Bottom Back.
                             ((tube_W+0.1, 0.0,        body_L-0.1),),)  # Bottom Middle.
            p.Set(faces=faces, name='BC_CUPS')
            # Create a set for contact with the cup.
            faces = f.findAt(((0.1,    body_H,     body_L-offset-0.1), ),  # top.
                             ((body_W, body_H-0.1, body_L-offset-0.1), ),  # back top.
                             ((body_W, 0.1,        body_L-offset-0.1), ))  # back bottom.
            p.Surface(side1Faces=faces, name='CUP_CONTACT')

    # For a bending actuator, also apply the pressure to the membrane surface.
    # We have this split apart from the first pressure definition because otherwise
    # we can split chambers in half and pressure might only be applied to one half.
    if args.actuator=='bnd':
        allfaces = p.surfaces['PRESSURE'].faces
        allfaces = allfaces + f.findAt(((0.1, 0.0, wall+0.1),))              # membrane
        if args.test=='F':
            faces = f.findAt(((0.1,    tube_H,     body_L-offset-0.1),),   # inlet-top
                             ((tube_W, tube_H/2.0, body_L-offset-0.1),),)  # inlet-side
            allfaces = allfaces + faces
        p.Surface(side1Faces=allfaces, name='PRESSURE')


    #================================================================================
    # If this is for a bending SPA, add a bending section with silk properties.
    # The values to mess with here are Young's Modulus and thickness.
    #================================================================================
    if args.actuator=='bnd':
        # Create a set for the symmetry condition on the silk film.
        edges = e.findAt(((0.0, 0.0, wall+0.1),), ((0.0, 0.0, body_L-0.1),))
        p.Set(edges=edges, name='SYMM_SHELL')
        # Create the material.
        mdb.models['Model-1'].Material(name='SILK')
        mdb.models['Model-1'].materials['SILK'].Density(table=((mass_scaling*0.0013, ), ))
        mdb.models['Model-1'].materials['SILK'].Elastic(table=((4100.0, 0.49), ))
        # Create a material section. Note that increasing the thickness will increase
        # the tension resistance, but will take longer to converge.
        mdb.models['Model-1'].MembraneSection(name='SILK', material='SILK', thicknessType=UNIFORM,
                thickness=film_thickness, thicknessField='', poissonDefinition=DEFAULT)

        # Assign the material section to the model.
        faces = f.findAt(((body_W-0.1, 0.0, 0.1),),         # Back wall.
                         ((0.1,        0.0, 0.1),),)        # End cap.
        for i in range(nc):
            zoff = body_L-offset-wall-i*(cwall+chamber_L)
            newfaces = f.findAt(((inlet_W+0.1, 0.0, zoff+0.1),),)
            faces = faces + newfaces
        p.Skin(faces=faces, name='SILK_SKIN')       # Surfaces.
        region1 = regionToolset.Region(skinFaces=(('SILK_SKIN', faces), ))
        p.SectionAssignment(region=region1, sectionName='SILK', offset=0.0,
            offsetType=TOP_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

        faces = f.findAt(((0.1, 0.0, wall+0.1),),   # Gaps covered with extruded shell.
                         ((0.1, 0.0, body_L-0.1),)) # Tube gap.
        region2 = regionToolset.Region(faces=faces)
        p.SectionAssignment(region=region2, sectionName='SILK', offset=0.0,
            offsetType=TOP_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

        # Assign the element type.
        if quadelms:
            elemType1 = mesh.ElemType(elemCode=M3D8R, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=M3D6, elemLibrary=STANDARD)
        else:
            elemType1 = mesh.ElemType(elemCode=M3D4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
            elemType2 = mesh.ElemType(elemCode=M3D3, elemLibrary=STANDARD)
        p.setElementType(regions=region1, elemTypes=(elemType1, elemType2))
        p.setElementType(regions=region2, elemTypes=(elemType1, elemType2))


    #--------------------------------------------------------------------------------
    # Mesh the geometry.
    #--------------------------------------------------------------------------------
    if quadelms: # Quadratic elements. These provide a much faster rate of convergence than linear elements.
        elemType1 = mesh.ElemType(elemCode=C3D20RH, elemLibrary=STANDARD)
        elemType2 = mesh.ElemType(elemCode=C3D15, elemLibrary=STANDARD)
        elemType3 = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD)
    else:        # Linear elements.
        elemType1 = mesh.ElemType(elemCode=C3D8RH, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN, hourglassControl=DEFAULT)
        elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
        elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)
    region = p.sets['ALL']
    p.setElementType(regions=region, elemTypes=(elemType1, elemType2, elemType3))
    p.seedPart(size=args.mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    p.generateMesh()

    #--------------------------------------------------------------------------------
    # Set the output frequency.
    #--------------------------------------------------------------------------------
    num_nodes = len(p.nodes)
    output_frequency = 1 if args.actuator=='lin' and num_nodes<100000 else 20

    #--------------------------------------------------------------------------------
    # Create and assign the material.
    #--------------------------------------------------------------------------------
    if mat['density']==0.0: print 'WARNING: material density equals 0.0'
    if mat['matname']=='???': print 'WARNING: material name is given as',mat['matname']
    mdb.models['Model-1'].Material(name=mat['matname'])
    mdb.models['Model-1'].materials[mat['matname']].Density(table=((mass_scaling*mat['density'], ), ))
    allparams = mat['params']+mat['D']
    if 'order' in mat:
        mdb.models['Model-1'].materials[mat['matname']].Hyperelastic(
            materialType=ISOTROPIC, testData=OFF, volumetricResponse=VOLUMETRIC_DATA,
            type=get_abq_model(mat['model']), n=mat['order'], table=(allparams, ),
            moduliTimeScale=LONG_TERM)
            # moduliTimeScale=INSTANTANEOUS)
    else:
        mdb.models['Model-1'].materials[mat['matname']].Hyperelastic(
            materialType=ISOTROPIC, testData=OFF, volumetricResponse=VOLUMETRIC_DATA,
            type=get_abq_model(mat['model']), table=(allparams, ),
            moduliTimeScale=LONG_TERM)
            # moduliTimeScale=INSTANTANEOUS)
    mdb.models['Model-1'].HomogeneousSolidSection(name='SPA',material=mat['matname'],thickness=None)
    region = p.sets['ALL']
    p.SectionAssignment(region=region, sectionName='SPA', offset=0.0,
        offsetType=MIDDLE_SURFACE, offsetField='',thicknessAssignment=FROM_SECTION)
    # Support viscoelastic material properties.
    if not len(vis)==0:
        vparams = np.zeros((len(vis['tau']),3))
        if 'g' in vis: vparams[:,0] = vis['g']
        if 'k' in vis: vparams[:,1] = vis['k']
        vparams[:,2] = vis['tau']
        mdb.models['Model-1'].materials[mat['matname']].Viscoelastic(
            domain=TIME, time=PRONY, table=tuple(vparams))
    # Support hysteretic properties.
    if 'hyst_S' in mat:
        mdb.models['Model-1'].materials[mat['matname']].hyperelastic.Hysteresis(
            table=((mat['hyst_S'], mat['hyst_A'], mat['hyst_m'], mat['hyst_C']), ))


    #--------------------------------------------------------------------------------
    # Create the assembly.
    #--------------------------------------------------------------------------------
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    a.Instance(name='SPA', part=p, dependent=ON)

    #--------------------------------------------------------------------------------
    # Create a step.
    #--------------------------------------------------------------------------------
    # inc = 0.01*time
    inc = min(15,0.1*time)
    if len(vis)==0:         # No viscoelasticity.
        # Defaults: initialInc=1.0, maxNumInc=100, minInc=1e-5, maxInc=1.0
        # mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial', timePeriod=time,
                # nlgeom=ON, initialInc=inc, maxNumInc=args.maxnuminc, minInc=1e-06, maxInc=inc,
                # solutionTechnique=QUASI_NEWTON,
                # extrapolation=PARABOLIC)
        # Enable stabilization.
        # mdb.models['Model-1'].steps['Step-1'].setValues(stabilizationMagnitude=0.0002,
            # stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
            # continueDampingFactors=False, adaptiveDampingRatio=0.05)
        # mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=time)
        # For bending actuators with membrane elements we use an implicit dynamics step.
        # Unloaded membrane elements can rapidly become unstable in a static analysis
        # because they have no bending stiffness.
        # For linear actuators, they also let us converge at higher pressures.
        if args.actuator=='bnd': init_inc = inc/10.0
        else: init_inc = inc
        mdb.models['Model-1'].ImplicitDynamicsStep(name='Step-1', previous='Initial',
            timePeriod=time, application=QUASI_STATIC, initialInc=init_inc, minInc=1e-08, maxInc=inc,
            maxNumInc=args.maxnuminc, nohaf=OFF, amplitude=RAMP, alpha=DEFAULT, initialConditions=OFF, nlgeom=ON)
            # matrixStorage=UNSYMMETRIC,
            # solutionTechnique=QUASI_NEWTON,
        if args.actuator=='bnd':
            # mdb.models['Model-1'].materials[mat['matname']].Damping(alpha=0.0001*time, beta=0.0)
            print 'No damping.'
        else:
            mdb.models['Model-1'].materials[mat['matname']].Damping(alpha=0.002*time, beta=0.0)
    else:                   # Viscoelasticity
        # For viscoelastic solutions, we need to solve for the long-term solution.
        mdb.models['Model-1'].ViscoStep(name='Step-1', previous='Initial',
                timePeriod=time, cetol=1.00, extrapolation=PARABOLIC,
                nlgeom=ON,initialInc=inc, maxNumInc=500, minInc=1e-06, maxInc=inc)
        # mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=time)
    # Enable line search for Quasi-Newton algorithm, see manual section 7.2.2
    # mdb.models['Model-1'].steps['Step-1'].control.setValues(allowPropagation=OFF,
            # resetDefaultValues=OFF, lineSearch=(5.0, 1.0, 0.0001, 0.25, 0.1))
    # Assign the ALE mesh if requested.
    if ALE:
        region=a.instances['SPA'].sets['ALL']
        mdb.models['Model-1'].steps['Step-1'].AdaptiveMeshDomain(region=region,controls=None,
            frequency=100,meshSweeps=25)   # Default = 10,1

    #--------------------------------------------------------------------------------
    # Create the loads and boundary conditions.
    #--------------------------------------------------------------------------------
    region = a.instances['SPA'].sets['SYMM_W']
    mdb.models['Model-1'].XsymmBC(name='SYMM_W',createStepName='Step-1',region=region,localCsys=None)
    if args.actuator=='lin':
        region = a.instances['SPA'].sets['SYMM_H']
        mdb.models['Model-1'].YsymmBC(name='SYMM_H',createStepName='Step-1',region=region,localCsys=None)
    elif args.actuator=='bnd':
        region = a.instances['SPA'].sets['SYMM_SHELL']
        mdb.models['Model-1'].XsymmBC(name='SYMM_SHELL',createStepName='Step-1',region=region,localCsys=None)
    region = a.instances['SPA'].sets['TUBE']
    mdb.models['Model-1'].DisplacementBC(name='TUBE', createStepName='Initial',
        region=region, u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
        amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
    region = a.instances['SPA'].surfaces['PRESSURE']
    if len(vis)==0:
        # For a static simulation, it's fine to ramp up linearly over the entire timestep.
        mdb.models['Model-1'].Pressure(name='PRESSURE', createStepName='Step-1',
            region=region, distributionType=UNIFORM, field='', magnitude=args.pressure,
            amplitude=UNSET)
    else:
        # For a viscoelastic step, ramp up the pressure suddenly, then hold steady.
        # TODO - choose proper rampup time.
        mdb.models['Model-1'].TabularAmplitude(name='RAMPHOLD', timeSpan=STEP,
                smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (0.1, 1.0)))
                # smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (0.09, 1.1), (0.1, 1.0)))    # This enables a 10% overshoot.
        mdb.models['Model-1'].Pressure(name='PRESSURE', createStepName='Step-1',
            region=region, distributionType=UNIFORM, field='', magnitude=args.pressure,
            amplitude='RAMPHOLD')
    # If it's an actuator experiencing blocked force, set additional BCs.
    if args.test=='F':
        region = a.instances['SPA'].sets['BC_CUPS']
        mdb.models['Model-1'].DisplacementBC(name='CUPS', createStepName='Initial',
            region=region, u1=SET, u2=SET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
            amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)


    #--------------------------------------------------------------------------------
    # Create a reference point for calculating the length of the actuator.
    # We like having this for all tests, in case we need to know the length.
    #--------------------------------------------------------------------------------
    a.ReferencePoint(point=(body_W/2.0, body_H/2.0, body_L))
    rp = a.referencePoints[a.referencePoints.keys()[0]]
    a.Set(referencePoints=(rp,), name='LPT')
    region1 = a.sets['LPT']
    region2 = a.instances['SPA'].sets['TUBE']
    mdb.models['Model-1'].Coupling(name='LENGTH_MEASUREMENT', controlPoint=region1,
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=DISTRIBUTING,
        weightingMethod=UNIFORM, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON,
        ur2=ON, ur3=ON)

    #--------------------------------------------------------------------------------
    # Adjust the output requests.
    #--------------------------------------------------------------------------------
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(frequency=output_frequency,variables=(
        'S',                          # Stress components and invariants.
        # 'PE', 'PEEQ', 'PEMAG',        # Plastic strain, Equivalent plastic strain, Plastic strain magnitude.
        'EE', 'IE', 'NE', 'LE',       # Elastic strain, Inelastic strain, Nominal strain, Logarithmic strain.
        'U', 'V', 'A',                # Displacement, Velocity, Acceleration.
        'RF', 'CF', 'P',              # Reaction forces and moments, Concentrated forces and moments, Pressure loads.
        'CSTRESS', 'CDISP', 'CFORCE', # Contact stresses, Contact displacements, Contact forces.
        'ENER',                       # All energy magnitudes.
        # 'EVOL',                       # Element volume.
        ))
    regionDef = a.allInstances['SPA'].sets['PRESSURE']
    mdb.models['Model-1'].FieldOutputRequest(name='PRESSURE', createStepName='Step-1',
        variables=('P', ), region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE, frequency=output_frequency)
    regionDef=a.sets['LPT']
    mdb.models['Model-1'].HistoryOutputRequest(name='LENGTH', variables=(('COOR3'),),
        createStepName='Step-1', region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE, frequency=output_frequency)
    # H-Output-1 is the default history output request with the default variables.
    mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(frequency=output_frequency)





    #================================================================================
    # For a displacement test, add a nodeset to measure displacements.
    #================================================================================
    if args.test=='U':
        verts = v.findAt(((0.0, 0.0, 0.0), ))
        p.Set(vertices=verts, name='UPT')
        # Create a history output request for the displacement at the RP.
        regionDef = a.allInstances['SPA'].sets['UPT']
        mdb.models['Model-1'].HistoryOutputRequest(name='DISPLACEMENT', frequency=output_frequency,
            createStepName='Step-1', region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE,
            variables=('U1', 'U2', 'U3', 'COOR3'))


    #================================================================================
    # If the test is a blocked force test, add blocked force BCs.
    #================================================================================
    elif args.test=='F':
        #--------------------------------------------------------------------------------
        # Create the analytic rigid surface part.
        #--------------------------------------------------------------------------------
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
        s.setPrimaryObject(option=STANDALONE)
        if args.actuator=='lin':
            s.Line(point1=(0.0, 0.0), point2=(4.0*body_W, 0.0))
        elif args.actuator=='bnd':
            rbwidth = 2.0*body_W
            s.Line(point1=(0.0, 0.0), point2=(0.0, rbwidth))
        s.HorizontalConstraint(entity=s.geometry[2], addUndoState=False)
        pr = mdb.models['Model-1'].Part(name='WALL', dimensionality=THREE_D, type=ANALYTIC_RIGID_SURFACE)
        pr = mdb.models['Model-1'].parts['WALL']
        pr.AnalyticRigidSurfExtrude(sketch=s, depth=2.0*body_H)  # Depth is for display only (infinite depth).
        s.unsetPrimaryObject()
        pr = mdb.models['Model-1'].parts['WALL']
        del mdb.models['Model-1'].sketches['__profile__']

        #--------------------------------------------------------------------------------
        # Add an instance of the part, and move into place.
        #--------------------------------------------------------------------------------
        if args.actuator=='lin':
            a.Instance(name='WALL', part=pr, dependent=ON)
            a.rotate(instanceList=('WALL', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0), angle=90.0)
            a.translate(instanceList=('WALL', ), vector=(0.0, 0.5*body_H, 0.0))
            if dist_to_force!=0.0: a.translate(instanceList=('WALL', ), vector=(0.0, 0.0, -dist_to_force))
        elif args.actuator=='bnd':
            hw = 0.5*rbwidth
            a.Instance(name='WALL', part=pr, dependent=ON)
            a.translate(instanceList=('WALL', ), vector=(0.5*body_W, -hw, 0.0))
            # Translate and rotate for offset if requested.
            if dist_to_force!=0.0:
                shift = 0.5
                a.rotate(instanceList=('WALL', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0), angle=-dist_to_force)
                L = body_L-offset
                theta = np.radians(-dist_to_force)
                a.translate(instanceList=('WALL', ), vector=(0.0, shift*L*np.sin(theta), L-shift*L*np.cos(theta)))
            # The cuptop and cupside walls are translated an additional 0.5 units away from the unconstrained
            # material. This prevents the SPA from being in immediate contact with the wall at the start of the simulation.
            cup_position = body_L-offset+2.0
            a.Instance(name='CUPTOP', part=pr, dependent=ON)
            a.rotate(instanceList=('CUPTOP', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(0.0, 1.0, 0.0), angle=90.0)
            a.translate(instanceList=('CUPTOP', ), vector=(0.5*body_W, -hw, 0.0))
            a.rotate(instanceList=('CUPTOP', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0), angle=45.0)
            a.translate(instanceList=('CUPTOP', ), vector=(0.0, body_H, cup_position))
            a.Instance(name='CUPSIDE', part=pr, dependent=ON)
            a.rotate(instanceList=('CUPSIDE', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(0.0, 1.0, 0.0), angle=90.0)
            a.rotate(instanceList=('CUPSIDE', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(1.0, 0.0, 0.0), angle=90.0)
            a.rotate(instanceList=('CUPSIDE', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(0.0, 0.0, 1.0), angle=90.0)
            a.translate(instanceList=('CUPSIDE', ), vector=(body_W+hw, hw, cup_position-hw))

        #--------------------------------------------------------------------------------
        # Create the rigid body constraints.
        #--------------------------------------------------------------------------------
        # Create the reference points.
        if args.actuator=='lin':
            a.ReferencePoint(point=(0.0, 0.5*body_H, 0.0))
            a.features.changeKey(fromName='RP-2', toName='FORCE_RP')
            rp = a.referencePoints[a.referencePoints.keys()[0]]
            a.Set(referencePoints=(rp,), name='FPT')
            # Lock the reference point in place.
            region = a.sets['FPT']
            mdb.models['Model-1'].EncastreBC(name='WALL_RP', createStepName='Step-1', region=region, localCsys=None)
            # Constrain the rigid body "wall" to the reference point, which is fixed.
            s = a.instances['WALL'].faces
            faces = a.instances['WALL'].faces
            region1=regionToolset.Region(side1Faces=faces)
            region2 = a.sets['FPT']
            mdb.models['Model-1'].RigidBody(name='FIXWALL', refPointRegion=region2, surfaceRegion=region1)
        elif args.actuator=='bnd':
            a.ReferencePoint(point=(0.0, -5.0, 0.0))
            a.features.changeKey(fromName='RP-2', toName='FORCE_RP')
            rp = a.referencePoints[a.referencePoints.keys()[0]]
            a.Set(referencePoints=(rp,), name='FPT')
            a.ReferencePoint(point=(0.0, body_H+0.5*rbwidth, body_L-offset+5.0))
            a.features.changeKey(fromName='RP-2', toName='CUPTOP_RP')
            rp = a.referencePoints[a.referencePoints.keys()[0]]
            a.Set(referencePoints=(rp,), name='CUPTOPPT')
            a.ReferencePoint(point=(body_W+0.5*rbwidth, 0.0, body_L-offset+5.0))
            a.features.changeKey(fromName='RP-2', toName='CUPSIDE_RP')
            rp = a.referencePoints[a.referencePoints.keys()[0]]
            a.Set(referencePoints=(rp,), name='CUPSIDEPT')
            # Lock the reference point in place.
            region = a.sets['FPT']
            mdb.models['Model-1'].EncastreBC(name='WALL_RP', createStepName='Step-1', region=region, localCsys=None)
            region = a.sets['CUPTOPPT']
            mdb.models['Model-1'].EncastreBC(name='CUPTOP_RP', createStepName='Step-1', region=region, localCsys=None)
            region = a.sets['CUPSIDEPT']
            mdb.models['Model-1'].EncastreBC(name='CUPSIDE_RP', createStepName='Step-1', region=region, localCsys=None)
            # Constrain the rigid body "wall" to the reference point, which is fixed.
            faces = a.instances['WALL'].faces
            region1=regionToolset.Region(side2Faces=faces)
            region2 = a.sets['FPT']
            mdb.models['Model-1'].RigidBody(name='FIXWALL', refPointRegion=region2, surfaceRegion=region1)
            s = a.instances['CUPTOP'].faces
            faces = s.findAt(((0.1, body_H, cup_position), ))  # "cuptop"
            region1=regionToolset.Region(side1Faces=faces)
            region2 = a.sets['CUPTOPPT']
            mdb.models['Model-1'].RigidBody(name='FIXCUPTOP', refPointRegion=region2, surfaceRegion=region1)
            s = a.instances['CUPSIDE'].faces
            faces = s.findAt(((body_W+0.1, 0.1, cup_position), ))  # "cupside"
            region1=regionToolset.Region(side1Faces=faces)
            region2 = a.sets['CUPSIDEPT']
            mdb.models['Model-1'].RigidBody(name='FIXCUPSIDE', refPointRegion=region2, surfaceRegion=region1)

        #--------------------------------------------------------------------------------
        # Create the interaction between the wall and the SPA.
        #--------------------------------------------------------------------------------
        # Define the interaction properties between the SPA and the wall.
        # For force tests with a separated wall, use a "sticky" interaction to avoid bouncing.
        mdb.models['Model-1'].ContactProperty('SPA_CUP_INT')
        # mdb.models['Model-1'].interactionProperties['SPA_CUP_INT'].TangentialBehavior(formulation=ROUGH)
        # mdb.models['Model-1'].interactionProperties['SPA_CUP_INT'].NormalBehavior(
            # pressureOverclosure=HARD, allowSeparation=OFF, constraintEnforcementMethod=DIRECT)
        mdb.models['Model-1'].interactionProperties['SPA_CUP_INT'].NormalBehavior(
            pressureOverclosure=HARD, allowSeparation=ON, contactStiffness=DEFAULT,
            contactStiffnessScaleFactor=1.0, clearanceAtZeroContactPressure=0.0,
            constraintEnforcementMethod=AUGMENTED_LAGRANGE)
        # mdb.models['Model-1'].interactionProperties['SPA_CUP_INT'].NormalBehavior(
            # pressureOverclosure=HARD, allowSeparation=ON, contactStiffness=DEFAULT,
            # contactStiffnessScaleFactor=1.0, clearanceAtZeroContactPressure=0.0,
            # stiffnessBehavior=NONLINEAR, stiffnessRatio=0.01,
            # upperQuadraticFactor=0.03, lowerQuadraticRatio=0.33333,
            # constraintEnforcementMethod=PENALTY)
        # mdb.models['Model-1'].interactionProperties['SPA_CUP_INT'].NormalBehavior(
            # pressureOverclosure=HARD, allowSeparation=ON, contactStiffness=DEFAULT,
            # contactStiffnessScaleFactor=1.0, clearanceAtZeroContactPressure=0.0,
            # stiffnessBehavior=LINEAR, constraintEnforcementMethod=PENALTY)
        mdb.models['Model-1'].ContactProperty('SPA_WALL_INT')
        if dist_to_force==0.0:
            mdb.models['Model-1'].interactionProperties['SPA_WALL_INT'].NormalBehavior(
                pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DIRECT)
        else:
            mdb.models['Model-1'].StdContactControl(name='ContCtrl-1',stabilizeChoice=AUTOMATIC,
                stiffnessScaleFactor=100.0, relativePenetrationTolerance=100.0)   # Overrides other settings.
            # mdb.models['Model-1'].interactionProperties['SPA_WALL_INT'].TangentialBehavior(formulation=ROUGH)
            mdb.models['Model-1'].interactionProperties['SPA_WALL_INT'].TangentialBehavior(
                formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF,
                pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
                0.9, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION,
                fraction=0.005, elasticSlipStiffness=None)
            mdb.models['Model-1'].interactionProperties['SPA_WALL_INT'].NormalBehavior(
                 pressureOverclosure=HARD, allowSeparation=OFF, constraintEnforcementMethod=DIRECT)
            # mdb.models['Model-1'].interactionProperties['SPA_WALL_INT'].NormalBehavior(
                # pressureOverclosure=HARD, contactStiffness=DEFAULT, contactStiffnessScaleFactor=1.0,
                # clearanceAtZeroContactPressure=0.0, constraintEnforcementMethod=AUGMENTED_LAGRANGE,
                # allowSeparation=OFF)
                # allowSeparation=ON)
            # mdb.models['Model-1'].interactionProperties['SPA_WALL_INT'].NormalBehavior(
                # pressureOverclosure=HARD, allowSeparation=ON, contactStiffness=DEFAULT,
                # contactStiffnessScaleFactor=1.0, clearanceAtZeroContactPressure=0.0,
                # stiffnessBehavior=NONLINEAR, stiffnessRatio=0.01,
                # upperQuadraticFactor=0.03, lowerQuadraticRatio=0.33333,
                # constraintEnforcementMethod=PENALTY)
        # Define which surfaces and methods are used to enforce interaction.
        faces = a.instances['WALL'].faces
        region1=regionToolset.Region(side1Faces=faces)
        region2=a.instances['SPA'].surfaces['BLOCKED']
        mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='SPA_WALL',
            createStepName='Step-1', master=region1, slave=region2, sliding=FINITE,
            enforcement=NODE_TO_SURFACE, thickness=OFF, interactionProperty='SPA_WALL_INT',
            surfaceSmoothing=NONE, adjustMethod=NONE, smooth=0.2,
            initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
        if args.actuator=='bnd':
            s = a.instances['CUPTOP'].faces
            faces = s.findAt(((0.1, body_H, cup_position), ))
            region1=regionToolset.Region(side1Faces=faces)
            region2=a.instances['SPA'].surfaces['CUP_CONTACT']
            # Enable contact on top of SPA.
            mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='SPA_CUPTOP',
                createStepName='Step-1', master=region1, slave=region2, sliding=FINITE,
                thickness=OFF, interactionProperty='SPA_CUP_INT', surfaceSmoothing=NONE, adjustMethod=NONE,
                smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None,
                # enforcement=NODE_TO_SURFACE)
                enforcement=SURFACE_TO_SURFACE)
            s = a.instances['CUPSIDE'].faces
            faces = s.findAt(((body_W+0.1, 0.1, cup_position), ))
            region1=regionToolset.Region(side1Faces=faces)
            region2=a.instances['SPA'].surfaces['CUP_CONTACT']
            # Enable contact on side of SPA.
            # mdb.models['Model-1'].SurfaceToSurfaceContactStd(name='SPA_CUPSIDE',
                # createStepName='Step-1', master=region1, slave=region2, sliding=FINITE,
                # thickness=OFF, interactionProperty='SPA_CUP_INT', surfaceSmoothing=NONE, adjustMethod=NONE,
                # smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None,
                # enforcement=NODE_TO_SURFACE)
                # enforcement=SURFACE_TO_SURFACE)

        #--------------------------------------------------------------------------------
        # Create a history output request for the blocked force at the RP.
        #--------------------------------------------------------------------------------
        regionDef=a.sets['FPT']
        mdb.models['Model-1'].HistoryOutputRequest(name='BLOCKED_FORCE', frequency=output_frequency,
            createStepName='Step-1', region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE,
            variables=('RF1', 'RF2', 'RF3', 'COOR3'))


    #================================================================================
    # Create the job and save the file.
    # Note: number of CPUs isn't written to the .inp file, it's only used if you
    #       run this job from the GUI.
    #================================================================================
    jname = os.path.split(os.path.splitext(args.cae)[0])
    mdb.Job(name=jname[1], model='Model-1', description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=FULL, echoPrint=OFF,
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
        scratch='', multiprocessingMode=DEFAULT, numCpus=8, numDomains=8,
        numGPUs=0)
    if jname[0]!='':
        print 'WARNING: Abaqus can\'t write a .inp file anywhere other than the current directory.'
        print 'Abaqus will try to write it in the current directory, and then we will move it.'
        if os.path.isfile(jname[1]+'.inp'):
            print 'ERROR: file exists!'
            exit(1)
    mdb.jobs[jname[1]].writeInput(consistencyChecking=OFF)
    if jname[0]!='':
        path1 = jname[1]+'.inp'
        path2 = os.path.join(jname[0],path1)
        shutil.move(path1, path2)
        print 'Successfully moved the file from',path1,'to',path2
    mdb.saveAs(pathName=args.cae)
