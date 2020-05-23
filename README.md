# CSTclass_Henri
CST and its application for aerodynamic design
## <CST.py> 
This is provided for conventional CST parameterization. The solving uses least square method.
## <Mesh_prep_via_CST.py> 
This is provided to make preparation for subsequent airfoil flowfield meshing. The mesh points are uniformly distributed along curve. The ratio may be adapted later. This is better than distributing points along x-direction where the leading edge would be overlooked. 
### tips on accuracy
The accuracy is highly dependent on the raw airfoil, which is required to be:
 1. contains (0,0) point;
 2. starts at top of tail edge
 3. ends at low of tail edge
 4. first line's x coordinate is zero, so is last line's.
 The order of CST is also impacting curve accuracy. If there is a regon lacking enough points, a Runga effect would occur.
 
 
 
