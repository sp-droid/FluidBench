// Uniform sampling grid for the cylinder_8000_interp datasets.
// Simplified from cylinder_v2.geo: same 40 x 20 domain and cylinder (D = 2),
// but meshed as the 64 x 32 node grid the CFD solution is interpolated onto
// (np.linspace over each axis, so spacing is 40/63 by 20/31).
// The cylinder is not cut out of the grid: nodes inside it are masked as invalid.

// Params
Lx = 40; Ly = 20; Lz = 1;
Nx = 64; Ny = 32;
Xc = 8; Yc = 10; R = 1;

// Domain points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {Lx, 0, 0, 1.0};
Point(3) = {Lx, Ly, 0, 1.0};
Point(4) = {0, Ly, 0, 1.0};
// Cylinder points
Point(5) = {Xc, Yc, 0, 1.0};
Point(6) = {Xc + R, Yc, 0, 1.0};
Point(7) = {Xc, Yc + R, 0, 1.0};
Point(8) = {Xc - R, Yc, 0, 1.0};
Point(9) = {Xc, Yc - R, 0, 1.0};

// Domain curves
Line(1) = {1, 2}; Transfinite Curve {1} = Nx Using Progression 1;
Line(2) = {2, 3}; Transfinite Curve {2} = Ny Using Progression 1;
Line(3) = {4, 3}; Transfinite Curve {3} = Nx Using Progression 1;
Line(4) = {1, 4}; Transfinite Curve {4} = Ny Using Progression 1;
// Cylinder curves (reference only, not part of the grid surface)
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Surface
Curve Loop(1) = {1, 2, -3, -4};
Plane Surface(1) = {1};
Transfinite Surface {1};
Recombine Surface {1};

// Extrude one cell layer in z, as in the original 3D OpenFOAM case
grid[] = Extrude {0, 0, Lz} { Surface{1}; Layers{1}; Recombine; };
cyl[] = Extrude {0, 0, Lz} { Curve{5, 6, 7, 8}; Layers{1}; Recombine; };
// grid[]: 0 top surface, 1 volume, 2-5 lateral surfaces from curves 1, 2, -3, -4
// cyl[]: top curve and lateral surface for each cylinder arc

Physical Surface("inlet") = {grid[5]};
Physical Surface("outlet") = {grid[3]};
Physical Surface("topAndBottom") = {grid[2], grid[4]};
Physical Surface("cylinder") = {cyl[1], cyl[3], cyl[5], cyl[7]};
Physical Surface("frontAndBack") = {1, grid[0]};
Physical Volume("volume") = {grid[1]};
