This program simulates the action of atoms of a single element (typically of 
an inert gas) under the influence of a pairwise Lennard-Jones potential of 
the form 

U = 4 epsilon ((sigma / r)^12 - (sigma / r)^6)

between all atoms, where r is the distance between any two atoms. At each
time step, for each point, the velocity of each atom is updated by computing 
the force from all other atoms in the simulation. The effects of collections 
of distant atoms are approximated from the position fo their centroid. This
algorithm is unpublished.

Compile with fl-build main

Run with ./main to implement the simulation on a default set of test data, or 
specify an input file using --data to give the program a set of 
three-dimensional atom coordinates to begin from. All initial velocities are
zero.

Options:

double --dt specifies the size of the time step takne, in nanoseconds. Default 
is 10e-3.

double --tf specifies the duration fo the simulation, in nanoseconds. Default 
is 10e0.

double --param/eps specifies the value of epsilon, governing the overall 
magnitude of the potential energy. It is in units of electronvolts. Default
value is .0104 eV, corresponding to the empirically determined value for Argon.

double --param/sig specifies the value of sigma in units of Angstroms, 
dictating the length scale at which the potential will switch from attractive 
to repulsive in nature. Default value is 2.74 A, corresponding to the 
empirically determined value for Argon.

double --param/mass gives the mass of each atom, in atomic mass units. Its 
default value, 40, corresponds to the most common isotope of Argon.

double --param/r_max gives the range, in Angstroms, at which the simulation
will begin approximating the effect of distant clusters of atoms, rather
than calculating their effects exactly.

boolean --check will run a naive (all-pairs) version of the simulation, and
compute the rms-deviation between the naive and tree-based implementations.

The positions of all atoms at the end of the simulation are recorded in the 
file out_tree.dat. If the 'check' function is used, the positions of all
atoms according to the naive simulation are recorded in the file out_naive.dat.
