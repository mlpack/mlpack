The files in this directory are taken from newer versions of Armadillo in order
to still support older versions of Armadillo.  Therefore some files are licensed
under the Mozilla Public License v2.0 (MPL2).

These are the files under the MPL:

 - fn_ind2sub.hpp
 - SpMat_extra_bones.hpp
 - SpMat_extra_meat.hpp
 - operator_minus.hpp
 - hdf5_misc.hpp
 - Mat_extra_bones.hpp
 - Mat_extra_meat.hpp

If you want a copy of mlpack without MPL code included, you will need to

 * Remove all of the above-listed files.
 * Remove the above-listed files from CMakeLists.txt.
 * Remove the above-listed files from arma_extend.hpp.
 * Modify the root CMakeLists.txt to require a sufficiently new version of
 * Armadillo that none of the above backports are required, by changing the line
   "find_package(Armadillo x.yyy.z REQUIRED)" to reference a sufficiently new
   version instead of x.yyy.z.
