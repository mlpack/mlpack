# We need to modify the Doxyfile slightly.  We'll copy the Doxyfile into the
# build directory, update the location of the source, and then run Doxygen and
# it will generate the documentation into the build directory.

# First, read the Doxyfile in as a variable.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile" DOXYFILE_CONTENTS)

# Now, modify all the "INPUT" paths.  I've written each of the three out by
# hand.  If more are added, they'll need to be added here too.
string(REPLACE
    "./src/mlpack"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack"
    DOXYFILE_AUXVAR "${DOXYFILE_CONTENTS}"
)

string(REPLACE
    "./doc/guide"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/guide"
    DOXYFILE_CONTENTS "${DOXYFILE_AUXVAR}"
)

string(REPLACE
    "./doc/tutorials"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/tutorials"
    DOXYFILE_AUXVAR "${DOXYFILE_CONTENTS}"
)

# Save the Doxyfile to its new location.
file(WRITE "${DESTDIR}/Doxyfile" "${DOXYFILE_AUXVAR}")
