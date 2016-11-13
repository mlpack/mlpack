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

string(REPLACE
    "./doc/policies"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/policies"
    DOXYFILE_CONTENTS "${DOXYFILE_AUXVAR}"
)

string(REPLACE
    "./doc/doxygen/footer.html"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/footer.html"
    DOXYFILE_AUXVAR "${DOXYFILE_CONTENTS}"
)

string(REPLACE
    "./doc/doxygen/extra-stylesheet.css"
    "${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen/extra-stylesheet.css"
    DOXYFILE_CONTENTS "${DOXYFILE_AUXVAR}")

# Change the STRIP_FROM_PATH so that it works right even in the build directory;
# otherwise, every file will have the full path in it.
string(REGEX REPLACE
    "(STRIP_FROM_PATH[ ]*=) ./"
    "\\1 ${CMAKE_CURRENT_SOURCE_DIR}/"
    DOXYFILE_AUXVAR ${DOXYFILE_CONTENTS})

# Apply the MathJax option. If the option is specified, we change the NO to
# YES. Otherwise, it's off by default, so we needn't modify anything.
if (MATHJAX)
  string(REGEX REPLACE
      "(USE_MATHJAX[ ]*=) NO"
      "\\1 YES"
      DOXYFILE_CONTENTS ${DOXYFILE_AUXVAR})
  # Include the path to MathJax. If we couldn't find the MathJax package,
  # we will use MathJax at the MathJax Content Delivery Network.
  if (MATHJAX_FOUND)
    string(CONCAT
        DOXYFILE_AUXVAR
        ${DOXYFILE_CONTENTS}
        "\nMATHJAX_RELPATH        = ${MATHJAX_PATH}")

    set(DOXYFILE_CONTENTS ${DOXYFILE_AUXVAR})
  endif()
else ()
  set(DOXYFILE_CONTENTS ${DOXYFILE_AUXVAR})
endif ()

# Save the Doxyfile to its new location.
file(WRITE "${DESTDIR}/Doxyfile" "${DOXYFILE_CONTENTS}")
