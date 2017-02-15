# Find the MathJax package.
# Once done this will define
#
# MATHJAX_FOUND - system has MathJax
# MATHJAX_JS_PATH - path to MathJax.js
# MATHJAX_PATH - path to the MathJax root directory

find_file (MATHJAX_JS_PATH
    NAMES
      MathJax.js
    PATHS
      ${MATHJAX_ROOT}
      /usr/share/javascript/mathjax/
      /usr/local/share/javascript/mathjax/)

get_filename_component (MATHJAX_PATH ${MATHJAX_JS_PATH} DIRECTORY)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(MathJax DEFAULT_MSG
    MATHJAX_JS_PATH)

mark_as_advanced (MATHJAX_JS_PATH)
