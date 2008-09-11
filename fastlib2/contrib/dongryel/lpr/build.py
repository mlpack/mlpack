librule(
    name = "lpr",
    sources = [],
    headers = ["lpr.h",
               "lpr_bandwidth_cv.h",
               "lpr_bandwidth_cv_impl.h",
               "lpr_impl.h"]
    deplibs = ["fastlib:fastlib"]
    )

binrule(
    name = "lpr_bin",
    sources = ["lpr_main.cc"],
    headers = [],
    deplibs = [":lpr",
               "fastlib:fastlib"]
    )

# to build:
# 1. make sure have environment variables set up:
#    $ source /full/path/to/fastlib/script/fl-env /full/path/to/fastlib
#    (you might want to put this in bashrc)
# 2. fl-build main
#    - this automatically will assume --mode=check, the default
#    - type fl-build --help for help
# 3. ./main
#    - to build same target again, type: make
#    - to force recompilation, type: make clean
