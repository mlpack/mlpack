
librule(
    name = "pca",              # this line can be safely omitted
    sources = [],              # files that must be compiled
    headers = ["pca.h"],
    deplibs = ["fastlib:fastlib"]        # dependency
    )

binrule(
    name = "main",
    sources = ["main.cc"],
    headers = [],
    deplibs = [":pca"]
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
