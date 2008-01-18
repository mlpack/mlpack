
librule(
    name = "example",              # this line can be safely omitted
    sources = ["LennardJones_main.cc"],       # files that must be compiled
    headers = ["AtomTree.h", "LennardJones.h"],      # include files part of the 'lib'
    deplibs = ["fastlib:fastlib", "fastlib:fastlib_int"]  # depends on fastlib core
    )

binrule(
    name = "main",                 # the executable name
    linkables = [":example"]
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
