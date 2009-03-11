
librule(
    name = "opt",
    sources = ["optimizers_reloaded.cc"],
    headers = ["optimizers_reloaded.h"],
    deplibs = ["fastlib:fastlib"],
    #tests = ["mog_em_tests.cc"]
    )

binrule(
    name = "main",
    sources = ["main.cc"],
    headers = ["phi.h"],
    deplibs = ["fastlib:fastlib",":opt"]
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
