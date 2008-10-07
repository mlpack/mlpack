
librule(
    name = "infomax_ica",              # this line can be safely omitted
    sources = ["infomax_ica.cc"],       # files that must be compiled
    headers = ["infomax_ica.h"],        # include files part of the 'lib'
    deplibs = ["fastlib:fastlib"]  # depends on fastlib core
    )

binrule(
    name = "infomax_ica_main",                 # the executable name
    sources = ["main.cc"],         # compile main.cc
    headers = [],                  # no extra headers
    deplibs = [":infomax_ica"]       # depends on InfomaxICA in this folder
    )

binrule(
    name = "infomax_ica_test_main",                 # the executable name
    sources = ["test_main.cc"],         # compile main.cc
    headers = [],                  # no extra headers
    deplibs = [":infomax_ica"]       # depends on InfomaxICA in this folder
    )

# to build:
# 1. make sure have environment variables set up:
#    $ source /full/path/to/fastlib/script/fl-env /full/path/to/fastlib
#    (you might want to put this in bashrc)
# 2. fl-build infomax_ica_main
#    - this automatically will assume --mode=check, the default
#    - type fl-build --help for help
# 3. ./main
#    - to build same target again, type: make
#    - to force recompilation, type: make clean

# some things to look out for:
#  - two build rules cannot have the same name.  that is why we can't have
#  an executable named "example" and a librule named "example"
#  - if a string contains ":" it is looked up as a rule in a build.py file,
#  but if it does not have a ":", it is assumed to be a file that already
#  exists.

