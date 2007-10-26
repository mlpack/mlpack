
librule(
    name = "kde",                            # this line can be safely omitted
    sources = ["ifgt_kde.cc",
               "kcenter_clustering.cc"],     # files that must be compiled
    headers = ["fft_kde.h",
               "fgt_kde.h",
               "ifgt_kde.h",
               "kcenter_clustering.h",
               "kde.h"],                     # include files part of the 'lib'
    deplibs = ["u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]        # dependency
    )

binrule(
    name = "kde_bin",                        # the executable name
    sources = ["main.cc"],                   #
    headers = [],                            # no extra headers
    deplibs = [":kde",
               "u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
    )

binrule(
    name = "ifgt_bin",                        # the executable name
    sources = ["ifgt_main.cc"],               #
    headers = [],                             # no extra headers
    deplibs = [":kde",
               "u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
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
