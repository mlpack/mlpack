
librule(
    name = "kde",                            # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["fgt_kde.h",
               "kde.h"],                     # include files part of the 'lib'
    deplibs = ["u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]        # dependency
    )

librule(
    name = "fgt_kde",
    sources = [],
    headers = ["fgt_kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

librule(
    name = "ifgt_kde",
    sources = ["ifgt_kde.cc",
               "ifgt_choose_parameters.cc",
               "ifgt_choose_truncation_number.cc",
               "kcenter_clustering.cc"],
    headers = ["ifgt_kde.h",
               "ifgt_choose_parameters.h",
               "ifgt_choose_truncation_number.h",
               "kcenter_clustering.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

librule(
    name = "fft_kde",                        # this line can be safely omitted
    sources = [],                            # files that must be compiled
    headers = ["fft_kde.h"],                 # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"]        # dependency
    )

binrule(
    name = "fft_kde_bin",
    sources = ["fft_kde_main.cc"],
    headers = ["fft_kde.h",
               "kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

binrule(
    name = "fgt_kde_bin",
    sources = ["fgt_kde_main.cc"],
    headers = ["fgt_kde.h",
               "kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

binrule(
    name = "thor_kde_bin",
    sources = ["thor_kde_main.cc"],
    headers = ["thor_kde.h"],
    deplibs = ["u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int",
               "thor:thor"]
    )

binrule(
    name = "kde_bin",                        # the executable name
    sources = ["main.cc"],                   #
    headers = [],                            # no extra headers
    deplibs = [":fft_kde",
               ":fgt_kde",
               ":kde",
               "u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
    )

binrule(
    name = "ifgt_bin",                        # the executable name
    sources = ["ifgt_main.cc"],               #
    headers = [],                             # no extra headers
    deplibs = [":ifgt_kde",
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
