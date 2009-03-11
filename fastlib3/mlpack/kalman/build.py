librule(
    name = "kalman",    # this line can be safely omitted
    sources = ["kalman_helper.cc","kalman.cc"], # files that must be compiled
    headers = ["kalman_helper.h","kalman.h"], # include files part of the 'lib'
    deplibs = ["fastlib:fastlib_int"] # depends on fastlib core
    )

binrule(
    name = "kalman_main",         # the executable name
    sources = ["kalman_main.cc"], # compile kalman_main.cc
    headers = [],                 # no extra headers
    deplibs = [":kalman"]         # depends on example in this folder
    )
