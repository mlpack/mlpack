# The linkable library for original dualtree KDE algorithm
librule(
    name = "dualtree_kde",
    sources = [],
    headers = ["dataset_scaler.h",
               "dualtree_kde.h",
               "dualtree_kde_impl.h",
               "inverse_normal_cdf.h",
               "naive_kde.h"],
    deplibs = ["../series_expansion:series_expansion",
               "contrib/dongryel/proximity_project:proximity_project",
               "fastlib:fastlib_int"]
    )

# The linkable library for KDE using the original fast Gauss transform
librule(
    name = "fgt_kde",
    sources = [],
    headers = ["fgt_kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The linkable library for KDE based on the original improved fast
# Gauss transform.
librule(
    name = "original_ifgt",
    sources = [],
    headers = ["original_ifgt.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The linkable library for FFT-based KDE
librule(
    name = "fft_kde",                        
    sources = [],                            
    headers = ["fft_kde.h"],                 
    deplibs = ["fastlib:fastlib_int"]
    )

# The test driver for the FFT-based KDE
binrule(
    name = "fft_kde_bin",
    sources = ["fft_kde_main.cc"],
    headers = ["fft_kde.h",
               "kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The test driver for the FGT-based KDE
binrule(
    name = "fgt_kde_bin",
    sources = ["fgt_kde_main.cc"],
    headers = ["fgt_kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The test-driver for the THOR-basd KDE
binrule(
    name = "thor_kde_bin",
    sources = ["thor_kde_main.cc"],
    headers = ["thor_kde.h"],
    deplibs = ["../series_expansion:series_expansion",
               "fastlib:fastlib_int",
               "fastlib/thor:thor"]
    )

# The test-driver for the original dualtree KDE
binrule(
    name = "dualtree_kde_bin",
    sources = ["dualtree_kde_main.cc"],
    headers = [],
    deplibs = [":dualtree_kde",
               "../series_expansion:series_expansion",
               "fastlib:fastlib_int"]
    )

# The test-driver for the original IFGT-based KDE
binrule(
    name = "original_ifgt_bin",
    sources = ["original_ifgt_main.cc"],              
    headers = [],                            
    deplibs = [":original_ifgt",
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
