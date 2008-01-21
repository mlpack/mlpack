# The linkable library for fast KDE algorithm
librule(
    name = "kde",                    
    sources = [],                    
    headers = ["dataset_scaler.h",
               "kde.h",
               "naive_kde.h"],       
    deplibs = ["u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
    )

# The linkable library for KDE using the original fast Gauss transform
librule(
    name = "fgt_kde",
    sources = [],
    headers = ["fgt_kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The linkable library for KDE based on the improved fast Gauss transform
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
    headers = ["fgt_kde.h",
               "kde.h"],
    deplibs = ["fastlib:fastlib_int"]
    )

# The test-driver for the THOR-basd KDE
binrule(
    name = "thor_kde_bin",
    sources = ["thor_kde_main.cc"],
    headers = ["thor_kde.h"],
    deplibs = ["u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int",
               "thor:thor"]
    )

# The test-driver for the fast KDE
binrule(
    name = "kde_bin",                       
    sources = ["kde_main.cc"],              
    headers = [],                           
    deplibs = [":fft_kde",
               ":fgt_kde",
               ":kde",
               "u/dongryel/series_expansion:series_expansion",
               "fastlib:fastlib_int"]
    )

# The test-driver for the IFGT-based KDE
binrule(
    name = "ifgt_bin",
    sources = ["ifgt_main.cc"],              
    headers = [],                            
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
