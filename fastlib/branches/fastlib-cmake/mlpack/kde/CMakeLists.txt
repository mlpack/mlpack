# The linkable library for original dualtree KDE algorithm
add_library( dualtree_kde 
    bandwidth_lscv.h
    dataset_scaler.h
    dualtree_kde.h
    dualtree_kde_common.h
    dualtree_kde_cv.h
    dualtree_kde_cv_common.h
    dualtree_kde_cv_impl.h
    dualtree_kde_impl.h
    dualtree_vkde.h
    dualtree_vkde_impl.h
    general_spacetree.h
    gen_metric_tree.h
    gen_metric_tree_impl.h
    inverse_normal_cdf.h
    kde_stat.h
    naive_kde.h
)
set_target_properties( dualtree_kde PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( dualtree_kde
    series_expansion
    allknn
    ${MLPACK_STD_LIBS}
)

# The linkable library for KDE using the original fast Gauss transform
add_library( fgt_kde
    fgt_kde.h
)
set_target_properties( fgt_kde PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( fgt_kde ${MLPACK_STD_LIBS} )

# The linkable library for KDE based on the original improved fast
# Gauss transform.
add_library( original_ifgt
    original_ifgt.h
)
set_target_properties( original_ifgt PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( original_ifgt ${MLPACK_STD_LIBS} )

# The linkable library for FFT-based KDE
add_library( fft_kde
    fft_kde.h
)
set_target_properties( fft_kde PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( fft_kde ${MLPACK_STD_LIBS} )

# The test driver for the FFT-based KDE
add_executable( fft_kde_bin
    fft_kde_main.cc
    fft_kde.h
)
target_link_libraries( fft_kde_bin ${MLPACK_STD_LIBS} )

# The test driver for the FGT-based KDE
add_executable( fgt_kde_bin
    fgt_kde_main.cc
    fgt_kde.h
)
target_link_libraries( fgt_kde_bin ${MLPACK_STD_LIBS} )

# The test-driver for the original dualtree KDE
add_executable( dualtree_kde_bin dualtree_kde_main.cc )
target_link_libraries(dualtree_kde_bin dualtree_kde)
#    headers = [],
#    deplibs = [":dualtree_kde",
#               "../series_expansion:series_expansion",
#               "fastlib:fastlib_int"]

# The test-driver for the original IFGT-based KDE
add_executable( original_ifgt_bin original_ifgt_main.cc )
target_link_libraries( original_ifgt_bin original_ifgt )  # ${MLPACK_STD_LIBS} )

# The driver for the bandwidth cross-validator.
add_executable( kde_cv_bin kde_bandwidth_cv_main.cc )
target_link_libraries( kde_cv_bin dualtree_kde )
