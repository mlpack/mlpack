cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
# In this library, these are specified twice, once here, and once for the individual library it belongs to, so make sure
# that you have files in both sections
set(SOURCES
   # for dualtree_kde
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
   # for fgt_kde
   fgt_kde.h
   # for original_ifgt
   original_ifgt.h
   # for fft_kde
   fft_kde.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The executable for the FFT-based KDE
add_executable(fft_kde
    fft_kde_main.cc
)
target_link_libraries(fft_kde
   mlpack
   fastlib
)   

# The executable for the FGT-based KDE
add_executable(fgt_kde
   fgt_kde_main.cc
)
target_link_libraries(fgt_kde
   mlpack
   fastlib
)

# The executable for the original dualtree KDE
add_executable(dualtree_kde
   dualtree_kde_main.cc
)
target_link_libraries(dualtree_kde
   mlpack
   fastlib
)

# The executable for the original IFGT-based KDE
add_executable(original_ifgt
   original_ifgt_main.cc
)
target_link_libraries(original_ifgt
   mlpack
   fastlib
)

# The executable for the bandwidth cross-validator.
add_executable(kde_cv
   kde_bandwidth_cv_main.cc
)
target_link_libraries(kde_cv
   mlpack
   fastlib
)
