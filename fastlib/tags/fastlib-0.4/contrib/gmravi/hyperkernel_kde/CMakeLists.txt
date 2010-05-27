cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # hk_ippc_lib
   hyperkernels.h
   interior_point_pred_corr2.h
   ichol_dynamic.h
   dte.h
   special_la.h
   # test_ichol_lib ???
   ichol_dynamic.h
   # adaptive_kde_lib
   adaptive_kde.h
   # variable_kde_lib
   variable_nn_based_kde.h
   # nn_kde_lib
   nn_kde.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(hk_ippc
   EXCLUDE_FROM_ALL
   interior_point_pred_corr_main.cc
)
# link dependencies of executable
target_link_libraries(hk_ippc
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(varying_effects
#   varying_effects.cc
#)
#target_link_libraries(varying_effects
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(special_la_test # renamed from nonexplanatory "test"
   EXCLUDE_FROM_ALL
   special_la_main.cc
)
target_link_libraries(special_la_test
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(ichol_test
#   test_ichol_main.cc
#)
#target_link_libraries(ichol_test
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(adaptive_kde
   EXCLUDE_FROM_ALL
   adaptive_kde_main.cc
)
target_link_libraries(adaptive_kde
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(variable_nn_based_kde
   EXCLUDE_FROM_ALL
   variable_nn_based_kde.h
   variable_nn_based_kde_main.cc
)
target_link_libraries(variable_nn_based_kde
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(nn_kde
   EXCLUDE_FROM_ALL
   nn_kde_main.cc
)
target_link_libraries(nn_kde
   fastlib
   mlpack
   mlpack_contrib
)
