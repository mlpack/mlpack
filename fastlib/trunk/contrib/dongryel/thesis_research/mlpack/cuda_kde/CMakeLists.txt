cmake_minimum_required(VERSION 2.6)

# the library for cuda_kde_kernel
cuda_add_library(cuda_kde_kernel
                 EXCLUDE_FROM_ALL
                 cuda_kde_kernel.cu
                )
# link dependencies of cuda_kde_kernel
target_link_libraries(cuda_kde_kernel
                      core
                      ${Boost_LIBRARIES}
                     )

# the cuda_kde driver
add_executable(cuda_kde
               EXCLUDE_FROM_ALL
               cuda_kde.cc
              )
#link dependencies of cuda_kde
target_link_libraries(cuda_kde
                      cuda_kde_kernel
                      core
                      ${Boost_LIBRARIES}
                     )
