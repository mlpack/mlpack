cmake_minimum_required(VERSION 2.6)

# the main driver for cuda_kde
cuda_add_executable(cuda_kde
                    EXCLUDE_FROM_ALL
                    cuda_kde.cu
                   )
# link dependencies of kde
target_link_libraries(cuda_kde
                      core
                      ${Boost_LIBRARIES}
                     )
