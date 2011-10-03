cmake_minimum_required(VERSION 2.6)

# the library for cuda_two_point_kernel
cuda_add_library(cuda_two_point_kernel
                 EXCLUDE_FROM_ALL
                 cuda_two_point_kernel.cu
                )
# link dependencies of cuda_two_point_kernel
target_link_libraries(cuda_two_point_kernel
                      core
                      ${Boost_LIBRARIES}
                     )

# the cuda_two_point driver
add_executable(cuda_two_point
               EXCLUDE_FROM_ALL
               cuda_two_point.cc
              )
#link dependencies of cuda_two_point
target_link_libraries(cuda_two_point
                      cuda_two_point_kernel
                      core
                      ${Boost_LIBRARIES}
                     )

# the test driver 
add_executable(cuda_two_point_test
               EXCLUDE_FROM_ALL
               cuda_two_point.test.cc
              )
# link dependencies 
target_link_libraries(cuda_two_point_test
		      cuda_two_point_kernel
                      core
                      ${Boost_LIBRARIES}
                     )

