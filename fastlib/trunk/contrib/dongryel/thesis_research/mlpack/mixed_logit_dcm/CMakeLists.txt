cmake_minimum_required(VERSION 2.6)

# the main driver for mixed_logit_dcm
add_executable(mixed_logit_dcm
               EXCLUDE_FROM_ALL
               mixed_logit_dcm.cc
              )

# link dependencies of mixed_logit_dcm
target_link_libraries(mixed_logit_dcm
                      core
                      ${Boost_LIBRARIES}
                     )
