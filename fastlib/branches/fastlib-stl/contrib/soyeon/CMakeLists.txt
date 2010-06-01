cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   DDCM    ## unusable code
#   ddcm_sim1  ## does not link; looking for nonexistent data::Load function
#   ddcm_test  ## apparent one-time use code
#   ddcm_test2 ## apparent one-time use code
#   ddcm_test3 ## apparent one-time use code
#   obj ## apparently never finished
#   objective ## newer code in objective2
#   objective2 ## used in ddcm_sim1?
#   optimization ## used in ddcm_sim1?
#   sampling ## used in ddcm_sim1?
#   test    ## apparently nothing useful (hello world? really?)
#  test_trust ## appears to be a copy of other code
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
