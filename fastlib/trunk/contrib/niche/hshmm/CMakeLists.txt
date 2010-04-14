cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   hmm.h
   hmm_distance.h
   gaussian.h
   distribution.h
   hshmm.h
   lds.h
   mmk.h
   multinomial.h
#   svm.h   ## does not exist!
#   svm.cc  ## does not exist!
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(test_hmm_multinomial
   load_profile.cc
   test_hmm_multinomial.cc
   
)
# link dependencies of executable
target_link_libraries(test_hmm_multinomial
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_hmm_gaussian
   test_hmm_gaussian.cc
)
target_link_libraries(test_hmm_gaussian
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_lds
   test_lds.cc
)
target_link_libraries(test_lds
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_gaussian
   test_gaussian.cc
)
target_link_libraries(test_gaussian
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(test_distribution
   test_distribution.cc
)
target_link_libraries(test_distribution
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(test_debug
#   testdebug.cc
#)
#target_link_libraries(test_debug
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(load_profile
   load_profile.cc
)
target_link_libraries(load_profile
   fastlib
   mlpack   
   mlpack_contrib
)

add_executable(hmm_testing
   hmm_testing.cc
   hmm_testing.h
   test_hmm_multinomial.cc
   ../mmf/mmf3.cc
   ../svm/svm.h
   ../svm/smo.h
)
target_link_libraries(hmm_testing
   fastlib
   mlpack   
   mlpack_contrib
)

add_executable(hmm_generative_classifier
   hmm_generative_classifier.cc
   test_hmm_multinomial.cc
   ../mmf/mmf3.cc
   hmm_testing.h
   ../svm/svm.h
)
target_link_libraries(hmm_generative_classifier
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(read_variable_file
   read_variable_file.cc
   test_hmm_multinomial.cc
   ../mmf/mmf3.cc
   hmm_testing.h
   ../svm/svm.h
)
target_link_libraries(read_variable_file
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(perm_dna
#   perm_dna.cc
#   test_hmm_multinomial.cc
#   ../mmf/mmf3.cc
#   hmm_testing.h
#   ../svm/svm.h
#)
#target_link_libraries(perm_dna
#   fastlib  
#   mlpack
#   mlpack_contrib
#)

#add_executable(test_serial
#   test_serial.cc
#)
#target_link_libraries(test_serial
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(test_obs_kernel
   test_obs_kernel.cc
   mmk.h
#   ppk.h  ## does not exist!
)
target_link_libraries(test_obs_kernel
   fastlib
   mlpack
   mlpack_contrib
)
