cmake_minimum_required(VERSION 2.8)
project(POLE_PT)
 
set(CMAKE_CXX_COMPILER g++)
FIND_PACKAGE (Boost COMPONENTS program_options thread REQUIRED)

add_definitions( -pthread -lgsl -Wall -O2 -export)
add_executable(pole_pt pole_pt.cc pole.cc data.cc sparsela.cc learner.cc weak_learner.cc loss.cc kernel.cc transform.cc log.cc opt_ogd.cc opt_oeg.cc opt_wm.cc)

target_link_libraries(pole_pt
  gsl
  gslcblas
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
)
