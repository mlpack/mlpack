cmake_minimum_required(VERSION 2.8)
project(POLE_PT)
 
#set(CMAKE_PREFIX_PATH /scratch/app/boost)
set(CMAKE_CXX_COMPILER g++)
FIND_PACKAGE (Boost COMPONENTS program_options thread REQUIRED)
 
#link_directories(/scratch/app/boost/lib)
#include_directories(/scratch/app/boost/include)

add_definitions( -Wall -O2 )
add_executable(pole_pt pole_pt.cc pole.cc data.cc learner.cc loss.cc opt_ogd.cc opt_oeg.cc opt_dwm_i.cc opt_dwm_a.cc)

target_link_libraries(pole_pt 
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${Boost_THREAD_LIBRARY}
)
