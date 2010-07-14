cmake_minimum_required(VERSION 2.8)

add_executable(optim_tqlong_test
   EXCLUDE_FROM_ALL
   test1.cpp
   FunctionTemplate
   NelderMead
   GradientDescent
)
# link dependencies of executable
target_link_libraries(optim_tqlong_test
   fastlib
)

