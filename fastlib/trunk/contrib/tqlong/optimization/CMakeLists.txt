add_executable(optim_tqlong_test
   EXCLUDE_FROM_ALL
   test1.cpp
   FunctionTemplate
   NelderMead
   GradientDescent
   BFGS
   LBFGS
)
# link dependencies of executable

target_link_libraries(optim_tqlong_test
  fastlib
)

