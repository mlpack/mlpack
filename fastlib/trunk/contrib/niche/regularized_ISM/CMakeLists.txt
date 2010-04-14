cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   # I don't know what the hell is in csdp_testing so I will ignore it
   write_cvxmod_program
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()
