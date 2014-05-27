add_executable(nmf
  nmf_main.cpp
)
target_link_libraries(nmf
  mlpack
)
install(TARGETS nmf RUNTIME DESTINATION bin)
