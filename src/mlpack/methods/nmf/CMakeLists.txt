add_executable(mlpack_nmf
  nmf_main.cpp
)
target_link_libraries(mlpack_nmf
  mlpack
)
install(TARGETS mlpack_nmf RUNTIME DESTINATION bin)
