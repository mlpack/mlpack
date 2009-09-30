add_library( series_expansion
    series_expansion_aux.cc
    bounds_aux.h
    farfield_expansion.h
    farfield_expansion_impl.h
    inverse_pow_dist_kernel.h
    inverse_pow_dist_kernel_aux.h
    kernel_aux.h
    local_expansion.h
    local_expansion_impl.h
    mult_farfield_expansion.h
    mult_farfield_expansion_impl.h
    mult_local_expansion.h
    mult_local_expansion_impl.h
    mult_series_expansion_aux.h
    series_expansion_aux.h
)
target_link_libraries(series_expansion ${MLPACK_STD_LIBS})

add_executable(series_expansion_main main.cc)
target_link_libraries(series_expansion_main series_expansion)
