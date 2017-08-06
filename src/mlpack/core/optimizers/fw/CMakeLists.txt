set(SOURCES
  frank_wolfe.hpp
  frank_wolfe_impl.hpp
  constr_lpball.hpp
  update_classic.hpp
  update_span.hpp
  func_sq.hpp
  test_func_fw.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
