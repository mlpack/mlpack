# Define the files that we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  dataset_mapper.hpp
  dataset_mapper_impl.hpp
  extension.hpp
  format.hpp
  load.hpp
  load_impl.hpp
  load_arff.hpp
  load_arff_impl.hpp
  normalize_labels.hpp
  normalize_labels_impl.hpp
  save.hpp
  save_impl.hpp
  serialization_shim.hpp
  split_data.hpp
  imputer.hpp
  binarize.hpp
  file_reader/csv_reader.hpp
  file_reader/line_reader.hpp
  file_reader/parser.hpp
  file_reader/policy.hpp
  file_reader/reader_exceptions.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Append sources (with directory name) to list of all mlpack sources (used at
# parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
