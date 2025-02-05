/**
 * @file core/data/data.hpp
 * @author Ryan Curtin
 *
 * Convenience include for everything in the mlpack::data namespace.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATA_HPP
#define MLPACK_CORE_DATA_DATA_HPP

#include "detect_file_type.hpp"
#include "extension.hpp"
#include "format.hpp"
#include "has_serialize.hpp"

#include "load.hpp"
#include "save.hpp"

#include "imputation_methods/imputation_methods.hpp"
#include "map_policies/map_policies.hpp"
#include "scaler_methods/scaler_methods.hpp"
#include "string_encoding_policies/string_encoding_policies.hpp"
#include "tokenizers/tokenizers.hpp"

#include "binarize.hpp"
#include "check_categorical_param.hpp"
#include "confusion_matrix.hpp"
#include "dataset_mapper.hpp"
#include "image_info.hpp"
#include "image_resize_crop.hpp"
#include "imputer.hpp"
#include "is_naninf.hpp"
#include "normalize_labels.hpp"
#include "one_hot_encoding.hpp"
#include "split_data.hpp"
#include "string_algorithms.hpp"
#include "types.hpp"

#endif
