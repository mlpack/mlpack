/**
 * @file prereqs.hpp
 *
 * The core includes that mlpack expects; standard C++ includes, Armadillo,
 * cereal, and a few basic mlpack utilities.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_PREREQS_HPP
#define MLPACK_PREREQS_HPP

#include "base.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/tuple.hpp>
#include <mlpack/core/cereal/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

#include <mlpack/core/cereal/is_loading.hpp>
#include <mlpack/core/cereal/is_saving.hpp>
#include <mlpack/core/arma_extend/serialize_armadillo.hpp>
#include <mlpack/core/cereal/array_wrapper.hpp>
#include <mlpack/core/cereal/pointer_vector_wrapper.hpp>
#include <mlpack/core/cereal/pointer_wrapper.hpp>
#include <mlpack/core/cereal/template_class_version.hpp>
#include <mlpack/core/data/has_serialize.hpp>

// Include ready to use utility function to check sizes of datasets.
#include <mlpack/core/util/size_checks.hpp>

#endif
