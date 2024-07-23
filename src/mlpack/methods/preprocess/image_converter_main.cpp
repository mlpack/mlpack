/**
 * @file methods/preprocess/image_converter_main.cpp
 * @author Jeffin Sam
 *
 * A binding to load and save a image dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME image_converter

#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;
using namespace mlpack::data;

// Program Name.
BINDING_USER_NAME("Image Converter");

// Short description.
BINDING_SHORT_DESC(
    "A utility to load an image or set of images into a single dataset that"
    " can then be used by other mlpack methods and utilities. This can also"
    " unpack an image dataset into individual files, for instance after mlpack"
    " methods have been used.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes an image or an array of images and loads them to a"
    " matrix. You can optionally specify the height " +
    PRINT_PARAM_STRING("height") + " width " + PRINT_PARAM_STRING("width")
    + " and channel " + PRINT_PARAM_STRING("channels") + " of the images that"
    " needs to be loaded; otherwise, these parameters will be automatically"
    " detected from the image."
    "\n"
    "There are other options too, that can be specified such as " +
    PRINT_PARAM_STRING("quality")
    + ".\n\n" +
    "You can also provide a dataset and save them as images using " +
    PRINT_PARAM_STRING("dataset") + " and " + PRINT_PARAM_STRING("save") +
    " as an parameter.");

// Example.
BINDING_EXAMPLE(
    " An example to load an image : "
    "\n\n" +
    PRINT_CALL("image_converter", "input", "X", "height", 256, "width", 256,
        "channels", 3, "output", "Y") +
    "\n\n" +
    " An example to save an image is :"
    "\n\n" +
    PRINT_CALL("image_converter", "input", "X", "height", 256, "width", 256,
        "channels", 3, "dataset", "Y", "save", true));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
#if BINDING_TYPE == BINDING_TYPE_CLI
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");
#endif

// DEFINE PARAM
PARAM_VECTOR_IN_REQ(string, "input", "Image filenames which have to "
    "be loaded/saved.", "i");

PARAM_INT_IN("width", "Width of the image.", "w", 0);
PARAM_INT_IN("channels", "Number of channels in the image.", "c",  0);

PARAM_MATRIX_OUT("output", "Matrix to save images data to, Only"
    "needed if you are specifying 'save' option.", "o");

PARAM_INT_IN("quality", "Compression of the image if saved as jpg (0-100).",
    "q", 90);

PARAM_INT_IN("height", "Height of the images.", "H", 0);
PARAM_FLAG("save", "Save a dataset as images.", "s");
PARAM_MATRIX_IN("dataset", "Input matrix to save as images.", "I");

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Parse command line options.
  const vector<string> fileNames = params.Get<vector<string> >("input");
  arma::mat out;

  if (!params.Has("save"))
  {
    ReportIgnoredParam(params, "width", "Width of image is determined from "
        "file.");
    ReportIgnoredParam(params, "height", "Height of image is determined from "
        "file.");
    ReportIgnoredParam(params, "channels", "Number of channels determined from "
        "file.");
    data::ImageInfo info;
    Load(fileNames, out, info, true);
    if (params.Has("output"))
      params.Get<arma::mat>("output") = std::move(out);
  }
  else
  {
    RequireNoneOrAllPassed(params, { "save", "width", "height", "channels",
        "dataset" } , true, "Image size information is needed when 'save' is "
        "specified!");
    // Positive value for width.
    RequireParamValue<int>(params, "width", [](int x) { return x >= 0;}, true,
        "width must be positive");
    // Positive value for height.
    RequireParamValue<int>(params, "height", [](int x) { return x >= 0;}, true,
        "height must be positive");
    // Positive value for channel.
    RequireParamValue<int>(params, "channels", [](int x) { return x >= 0;},
        true, "channels must be positive");
    // Positive value for quality.
    RequireParamValue<int>(params, "quality", [](int x) { return x >= 0;}, true,
        "quality must be positive");

    const size_t height = params.Get<int>("height");
    const size_t width = params.Get<int>("width");
    const size_t channels = params.Get<int>("channels");
    const size_t quality = params.Get<int>("quality");
    data::ImageInfo info(width, height, channels, quality);
    Save(fileNames, params.Get<arma::mat>("dataset"), info, true);
  }
}
