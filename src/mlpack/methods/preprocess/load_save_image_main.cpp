/**
 * @file load_save_image_main.cpp
 * @author Jeffin Sam
 *
 * A CLI executable to load and image dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifdef HAS_STB // Compile this only if stb is present.

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/image_info.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

PROGRAM_INFO("Load Image",
    // Short description.
    "A utility to load and save image dataset. This utility will allow you to "
    "load and save a single image or an array of images.",
    // Long description.
    "This utility takes a images or an array of images and loads them to arma matrix."
    "You can specify the height " + PRINT_PARAM_STRING("height") + "width " +
    PRINT_PARAM_STRING("width") + " and channel " + PRINT_PARAM_STRING("channel") + 
    "of the images that needs to be loaded. \n There are other options too, that" 
    "can be specified such as quality " + PRINT_PARAM_STRING("quality") + " and " +
    PRINT_PARAM_STRING("transpose") + 
    "\n\n" +
    "You can also provide a dataset and save them as image using " + 
    PRINT_PARAM_STRING("dataset") + "and " + PRINT_PARAM_STRING("save") +
    "as an parameter. An example to load an "
    "image"  + "\n\n" + 
    PRINT_CALL("load_save_image", "input", "X", "height", 256, "width", 256,
        "channel", 3, "output", "Y") + "\n\n" +
    " An example to save an image is :" + "\n\n" + 
    PRINT_CALL("load_save_image", "input", "X", "height", 256, "width", 256,
        "channel", 3, "dataset", "Y"),
    SEE_ALSO("@preprocess_binarize", "#preprocess_binarize"),
    SEE_ALSO("@preprocess_describe", "#preprocess_describe"),
    SEE_ALSO("@preprocess_imputer", "#preprocess_imputer"));

//DEFINE PATAM
PARAM_VECTOR_IN_REQ(string, "input", "Image filenames which has to "
    "be loaded/saved", "i"); 

PARAM_INT_IN_REQ("width", "Width of the image", "W");
PARAM_INT_IN_REQ("channel", "Number of channel", "C");


PARAM_MATRIX_OUT("output", "Matrix to save images data to.", "o");

PARAM_INT_IN("quality", "Compression of the image if saved as jpg (0-100).", "q",90);

PARAM_FLAG("transpose", "Loaded dataset to be transposed", "t");

PARAM_INT_IN_REQ("height", "Height of the images", "H");
PARAM_FLAG("save", "Save a dataset as images", "s");
PARAM_MATRIX_IN("dataset", "Input matrix to save as images.", "I");

static void mlpackMain()
{
  // Parse command line options.
  const int height = CLI::GetParam<int>("height");
  const int width = CLI::GetParam<int>("width");
  const int channel = CLI::GetParam<int>("channel");
  const int quality = CLI::GetParam<int>("quality");
  data::ImageInfo info(height, width, channel, quality);
  const vector<string> fileNames =
      CLI::GetParam<vector<string> >("input");
  arma::mat out;
  if(!CLI::HasParam("save"))
    data::Load(fileNames, out, info, false, !CLI::HasParam("transpose"));
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(out);
  if(CLI::HasParam("save"))
  {
    if(!CLI::HasParam("dataset"))
    {
      throw std::runtime_error("Please provide a input matrix to save images from");
    }
    data::Save(fileNames, CLI::GetParam<arma::mat>("dataset"), info, false, !CLI::HasParam("transpose"));
  }
}

#endif // HAS_STB.
