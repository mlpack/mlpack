#include <mlpack.hpp>

#include "image_resize_crop.hpp"
#include "image_info.hpp"

using namespace mlpack;
using namespace mlpack::data;

int main()
{
  arma::Mat<unsigned char> image, images;
  data::ImageInfo info;
  std::vector<std::string> files =
      {"sheep_1.jpg", "sheep_2.jpg","sheep_3.jpg", "sheep_4.jpg",
       "sheep_5.jpg", "sheep_6.jpg"};
  std::vector<std::string> re_sheeps =
      {"re_sheep_1.jpg", "re_sheep_2.jpg","re_sheep_3.jpg", "re_sheep_4.jpg",
       "re_sheep_5.jpg", "re_sheep_6.jpg"};

  // Load and Resize each one of them individually, because they do not have
  // the same sizes, and then the resized images, will be used in the next
  // test.
  for (size_t i = 0; i < files.size(); i++)
  {
    data::Load(files.at(i), image, info, false);
    Resize(image, info, 320, 320);
    data::Save(re_sheeps.at(i), image, info, false);
  }

  std::vector<std::string> sm_sheeps =
      {"sm_sheep_1.jpg", "sm_sheep_2.jpg","sm_sheep_3.jpg", "sm_sheep_4.jpg",
       "sm_sheep_5.jpg", "sm_sheep_6.jpg"};

  data::Load(re_sheeps, images, info, false);

  ResizeImages(images, info, 160, 160);

  data::Save(sm_sheeps, images, info, false);

}
