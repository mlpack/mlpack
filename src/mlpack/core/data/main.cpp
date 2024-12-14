
#include "resize_crop_image.hpp"

int main()
{  
  int image_width;
  int image_height;
  int num_channel;
  unsigned char* image_data;
  std::string filename = "image_test.jpg";
  load_from_file(filename.c_str(), image_data, &image_width, &image_height, &num_channel);

  std::cout << "image width: " << image_width << std::endl;
  std::cout << "image height : " << image_height << std::endl;
  std::cout << "Num channel: " << num_channel << std::endl;


  // STB resize and cropping
  unsigned char* frame_buffer_out =
    (unsigned char*)malloc(720 * 720 * num_channel * sizeof (unsigned char));
  resize_and_crop_stb(image_data, image_width, image_height,
                      frame_buffer_out, 720, 720, num_channel);

  // save_texture_to_file("resized_stb_720.jpg", 720, 720, frame_buffer_out);

  unsigned char* frame_buffer_out_2 =
    (unsigned char*)malloc(320 * 320 * num_channel * sizeof (unsigned char));
  resize_and_crop_stb(frame_buffer_out, 720, 720,
                      frame_buffer_out_2, 320, 320, num_channel);

  float features[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT];
  size_t feature_ix = 0;
  //#pragma omp parallel for
  for (int cx = 0; cx < EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * num_channel; cx = cx + 3)
  {
    uint8_t r = frame_buffer_out[cx];
    uint8_t g = frame_buffer_out[cx + 1];
    uint8_t b = frame_buffer_out[cx + 2];
    features[feature_ix++] = (r << 16) + (g << 8) + b;
  }

  std::cout << "features values: " << features[0] << " " << features[1] <<" " << features[2]
    << " " << features[3] << std::endl;

  save_texture_to_file("resized_stb_320.jpg", 320, 320, frame_buffer_out_2);
 
  free(frame_buffer_out);
  free(frame_buffer_out_2);
  free(image_data);
}
