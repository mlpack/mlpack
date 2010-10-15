#include "feature_detector.h"

struct fp_img_ *fpi_img_new(size_t length)
{
    struct fp_img_ *img = (struct fp_img_ *) malloc(sizeof(*img) + length);
    memset(img, 0, sizeof(*img));
//    fp_dbg("length=%zd", length);
    img->length = length;
    return img;
}

struct fp_img_ *fpi_img_new_for_imgdev(int width, int height)
{
    struct fp_img_ *img = fpi_img_new(width * height);
    img->width = width;
    img->height = height;
    return img;
}
