#include <opencv/cv.h>
#include "image_view.h"

ImageView::ImageView(QWidget *parent) :
    QLabel(parent)
{
  setBackgroundRole(QPalette::Base);
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setScaledContents(true);
}

void ImageView::setImage(const QImage &image)
{
  setPixmap(QPixmap::fromImage(image));
  adjustSize();
}

void ImageView::setImage(const cv::Mat &mat)
{
  QImage image(mat.data, mat.size().width, mat.size().height, mat.step, QImage::Format_RGB888);
  image = image.rgbSwapped();
  setImage(image);
}
