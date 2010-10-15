#ifndef IMAGE_VIEW_H
#define IMAGE_VIEW_H

#include <QLabel>
namespace cv { class Mat; }

class ImageView : public QLabel
{
    Q_OBJECT
public:
    explicit ImageView(QWidget *parent = 0);

signals:

public slots:
  void setImage(const QImage& image);
  void setImage(const cv::Mat& image);
};

#endif // IMAGE_VIEW_H
