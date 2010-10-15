#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <QObject>
#include <QVector>
#include <cv.h>

class FeatureDetector : public QObject
{
public:
  typedef cv::Point                    feature_type;
  typedef QVector<feature_type>        feature_list;

  explicit FeatureDetector(const cv::Mat& img, double t1, double t2, QObject *parent = 0)
    : QObject(parent), threshold1_(t1), threshold2_(t2)
  {
    detect(img);
  }

  explicit FeatureDetector(const cv::Mat& img, QObject *parent = 0)
    : QObject(parent), threshold1_(50), threshold2_(150)
  {
    detect(img);
  }

  void detect(const cv::Mat& img)
  {
    cv::Mat edges;
    cv::Canny(img, edges, threshold1_, threshold2_, 3);
//    QTextStream(stdout) << "depth = " << edges.depth() << " channels = " << edges.channels() << endl;
//    QTextStream(stdout) << "CV_8U = " << CV_8U << endl;
//    //cv::imshow("test", edges);
    for (int row = 0; row < edges.rows; row++)
      for (int col = 0; col < edges.cols; col++)
        if (edges.at<unsigned char>(cv::Point(col, row)) > 0)
          edgeFeatures_ << cv::Point(col, row);
 }

  void draw(cv::Mat& img) const
  {
    Q_FOREACH(const cv::Point& p, edgeFeatures_)
    {
      cv::circle(img, p, 1, cv::Scalar(0,0,192));
    }
  }

  const feature_list& features() const
  {
    return edgeFeatures_;
  }
private:
  double threshold1_, threshold2_;
  feature_list edgeFeatures_;
};

struct fp_img_ {
    int width;
    int height;
    size_t length;
    uint16_t flags;
    struct fp_minutiae *minutiae;
    unsigned char *binarized;
    unsigned char data[0];
};


#endif // FEATURE_DETECTOR_H
