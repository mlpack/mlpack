#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
class QAction;
class QMenu;
class QToolBar;
class QLabel;
class QScrollArea;
class ImageView;
class QTimer;
namespace cv { class VideoCapture; class Mat; }

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = 0);

signals:

public slots:
  void loadImage();
  void loadCamera();
  void captureTimeOut();
protected:
  QAction     *loadImageAct, *loadCameraAct;
  QMenu       *fileMenu;
  QToolBar    *toolbar;
  ImageView   *imageLabel;
  QScrollArea *scrollArea;
  QTimer      *captureTimer;
  cv::VideoCapture *cam;

  void createActions();
  void createMenu();
  void processImage(const cv::Mat& img, cv::Mat& pImg);
};

#endif // MAIN_WINDOW_H
