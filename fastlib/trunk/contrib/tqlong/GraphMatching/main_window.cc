#include <QtGui>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "main_window.h"
#include "image_view.h"
#include "feature_detector.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
  imageLabel = new ImageView;

  scrollArea = new QScrollArea;
  scrollArea->setBackgroundRole(QPalette::Dark);
  scrollArea->setWidget(imageLabel);
  setCentralWidget(scrollArea);

  createActions();
  createMenu();
}

void MainWindow::createActions()
{
  loadImageAct = new QAction(tr("&Open Image..."), this);
  loadImageAct->setShortcut(tr("Ctrl+O"));
  connect(loadImageAct, SIGNAL(triggered()), this, SLOT(loadImage()));

  loadCameraAct = new QAction(tr("&Camera..."), this);
  loadCameraAct->setShortcut(tr("Ctrl+M"));
  connect(loadCameraAct, SIGNAL(triggered()), this, SLOT(loadCamera()));

  captureTimer = new QTimer(this);
  connect(captureTimer, SIGNAL(timeout()), this, SLOT(captureTimeOut()));
}

void MainWindow::createMenu()
{
  fileMenu = new QMenu(tr("&File"), this);
  fileMenu->addAction(loadImageAct);
  fileMenu->addAction(loadCameraAct);

  toolbar = new QToolBar(this);
  toolbar->addAction(loadImageAct);
  toolbar->addAction(loadCameraAct);

  menuBar()->addMenu(fileMenu);
  addToolBar(toolbar);
}

void MainWindow::loadImage()
{
  QString fileName = QFileDialog::getOpenFileName(this, "Open an image", "/home/tqlong/Pictures", "JPG files (*.jpg);; All files (*)");
  if (fileName.isEmpty()) return;
//  QImage image(fileName);
//  if (image.isNull()) {
//    QMessageBox::information(this, tr("Graph Matching"), tr("Cannot load %1.").arg(fileName));
//    return;
//  }
//  imageLabel->setImage(image);
  cv::Mat img = cv::imread(fileName.toStdString());
  if (img.empty()) {
    QMessageBox::information(this, tr("Graph Matching"), tr("Cannot load %1.").arg(fileName));
    return;
  }
  cv::Mat processedImg;
  processImage(img, processedImg);
  imageLabel->setImage(processedImg);
}

void MainWindow::loadCamera()
{
  cam = new cv::VideoCapture(0);
  if (!cam->isOpened())
  {
    QMessageBox::information(this, tr("Graph Matching"), tr("Cannot load camera"));
    return;
  }
  captureTimer->start(5);
}

void MainWindow::captureTimeOut()
{
  cv::Mat image;
  (*cam) >> image;
  if (image.empty())
  {
    QMessageBox::information(this, tr("Graph Matching"), tr("No image loaded via camera"));
    captureTimer->stop();
    delete cam;
    return;
  }
  imageLabel->setImage(image);
}

void MainWindow::processImage(const cv::Mat& img, cv::Mat& pImg)
{
  cv::Mat gray;
  cv::cvtColor(img, gray, CV_RGB2GRAY);
  pImg = img.clone();
  FeatureDetector fd(gray, 100, 150);
  fd.draw(pImg);
}
