#include <QtGui>
#include "main_window.h"

int main(int argc, char** argv)
{
  QApplication app(argc, argv);
  MainWindow w;
  w.setWindowTitle("Graph Matching");
  w.resize(1024,768);
  w.show();

  return app.exec();
}
