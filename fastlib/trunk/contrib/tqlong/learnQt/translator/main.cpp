#include <QtGui/QApplication>
#include "mainwidget.h"

int main(int argc, char *argv[])
{
  for (int i =0; i < argc; i ++)
    QTextStream(stdout) << i << " = " << argv[i] << endl;
  QApplication a(argc, argv);
  MainWidget w;
  w.show();

  return a.exec();
}
