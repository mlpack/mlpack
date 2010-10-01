#ifndef BROWSER_TRANSLATE_H
#define BROWSER_TRANSLATE_H

#include <QtWebKit>
#include "translator.h"

class MainWidget;

class BrowserTranslate : public QWebView
{
  Q_OBJECT
public:
  explicit BrowserTranslate(MainWidget* widget = 0, QWidget *parent = 0);
  ~BrowserTranslate();
//  void load(const QUrl &url, bool translate = false);
  bool isAutoTranslate() const { return autoTranslate_; }
  void setAutoTranslate(bool autoTranslate = true) { autoTranslate_ = autoTranslate; }
  bool isTranslating() const { return translating_; }
  void setTranslating(bool translating = true) { translating_ = translating; }
protected:
  QWebView* createWindow(QWebPage::WebWindowType type );
signals:
  void doneTranslate();
public slots:
  void translateToVietnamese();
  void translateFinished();
  void loadFinishedHandle(bool);
  void loadFinishedCheckAutoTranslate(bool);
  void linkClickedHandle(QUrl);
  void increaseID() { id_++; }
  void markForDelete();
private:
  int id_; // to match with the translator id
  MainWidget* widget_;
  BrowserTranslate* tmpPage_;
  QSet<Translator*> translators_;
  bool autoTranslate_, translating_, deleting_;
};

#endif // BROWSER_TRANSLATE_H
