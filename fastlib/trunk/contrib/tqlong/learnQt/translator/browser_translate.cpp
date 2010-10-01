#include <QTextStream>
#include "mainwidget.h"
#include "browser_translate.h"

BrowserTranslate::BrowserTranslate(MainWidget* widget, QWidget *parent) :
    QWebView(parent), id_(0), widget_(widget), autoTranslate_(false), translating_(false), deleting_(false)
{
  connect(this, SIGNAL(loadStarted()), this, SLOT(increaseID()));
//  connect(this, SIGNAL(loadFinished(bool)), this, SLOT(loadFinishedHandle(bool)));
//  connect(this, SIGNAL(linkClicked(QUrl)), this, SLOT(linkClickedHandle(QUrl)));
  connect(this, SIGNAL(loadFinished(bool)), this, SLOT(loadFinishedCheckAutoTranslate(bool)));
  this->page()->settings()->setAttribute(QWebSettings::AutoLoadImages, false);
  this->page()->settings()->setAttribute(QWebSettings::PluginsEnabled, false);
  this->page()->settings()->setAttribute(QWebSettings::JavascriptEnabled, false);
  this->page()->settings()->setAttribute(QWebSettings::JavaEnabled, false);
}

BrowserTranslate::~BrowserTranslate()
{
  QTextStream(stdout) << "delete a browser\n";
}

void BrowserTranslate::translateToVietnamese()
{
  QString html = this->page()->mainFrame()->toHtml();
  Translator* translateEngine = new Translator(html, this->page()->mainFrame()->baseUrl(), this->id_, this);
  connect(translateEngine, SIGNAL(finished()), this, SLOT(translateFinished()));
  translators_ << translateEngine;
  translateEngine->start();
}

void BrowserTranslate::translateFinished()
{
  Translator* translateEngine = (Translator*) sender();
  if (id_ == translateEngine->getID())
  {
    setTranslating();
    QTextStream(stdout) << "start copy translate html id_ = " << id_ << " url = " << translateEngine->getUrl().toString() << endl;
    this->setHtml(translateEngine->getTranslatedHtml(), translateEngine->getUrl());
  }
  else
  {
    QTextStream(stdout) << "translator missed: id_ = " << id_
        << " translator->id_ = " << translateEngine->getID() << " url = " << translateEngine->getUrl().toString() << endl;
  }
  translators_.remove(translateEngine);
  if (deleting_) markForDelete();
  translateEngine->deleteLater();
  emit doneTranslate();
}

void BrowserTranslate::loadFinishedHandle(bool)
{
  BrowserTranslate* tmpPage = (BrowserTranslate*) sender();
  QTextStream(stdout) << "load finished: " << tmpPage->page()->mainFrame()->baseUrl().toString() << endl;

  QString html = tmpPage->page()->mainFrame()->toHtml();
  Translator* translateEngine = new Translator(html, tmpPage->page()->mainFrame()->baseUrl(), id_, this);
  connect(translateEngine, SIGNAL(finished()), this, SLOT(translateFinished()));
  translateEngine->start();

  tmpPage->deleteLater();
}

void BrowserTranslate::linkClickedHandle(QUrl url)
{
  load(url);
}

QWebView* BrowserTranslate::createWindow(QWebPage::WebWindowType)
{
  return widget_ ? widget_->addPage() : this;
}

//void BrowserTranslate::load(const QUrl &url, bool translate)
//{
//  id_++;
//  if (!translate)
//    QWebView::load(url);
//  else
//  {
//    QTextStream(stdout) << "start translate: " << url.toString() << endl;
//    BrowserTranslate* tmpPage = new BrowserTranslate();
//    tmpPage->id_ = this->id_;
//    connect(tmpPage, SIGNAL(loadFinished(bool)), this, SLOT(loadFinishedHandle(bool)));
//    tmpPage->load(url);
//  }
//}

void BrowserTranslate::loadFinishedCheckAutoTranslate(bool)
{
  if (isTranslating())
  {
    QTextStream(stdout) << "done copy translated html id = " << id_ << endl;
    setTranslating(false);
    return;
  }
  if (!isAutoTranslate()) return;
  QTextStream(stdout) << "auto translate starting id_ = " << id_ << endl;
  translateToVietnamese();
}

void BrowserTranslate::markForDelete()
{
  deleting_ = true;
  if (translators_.isEmpty()) deleteLater();
}
