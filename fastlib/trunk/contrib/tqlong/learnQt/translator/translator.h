#ifndef TRANSLATOR_H
#define TRANSLATOR_H

#include <QThread>
#include <QUrl>
#include <QVector>
#include <QMap>

class Translator : public QThread
{
  Q_OBJECT
public:
  explicit Translator(const QString& html, const QUrl& url, int id, QObject *parent = 0);
  void run();
  void setStringtoTranslate(const QString& html, const QUrl& url, int id);
  const QString& getTranslatedHtml() const { return translatedHtml_; }
  const QUrl& getUrl() const { return url_; }
  int getID() const { return id_; }

  static void loadDictionary();
signals:

public slots:
private:
  QString html_;
  QString translatedHtml_;
  QUrl url_;
  int id_; // to match with the caller id

  typedef QMap<QString, QStringList> Dictionary;

  static QVector<Dictionary> dictionaries;
  static void loadDictionary(const QString& filename);
  static QStringList splitHtml(const QString& html);
  static bool isTag(const QString& html);
public:
  static QString translate(const QString& s);
  static bool convert(const QString& s, QString& r);
};

#endif // TRANSLATOR_H
