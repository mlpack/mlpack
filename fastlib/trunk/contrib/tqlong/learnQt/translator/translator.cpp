#include <QUrl>
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include "translator.h"

QVector<QMap<QString, QStringList> > Translator::dictionaries;

Translator::Translator(const QString& html, const QUrl& url, int id, QObject *parent) :
    QThread(parent), html_(html), url_(url), id_(id)
{
  QTextStream(stdout) << "new translator id_ = " << id_ << endl;
}

void Translator::setStringtoTranslate(const QString& html, const QUrl& url, int id)
{
  html_ = html;
  url_ = url;
  id_ = id;
}

void Translator::run()
{
  if (html_.isEmpty()) return;
  translatedHtml_.clear();
  QStringList components = splitHtml(html_);
  Q_FOREACH(const QString& component, components)
  {
    QString translatedComponent = isTag(component) ? component : translate(component);
    translatedHtml_ += translatedComponent;
  }
}

void Translator::loadDictionary()
{
  loadDictionary("Names.txt");
  loadDictionary("VietPhrase.txt");
  loadDictionary("ChinesePhienAmWords.txt");
}

void Translator::loadDictionary(const QString& filename)
{
  Dictionary dic;
  QFile file(filename);
  file.open(QIODevice::Text | QIODevice::ReadOnly);
  Q_ASSERT(file.isOpen());
  QTextStream in(&file);

  while (!in.atEnd())
  {
    QStringList line = in.readLine().simplified().split(QRegExp("="), QString::SkipEmptyParts);
    if (line.size() != 2) continue;
    QString zhWord = line[0].simplified();
    QStringList viWords = line[1].split(QRegExp("[/:]"), QString::SkipEmptyParts);
    QStringList toAdd;
    Q_FOREACH(const QString& word, viWords)
      if (!word.simplified().isEmpty()) toAdd << word.simplified();
    if (toAdd.size() == 0) continue;

    dic[zhWord] << viWords;
  }
  dictionaries << dic;
  QTextStream(stdout) << filename << ": " << dic.size() << " glossaries." << endl;
}

QStringList Translator::splitHtml(const QString& html)
{
  QStringList components;
  int endTag = 0;
  int startTag = html.indexOf(QRegExp("<[*>]>"));
  while (startTag != -1)
  {
    // from endTag to startTag-1 is a non-tag component;
    if (startTag > endTag)
      components << html.mid(endTag, startTag-endTag);
    // find the new endTag
    endTag = html.indexOf('>', startTag);
    // add the tag
    components << html.mid(startTag, endTag-startTag+1);
    endTag++;
    // find the next tag
    startTag = html.indexOf(QRegExp("<[*>]>"), endTag);
  }
  // add the remains
  if (endTag != html.size())
    components << html.mid(endTag);
  return components;
}

bool Translator::isTag(const QString& html)
{
  return *html.begin() == '<' && *html.end() == '>';
}

QString Translator::translate(const QString& s)
{
  QString result;
  int pos = 0;
  while (pos < s.size())
  {
    int length = 20 < s.size()-pos ? 20 : s.size()-pos;
    QString phrase;
    for (; length > 0; length--)
      if (convert(s.mid(pos, length), phrase)) break;

    if (length > 0) // sucessfully converted
    {
      result += phrase+" ";
      pos += length;
    }
    else
    {
      result += s[pos];
      pos++;
    }
  }
  return result;
}

bool Translator::convert(const QString& s, QString& r)
{
  Q_FOREACH(const Dictionary& dic, dictionaries)
  {
    Dictionary::const_iterator it = dic.find(s);
    if (it != dic.end())
    {
      r = (*it)[0];
      return true;
    }
  }
  return false;
}
