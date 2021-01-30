/**
 * @file core/data/BERT_tokenizer_impl.hpp
 * @author Ayush Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_CONFUSION_MATRIX_IMPL_HPP
#define MLPACK_CORE_DATA_CONFUSION_MATRIX_IMPL_HPP

// In case it hasn't been included yet.
#include "BERT_tokenizer.hpp"

namespace mlpack {
namespace data {

static bool isStripChar(const wchar_t& ch) {
    return stripChar.find(ch) != std::wstring::npos;
}

static std::wstring strip(const std::wstring& text) {
    std::wstring ret =  text;
    if (ret.empty()) return ret;
    size_t pos = 0;
    while (pos < ret.size() && isStripChar(ret[pos])) pos++;
    if (pos != 0) ret = ret.substr(pos, ret.size() - pos);
    pos = ret.size() - 1;
    while (pos != (size_t)-1 && isStripChar(ret[pos])) pos--;
    return ret.substr(0, pos + 1);
}

static std::vector<std::wstring> split(const std::wstring& text) {
    std::vector<std::wstring>  result;
    boost::split(result, text, boost::is_any_of(stripChar));
    return result;
}

static std::vector<std::wstring> whitespaceTokenize(const std::wstring& text) {
    std::wstring rtext = strip(text);
    if (rtext.empty()) return std::vector<std::wstring>();
    return split(text);
}

void convertToUnicode(const std::string &s, std::wstring &ws) {

    std::wstring wsTmp(s.begin(), s.end());

    ws = wsTmp;
}

void convertFromUnicode(const std::wstring &ws, std::string &s) {

    std::string sTmp(ws.begin(), ws.end());

    s = sTmp;
}

static std::shared_ptr<Vocab> loadVocab(const std::string& vocabFile) {
    std::shared_ptr<Vocab> vocab(new Vocab);
    size_t index = 0;
    arma::mat temp;
    mlpack::data::DatasetInfo info;
    data::Load(vocabFile, temp, info);
    std::vector<std::string> vocabset;

    // Loading contents of vocab file from DatasetInfo object to vector<string>.
    for (size_t i = 0; i < info.NumMappings(0); ++i)
    {
      vocabset.push_back(info.UnmapString(i, 0));
    }

    // Loading contents of vocab file from vector<string> a shared_ptr pointing towards wstring variables.
    for (size_t i = 0; i < vocabset.size(); ++i) {
        std::wstring token;
        // Converting std::string datatype to std::wstring datatype.
        convertToUnicode(vocabset[i], token);
        token = strip(token);
        (*vocab)[token] = index;
        index++;
    }
    return vocab;
}

std::wstring BasicTokenizer::cleanText(const std::wstring& text) const {
    std::wstring output;
    for (const wchar_t& cp : text)  {
        if (isWhitespace(cp)) output += L" ";
        else output += cp;
    }
    return output;
}

bool BasicTokenizer::isWhitespace(const wchar_t& ch) const {
    if (ch== L' ' || ch== L'\t' || ch== L'\n' || ch== L'\r') return true;
    return false;
}

bool BasicTokenizer::isPunctuation(const wchar_t& ch) const {
    if (ch == '!' || ch == ',' || ch == ';' || ch == '.' || ch == '?' ||   
       ch == '-' || ch == '\'' || ch == '\"' || ch == ':' || ch == '(' ||
       ch == ')' || ch == '[' || ch == ']' || ch == '{' || ch == '}' ) return true;
    
    return false;
}

BasicTokenizer::BasicTokenizer(bool doLowerCase=true) 
    : mDoLowerCase(doLowerCase) {
}

std::vector<std::wstring> BasicTokenizer::runSplitOnPunc(const std::wstring& text) const {
    size_t i = 0;
    bool startNewWord = true;
    std::vector<std::wstring> output;
    while (i < text.size()) {
        wchar_t ch = text[i];
        if (isPunctuation(ch)) {
            output.push_back(std::wstring(&ch, 1));
            startNewWord = true;
        }
        else {
            if (startNewWord) output.push_back(std::wstring());
            startNewWord = false;
            output[output.size() - 1] += ch;
        }
        i++;
    }
    return output;
}

std::vector<std::wstring> BasicTokenizer::tokenize(const std::string& text) const {
    std::wstring nText;
    convertToUnicode(text, nText);
    nText = cleanText(nText);

    const std::vector<std::wstring>& origTokens = whitespaceTokenize(nText);
    std::vector<std::wstring> splitTokens;
    for (std::wstring token : origTokens) {
        if (mDoLowerCase) {
            std::string t_s; //Temporary variable used for converting character to lower.
            std::string token_s;
            //Convert to std::string format.
            convertFromUnicode(token, token_s);
            mlpack::util::ToLower(token_s, t_s);
            std::wstring t;
            //Convert to std::wstring format.
            convertToUnicode(t_s, t);
            token = t;
        }
        const auto& tokens = runSplitOnPunc(token);
        splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
    }
    return whitespaceTokenize(boost::join(splitTokens, L" "));
}

WordpieceTokenizer::WordpieceTokenizer(const std::shared_ptr<Vocab> vocab, const std::wstring& unkToken, size_t maxInputCharsPerWord)
    : mVocab(vocab),
    mUnkToken(unkToken),
    mMaxInputCharsPerWord(maxInputCharsPerWord) {
}

std::vector<std::wstring> WordpieceTokenizer::tokenize(const std::wstring& text) const {
    std::vector<std::wstring> outputTokens;
    for (auto& token : whitespaceTokenize(text)) {
        if (token.size() > mMaxInputCharsPerWord) {
            outputTokens.push_back(mUnkToken);
        }
        bool isBad = false;
        size_t start = 0;
        std::vector<std::wstring> subTokens;
        while (start < token.size()) {
            size_t end = token.size();
            std::wstring curSubstr;
            bool hasCurSubstr = false;
            while (start < end) {
                std::wstring substr = token.substr(start, end - start);
                if (start > 0) substr = L"##" + substr;
                if (mVocab->find(substr) != mVocab->end()) {
                    curSubstr = substr;
                    hasCurSubstr = true;
                    break;
                }
                end--;
            }
            if (!hasCurSubstr) {
                isBad = true;
                break;
            }
            subTokens.push_back(curSubstr);
            start = end;
        }
        if (isBad) outputTokens.push_back(mUnkToken);
        else outputTokens.insert(outputTokens.end(), subTokens.begin(), subTokens.end());
    }
    return outputTokens;
}

FullTokenizer::FullTokenizer(const std::string& vocabFile, bool doLowerCase) : 
    mVocab(loadVocab(vocabFile)), 
    mBasicTokenizer(BasicTokenizer(doLowerCase)),
    mWordpieceTokenizer(WordpieceTokenizer(mVocab)) {
    for (auto& v : *mVocab) mInvVocab[v.second] = v.first;
}

std::vector<std::wstring> FullTokenizer::tokenize(const std::string& text) const {
    std::vector<std::wstring> splitTokens;
    for (auto& token : mBasicTokenizer.tokenize(text))
        for (auto& subToken : mWordpieceTokenizer.tokenize(token))  
            splitTokens.push_back(subToken);
    return splitTokens;
}

std::vector<size_t> FullTokenizer::convertTokensToIds(const std::vector<std::wstring>& text) const {
    std::vector<size_t> ret(text.size());
    for (size_t i = 0; i < text.size(); i++) {
        ret[i] = (*mVocab)[text[i]];
    }
    return ret;
}

} // namespace data
} // namespace mlpack

#endif
