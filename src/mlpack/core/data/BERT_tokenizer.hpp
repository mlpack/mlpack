/**
 * @file core/data/BERT_tokenizer.hpp
 * @author Ayush Singh
 *
 * Definition of the StringEncoding class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_STRING_ENCODING_HPP
#define MLPACK_CORE_DATA_STRING_ENCODING_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/boost_backport/boost_backport_string_view.hpp>
#include <mlpack/core/data/string_encoding_dictionary.hpp>
#include <mlpack/core/data/string_encoding_policies/policy_traits.hpp>
#include <vector>
#include <mlpack/core/util/to_lower.hpp>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <string>

namespace mlpack {
namespace data {

/**
 * This class provides a dictionary interface for the purpose of string
 * encoding. It works like an adapter to the internal dictionary.
 *
 * @tparam Token Type of the token that the dictionary stores.
 */
const std::wstring stripChar = L" \t\n\r\v\f";
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t, std::wstring>;

class BasicTokenizer {
public:
	BasicTokenizer(bool doLowerCase);
    std::vector<std::wstring> tokenize(const std::string& text) const;

private:
    std::wstring cleanText(const std::wstring& text) const;
    bool isWhitespace(const wchar_t& ch) const;
    bool isPunctuation(const wchar_t& ch) const;
    bool isStripChar(const wchar_t& ch) const;
    std::wstring strip(const std::wstring& text) const;
    std::vector<std::wstring> split(const std::wstring& text) const;
    std::wstring runStripAccents(const std::wstring& text) const;
    std::vector<std::wstring> runSplitOnPunc(const std::wstring& text) const;

    bool mDoLowerCase;
};


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

class WordpieceTokenizer {
public:
    WordpieceTokenizer(std::shared_ptr<Vocab> vocab, const std::wstring& unkToken = L"[UNK]", size_t maxInputCharsPerWord=200);
    std::vector<std::wstring> tokenize(const std::wstring& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    std::wstring mUnkToken;
    size_t mMaxInputCharsPerWord;
};

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


class FullTokenizer {
public:
    FullTokenizer(const std::string& vocabFile, bool doLowerCase = true);
    std::vector<std::wstring> tokenize(const std::string& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    InvVocab mInvVocab;
    std::string mVocabFile;
    bool mDoLowerCase;
    BasicTokenizer mBasicTokenizer;
    WordpieceTokenizer mWordpieceTokenizer;
};

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


} // namespace ann
} // namespace mlpack


#endif
