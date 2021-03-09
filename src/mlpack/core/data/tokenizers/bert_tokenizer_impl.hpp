/**
 * @file core/data/tokenizers/bert_tokenizer_impl.hpp
 * @author Ayush Singh
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_TOKENIZERS_BERT_TOKENIZER_IMPL_HPP
#define MLPACK_CORE_DATA_TOKENIZERS_BERT_TOKENIZER_IMPL_HPP

// In case it hasn't been included yet.
#include "bert_tokenizer.hpp"

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

    // Loading contents of vocab file from file stream object to vector<string>.
    std::fstream f;
    f.open(vocabFile);

    std::string tkn;
    std::vector<std::string> vocabset;
    while (std::getline(f, tkn))
    {
        vocabset.push_back(tkn);
    }

    f.close();

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

std::wstring str_replacer(std::wstring str, std::wstring old_substring, std::wstring new_substring)
{
  size_t index = 0;
  while (true) {
     /* Locate the substring to replace. */
     index = str.find(old_substring, index);
     if (index == std::string::npos) break;

     /* Make the replacement. */
     str.replace(index, old_substring.size(), new_substring);

     /* Advance index forward so the next iteration doesn't pick it up as well. */
     //index += old_substring.size();
     index += old_substring.size();
  }
  return str;
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
    bool is_bert_tkn;
    for (std::wstring token : origTokens) {
        is_bert_tkn = false;
        //If is_bert_tkn is true, then we won't convert the token to smaller case.
        //The following list contains the list of BERT symbols.
        std::wstring predefined_bert_tkns[3] = {L"[CLS]", L"[SEP]", L"[MASK]"};
        for(int i=0; i<3;i++){
            if(token.find(predefined_bert_tkns[i]) != std::string::npos){
                is_bert_tkn = true;
            }
        }

        if (mDoLowerCase and !is_bert_tkn) {
            //Temporary variable used for converting string to lower.
            std::string token_s;
            //Convert to std::string format.
            convertFromUnicode(token, token_s);
            //Converting to lower case.
            transform(token_s.begin(), token_s.end(), token_s.begin(), ::tolower);
            //Convert to std::wstring format.
            convertToUnicode(token_s, token);
        }
        const auto& tokens = runSplitOnPunc(token);
        splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
    }

    //Ensuring that BERT symbols remain intact after Basic processing.
    std::wstring processed_text = str_replacer(boost::join(splitTokens, L" "), L"[ CLS ]", L"[CLS]");
    processed_text = str_replacer(processed_text, L"[ MASK ]", L"[MASK]");
    processed_text = str_replacer(processed_text, L"[ SEP ]", L"[SEP]");

    return whitespaceTokenize(processed_text);
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
        std::wstring substr_t; //Temporary string used to store the lower case version of a token.
        while (start < token.size()) {
            size_t end = token.size();
            std::wstring curSubstr;
            bool hasCurSubstr = false;
            while (start < end) {
                std::wstring substr = token.substr(start, end - start);
                if (start > 0) substr = L"##" + substr;
                
                //If the token has a BERT symbol, then we won't convert the token to smaller case.
                //The following list contains the list of BERT symbols.
                std::wstring predefined_bert_tkns[3] = {L"[CLS]", L"[SEP]", L"[MASK]"};
                for(int i=0; i<3;i++){
                    if(substr.find(predefined_bert_tkns[i]) != std::string::npos){
                        substr_t = substr;
                        break;
                    }
                    else{
                        substr_t = substr;
                        transform(substr_t.begin(), substr_t.end(), substr_t.begin(), ::tolower);
                    }
                }
                if (mVocab->find(substr_t) != mVocab->end()) {
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
