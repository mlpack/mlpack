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
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_ZS) return true;
    return false;
}

bool BasicTokenizer::isPunctuation(const wchar_t& ch) const {
    if (ch[i] == '!' || ch[i] == ',' || ch[i] == ';' || ch[i] == '.' || ch[i] == '?' ||   
       ch[i] == '-' || ch[i] == '\'' || ch[i] == '\"' || ch[i] == ':' || ch[i] == '(' ||
       ch[i] == ')' || ch[i] == '[' || ch[i] == ']' || ch[i] == '{' || ch[i] == '}' ) return true;
    
    return false;
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
    std::wstring nText = convertToUnicode(text);
    nText = cleanText(nText);

    const std::vector<std::wstring>& origTokens = whitespaceTokenize(nText);
    std::vector<std::wstring> splitTokens;
    for (std::wstring token : origTokens) {
        if (mDoLowerCase) {
            token = ToLower(token);
        }
        const auto& tokens = runSplitOnPunc(token);
        splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
    }
    return whitespaceTokenize(boost::join(splitTokens, L" "));
}

