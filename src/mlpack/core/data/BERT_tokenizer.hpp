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

static bool isStripChar(const wchar_t& ch);

static std::wstring strip(const std::wstring& text);

static std::vector<std::wstring> split(const std::wstring& text);

static std::vector<std::wstring> whitespaceTokenize(const std::wstring& text);

void convertToUnicode(const std::string &s, std::wstring &ws);

void convertFromUnicode(const std::wstring &ws, std::string &s);

class WordpieceTokenizer {
public:
    WordpieceTokenizer(std::shared_ptr<Vocab> vocab, const std::wstring& unkToken = L"[UNK]", size_t maxInputCharsPerWord=200);
    std::vector<std::wstring> tokenize(const std::wstring& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    std::wstring mUnkToken;
    size_t mMaxInputCharsPerWord;
};

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


} // namespace ann
} // namespace mlpack

// Include implementation.
#include "BERT_tokenizer_impl.hpp"


#endif
