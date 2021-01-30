/**
 * @file core/data/BERT_tokenizer.hpp
 * @author Ayush Singh
 *
 * Definition of BERT Tokenizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_BERT_TOKENIZER_HPP
#define MLPACK_CORE_DATA_BERT_TOKENIZER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include <vector>
#include <mlpack/core/util/to_lower.hpp>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <string>

namespace mlpack {
namespace data {
/**
 * To use a pre-trained BERT model, we need to convert the input data into an
 * appropriate format so that each sentence can be sent to the pre-trained model 
 * to obtain the corresponding embedding.
 * BERT Tokenizer breaks down the input string into tokens which can be easily 
 * encoded by the BERT model.
 * 
 * For more information, see the following.
 *
 * @code
 * @inproceedings{ACL2019,
 *   title  = {BERT: Pre-training of Deep Bidirectional Transformers 
 *             for Language Understanding},
 *   author = {Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova},
 *   year   = {2019},
 *   url    = {https://arxiv.org/abs/1810.04805}
 * }
 * @endcode
 */

// All the characters which cause break in tex in a string.
const std::wstring stripChar = L" \t\n\r\v\f";

// The datatype in which the vocab file is stored. Used for mapping token with its id.
using Vocab = std::unordered_map<std::wstring, size_t>;

// The datatype in which the vocab file is stored. Used for mapping id with its token.
using InvVocab = std::unordered_map<size_t, std::wstring>;

/**
 * This class is used for basic tokenization purposes.
 * Tasks like punctuation splitting, accent splitting, 
 * whitespace  and lower casing.
 */
class BasicTokenizer {
public:
	/**
	 * Enable a check condition for conversion to lower case.
	 * 
	 * @param doLowerCase Pass true if you want to process text after 
	 * converting all characters to lower case.
	 */
	BasicTokenizer(bool doLowerCase);

    /**
     * Execute basic tokenization by calling all the functions of this
     * class in a sequential manner.
     *
     * @param text The input text.
     */
    std::vector<std::wstring> tokenize(const std::string& text) const;

private:
	/**
	 * Perform invalid character removal and whitespace cleanup on text.
	 *
	 * @param text The input text.
	 */
    std::wstring cleanText(const std::wstring& text) const;

    /**
     * Checks whether the input character is a whitespace character.
     *
     * @param ch The input character.
     */
    bool isWhitespace(const wchar_t& ch) const;

    /**
     * Checks whether the input character is a punctuation.
     *
     * @param ch The input character.
     */
    bool isPunctuation(const wchar_t& ch) const;

    bool isStripChar(const wchar_t& ch) const;

    std::wstring runStripAccents(const std::wstring& text) const;

    /**
     * Splits punctuation on a piece of text.
     *
     * @param text The input text.
     */
    std::vector<std::wstring> runSplitOnPunc(const std::wstring& text) const;

    bool mDoLowerCase;
};

/**
 * Given a character, this function ascertains whether it
 * is either one of '\t', '\n', '\r', '\v' or '\f'.
 *
 * @param ch The character which is checked to be either of the strip chars.
 */
static bool isStripChar(const wchar_t& ch);

/**
 * This function removes all the characters of the stripChar
 * string from the end of the given input.
 * 
 * @param text The string from whose end stripping is done.
 */
static std::wstring strip(const std::wstring& text);

/**
 * This function splits the given text w.r.t. the characters of
 * stripChar.
 *
 * @param text The string which is to be split.
 */
static std::vector<std::wstring> split(const std::wstring& text);

/**
 * Runs basic whitespace cleaning and splitting on a piece of text.
 *
 * @param text The string which is stripped and split on.
 */
static std::vector<std::wstring> whitespaceTokenize(const std::wstring& text);

/**
 * Convert a std::string type variable to a std::wstring type variable.
 *
 * @param s The input string.
 * @param ws The output wstring which is the conversion of the input string.
 */
void convertToUnicode(const std::string &s, std::wstring &ws);

/**
 * Convert a std::wstring type variable to a std::string type variable.
 *
 * @param ws The input string.
 * @param s The output wstring which is the conversion of the input wstring.
 */
void convertFromUnicode(const std::wstring &ws, std::string &s);

/**
 * Function for loading the vocab.txt file from the BERT model into a
 * std::shared_ptr<std::unordered_map<std::wstring, size_t>>.
 *
 * @param vocabfile Path of BERT vocab file.
 */
static std::shared_ptr<Vocab> loadVocab(const std::string& vocabFile);


/**
 * The following class is used to run WordPiece tokenization which is a
 * subword segmentation algorithm used in natural language processing.
 */
class WordpieceTokenizer {
public:
	/**
	 * Initialize the WordpieceTokenizer class.
	 *
	 * @param vocab The pointer containing the BERT tokens along with corresponding id's.
	 * @param unkToken Token "[UNK]" of BERT Tokenizer.
	 * @param maxInputCharsPerWord Maximum characters allowed per word.
	 */
    WordpieceTokenizer(std::shared_ptr<Vocab> vocab, const std::wstring& unkToken = L"[UNK]", size_t maxInputCharsPerWord=200);

    /**
     * Tokenizes a piece of text into its word pieces.
     * This uses a greedy longest-match-first algorithm to perform tokenization
     * using the given vocabulary.
     * For example:
     * input = "unaffable"
     * output = ["un", "##aff", "##able"]
     *
     * @param text The input string which is already been passed
     * through `BasicTokenizer.
     */
    std::vector<std::wstring> tokenize(const std::wstring& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    std::wstring mUnkToken;
    size_t mMaxInputCharsPerWord;
};

/**
 * This class uses functions and classes previously defined
 * to run end-to-end tokenziation.
 */
class FullTokenizer {
public:
	/**
	 * Initialize the FullTokenizer class.
	 *
	 * @param vocabfile Path of BERT vocab file.
	 * @param doLowerCase Pass true if you want to process text after 
	 * converting all characters to lower case.
	 */
    FullTokenizer(const std::string& vocabFile, bool doLowerCase = true);

    /**
     * Function for firstly doing BasicTokenization and then
     * WordPieceTokenization.
     *
     * @param text The input string.
     */
    std::vector<std::wstring> tokenize(const std::string& text) const;

    /**
     * Given a list of tokens, this function returns their corresponding
     * Id's according to the vocab.txt file of the BERT model.
     *
     * @param text List of tokens.
     */
    std::vector<size_t> convertTokensToIds(const std::vector<std::wstring>& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    InvVocab mInvVocab;
    std::string mVocabFile;
    bool mDoLowerCase;
    BasicTokenizer mBasicTokenizer;
    WordpieceTokenizer mWordpieceTokenizer;
};

} // namespace data
} // namespace mlpack

// Include implementation.
#include "BERT_tokenizer_impl.hpp"

#endif
