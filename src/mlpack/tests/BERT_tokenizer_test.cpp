/**
 * @file tests/bias_svd_test.cpp
 * @author Ayush Singh
 *
 * Test BERT Tokenizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/BERT_tokenizer.hpp>
#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::data;

template<input_type, output_type>
void BERTTokenizerRunner(input_type input_list, output_type tokenized_output_true)
{
	// Storing the predicted tokenization.
	std::vector<std::vector<std::string>> tokenized_output_pred;
	// Initializing the BERT Tokenizer, along with giving path of vocab file.
    auto tokenizer = FullTokenizer("data/BERT-vocab.txt");

    // Initializing temporary variables, used for storing processed tokens.
    vector<string> temp;
    string temp_str;

    // Running tokenization for each string in the given input vector.
    for(int i = 0;i < input_list.size();i++)
    {
    	tokens = tokenizer.tokenize(input_list[i]);
    	for(int j = 0;j < tokens.size();j++)
    	{
    		// Converting each token we got to string format and storing as temp_str.
    		convertFromUnicode(tokens[j], temp_str);
    		temp.push_back(temp_str);
    	}
    	tokenized_output_pred.push_back(temp);
    	temp = {};
    }

    // Cheking if the predicted list of tokens matches with the actual list of tokens.
    REQUIRE(tokenized_output_pred.size() == tokenized_output_true.size());
    for(i = 0;i < tokenized_output_pred.size();i++)
    {
    	REQUIRE(tokenized_output_pred[i].size() == tokenized_output_true[i].size());
    	for(int j = 0;j < tokenized_output_pred[i];j++)
    		REQUIRE(tokenized_output_pred[i][j].size() == tokenized_output_true[i][j].size());
    }
}

/**
 * Testing if the tokenization performs correctly for a string
 * containing alphabets and spaces.
 */
TEST_CASE("BERTTokenizer-BasicTests", "[BERTTokenizerTest]")
{
	std::vector<std::string> input_list = {"this test case is for simple cases",
		                      "checkinging for spliting",
		                      "hello how are you",
		                      "helloabcdef checkup"};
	std::vector<std::vector<std::string>> tokenized_output_true = 
	{
		{"this", "test", "case", "is", "for", "simple", "cases"},
		{"checking", "##ing", "for", "split", "##ing"},
		{"hello", "how", "are", "you"},
		{"hello", "##ab", "##cd", "##ef", "check", "##up"}
	};

	BERTTokenizerRunner(input_list);
}

#endif