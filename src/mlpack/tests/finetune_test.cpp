#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_function.hpp>

#include <mlpack/methods/finetune/finetune.hpp>
#include <mlpack/methods/finetune/softmax_finetune.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

#include <iostream>

using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::distribution;

void initialize_last_col(arma::mat &inout)
{
    static double index = 0;
    for(size_t i = 0; i != inout.n_rows; ++i){
        inout(i, inout.n_cols - 1) = index++;
    }
}

arma::mat extractW1B1(arma::mat const &input)
{
    return input.submat(0, 0, (input.n_rows - 1)/2 - 1, input.n_cols - 1);
}

void sigmoid(arma::mat const &input, arma::mat &output)
{
    output = (1.0 / (1 + arma::exp(-input)));
}

size_t biasSize(arma::mat const &input)
{
    return (input.n_rows - 1)/2;
}

size_t encoderSize(arma::mat const &input)
{
    return (input.n_rows - 1) / 2 * input.n_cols;
}

size_t w1Size(arma::mat const &input)
{
    return (input.n_rows - 1) / 2 * (input.n_cols - 1);
}

struct TestData
{
    using FineTuneFunc =
    mlpack::nn::FineTuneFunction<
    SoftmaxRegressionFunction,
    mlpack::nn::SoftmaxFineTune
    >;

    TestData()
    {
        inputs.emplace_back(arma::randu<arma::mat>(3, 2));
        inputs.emplace_back(arma::randu<arma::mat>(2, 2));
        inputs.emplace_back(arma::randu<arma::mat>(2, 2));        

        arma::Row<size_t> labels(2);
        labels(0) = 0;
        labels(1) = 1;
        sm.reset(new SoftmaxRegressionFunction(inputs[2], labels, 2));

        params.emplace_back(arma::randu<arma::mat>(5,4));
        params.emplace_back(arma::randu<arma::mat>(5,3));
        params.emplace_back(sm->GetInitialPoint());

        initialize_last_col(params[0]);
        initialize_last_col(params[1]);

        for(size_t i = 0; i != inputs.size(); ++i){
            inputsPtr.emplace_back(&inputs[i]);
            paramsPtr.emplace_back(&params[i]);
        }

        finetune.reset(new FineTuneFunc(inputsPtr, paramsPtr, *sm));
    }

    size_t TotalEncoderSize() const
    {
        size_t totalEncoderSize = 0;
        for(size_t i = 0; i != params.size() - 1; ++i)
        {
            totalEncoderSize += encoderSize(params[i]);
        }

        return totalEncoderSize;
    }

    std::vector<arma::mat> inputs;
    std::vector<arma::mat> params;
    std::unique_ptr<SoftmaxRegressionFunction> sm;
    std::unique_ptr<FineTuneFunc> finetune;
    std::vector<arma::mat*> inputsPtr;
    std::vector<arma::mat*> paramsPtr;
};

BOOST_AUTO_TEST_SUITE(FinetuneTest);

BOOST_AUTO_TEST_CASE(FinetuneTestInitializeWeights)
{
    TestData testdata;

    auto const &initialPoint = testdata.finetune->GetInitialPoint();

    size_t index = 0;
    for(size_t i = 0; i != testdata.params.size() - 1; ++i){
        auto const w1b1 = extractW1B1(testdata.params[i]);
        for(size_t j = 0; j != w1b1.size(); ++j, ++index){
            BOOST_REQUIRE_CLOSE(w1b1[j], initialPoint[index], 1e-5);
        }
    }

    auto const &SoftmaxParams = testdata.params[testdata.params.size() - 1];
    for(size_t i = 0; i != SoftmaxParams.size(); ++i, ++index)
    {
        BOOST_REQUIRE_CLOSE(SoftmaxParams[i], initialPoint[index], 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(FinetuneTestEvaluate)
{
    TestData testdata;
    auto const &initialPoint = testdata.finetune->GetInitialPoint();
    auto const Eval1 = testdata.finetune->Evaluate(initialPoint);

    auto const &smInitialPoint = testdata.sm->GetInitialPoint();
    auto const Eval2 = testdata.sm->Evaluate(smInitialPoint);

    BOOST_REQUIRE_CLOSE(Eval1, Eval2, 1e-5);
}

//! Test the function UpdateInputData(private function) is valid or not
BOOST_AUTO_TEST_CASE(FinetuneTestUpdateInput)
{
    TestData testdata;
    auto const &initialPoint = testdata.finetune->GetInitialPoint();
    testdata.finetune->Evaluate(initialPoint);

    arma::mat gradient;
    testdata.finetune->Gradient(initialPoint, gradient);

    for(size_t i = 0; i != testdata.inputs.size() - 1; ++i){
        arma::mat const &param = testdata.params[i];
        arma::mat const w1 = param.submat(0, 0,
                                          (param.n_rows - 1)/2 - 1, param.n_cols - 2);
        arma::mat const b1 = param.submat(0, param.n_cols - 1,
                                          (param.n_rows - 1)/2 - 1, param.n_cols - 1);
        arma::mat sigmoidOutput;
        sigmoid(arma::reshape(w1, testdata.inputs[i+1].n_rows, testdata.inputs[i].n_rows) *
                testdata.inputs[i] + arma::repmat(b1, 1, testdata.inputs[i].n_cols),
                sigmoidOutput);
        for(size_t j = 0; j != sigmoidOutput.n_elem; ++j){
            BOOST_REQUIRE_CLOSE(sigmoidOutput(j), testdata.inputs[i + 1](j), 1e-5);
        }
        testdata.inputs[i + 1] = sigmoidOutput;
    }
}

BOOST_AUTO_TEST_CASE(FinetuneTestSoftmaxGradient)
{
    TestData testdata;
    auto initialPoint = testdata.finetune->GetInitialPoint();
    testdata.finetune->Evaluate(initialPoint);

    arma::mat gradient;
    testdata.finetune->Gradient(initialPoint, gradient);

    auto softmaxGradientCheck = [&testdata]
            (arma::mat &theta, size_t row, size_t col)
    {
        double const Epsillon = 1e-5;
        auto const OriginValue = theta(row, col);
        theta(row, col) = OriginValue + Epsillon;
        double const Plus = testdata.finetune->Evaluate(theta);
        theta(row, col) = OriginValue - Epsillon;
        double const Minus = testdata.finetune->Evaluate(theta);
        theta(row, col) = OriginValue;

        return (Plus - Minus) / (2 * Epsillon);
    };

    size_t const totalEncoderSize = testdata.TotalEncoderSize();

    for(size_t i = totalEncoderSize;
        i != totalEncoderSize + testdata.params.back().n_elem; ++i)
    {
        BOOST_REQUIRE_CLOSE(softmaxGradientCheck(initialPoint, i, 0),
                            gradient(i), 1e-4);
    }
}

BOOST_AUTO_TEST_CASE(FinetuneTestUpdateParameters)
{
    TestData testdata;
    auto initialPoint = testdata.finetune->GetInitialPoint();
    testdata.finetune->Evaluate(initialPoint);

    arma::mat gradient;
    testdata.finetune->Gradient(initialPoint, gradient);
    testdata.finetune->UpdateParameters(gradient);

    //test the parameters of encoders(w1 + b1) copy to correct place or not
    size_t index = 0;
    for(size_t i = 0; i != testdata.params.size() - 1; ++i)
    {
        arma::mat const &encoderParam = testdata.params[i];
        size_t const hiddenSize = (encoderParam.n_rows - 1)/2;
        arma::mat const &subparam =
                encoderParam.submat(0, 0, hiddenSize - 1,
                                    encoderParam.n_cols - 1);
        for(size_t j = 0; j != subparam.n_elem; ++j, ++index){
            BOOST_REQUIRE_CLOSE(gradient(index), subparam(j), 1e-5);
        }
    }

    //test the parameters of softmax copy to correct place or not
    arma::mat const &softmaxParam = testdata.params.back();
    for(size_t i = 0; i != softmaxParam.n_elem; ++i)
    {
        BOOST_REQUIRE_CLOSE(gradient(index++), softmaxParam(i), 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(FinetuneTestGradient)
{
    //The initial value of this test is come from the working octave codes
    TestData testdata;
    auto &sae1_params = testdata.params[0];
    sae1_params(0, 0) = 0.48924;
    sae1_params(0, 1) = 0.61225;
    sae1_params(0, 2) = -0.9208;
    sae1_params(0, 3) = 0;
    sae1_params(1, 0) = -0.30737;
    sae1_params(1, 1) = -0.95354;
    sae1_params(1, 2) = -0.8032;
    sae1_params(1, 3) = 0;

    auto &sae2_params = testdata.params[1];
    sae2_params(0, 0) = -0.77665;
    sae2_params(0, 1) = 0.43561;
    sae2_params(0, 2) = 0;
    sae2_params(1, 0) = 0.15307;
    sae2_params(1, 1) = 0.20814;
    sae2_params(1, 2) = 0;

    auto &sm_params = testdata.params[2];
    sm_params(0, 0) = -2.9973e-003;
    sm_params(1, 0) = 9.3189e-003;
    sm_params(0, 1) = -1.0638e-003;
    sm_params(1, 1) = -4.1538e-004;

    auto &sae1_input = testdata.inputs[0];
    sae1_input(0, 0) = 0.0013;
    sae1_input(0, 1) = 0.3503;
    sae1_input(1, 0) = 0.1933;
    sae1_input(1, 1) = 0.8228;
    sae1_input(2, 0) = 0.5850;
    sae1_input(2, 1) = 0.1741;

    testdata.finetune->InitializeWeights();
    auto initialPoint = testdata.finetune->GetInitialPoint();
    testdata.sm->Lambda() = 1e-4;
    testdata.finetune->Evaluate(initialPoint);
    testdata.finetune->Evaluate(initialPoint);

    arma::mat gradient;
    testdata.finetune->Gradient(initialPoint, gradient);

    //The results are come from octave codes too
    arma::mat result;
    result<<4.6563e-05<<-2.2390e-05<<8.2524e-05<<-3.7923e-05
         <<-5.9316e-05<<3.3925e-05<<-7.6629e-06<<1.2897e-05
        <<-1.6003e-04<<-9.1052e-06<<6.7504e-05<<3.2826e-06
       <<2.5440e-05<<2.8253e-07<<-1.3669e-02<<1.3669e-02
      <<3.9510e-04<<-3.9525e-04;
    for(size_t i = 0; i !=gradient.n_elem; ++i){
        BOOST_REQUIRE_CLOSE(gradient(i), result(i), 1e-2);
    }
}

BOOST_AUTO_TEST_SUITE_END();
