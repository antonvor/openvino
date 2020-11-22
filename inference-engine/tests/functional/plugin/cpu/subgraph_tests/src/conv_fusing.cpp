// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "subgraph_tests/include/conv_fusing.hpp"
#include "ngraph_functions/builders.hpp"

using ngraph::helpers::QuantizationGranularity;

namespace LayerTestsDefinitions {

std::string ConvFusingTest::getTestCaseName(const testing::TestParamInfo<convFusingTestParams> &obj) {
    ConvSpecificParams convSpecificParams;
    QuantParams quantParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    FusingCase fusingCase;
    std::tie(convSpecificParams, quantParams, fusingCase, netPrecision, inputShape, targetDevice) = obj.param;

    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels) = convSpecificParams;

    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    InferenceEngine::Precision fqPrec;
    std::tie(quantLevels, quantGranularity, fqPrec) = quantParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "Levels=" << quantLevels << "_";
    result << "QuantGranularity=" << quantGranularity << "_";
    result << "fq0PRC=" << fqPrec.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    result << "fusingCase=" << fusingCase << "_";

    return result.str();
}

void ConvFusingTest::SetUp() {
    ConvSpecificParams convSpecificParams;
    QuantParams quantParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    FusingCase fusingCase;
    std::tie(convSpecificParams, quantParams, fusingCase, netPrecision, inputShape, targetDevice) = this->GetParam();

    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels) = convSpecificParams;

    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    InferenceEngine::Precision fqPrec;
    std::tie(quantLevels, quantGranularity, fqPrec) = quantParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto makeFakeQuantizeNode = [ngPrc, quantLevels, quantGranularity](const ngraph::Output<ngraph::Node> &in,
            std::vector<size_t> inputShape, InferenceEngine::Precision prec) -> std::shared_ptr<ngraph::Node> {
        std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
        if (quantGranularity == ngraph::helpers::Perchannel)
            dataFqConstShapes[1] = inputShape[1];
        size_t constDataSize = ngraph::shape_size(dataFqConstShapes);
        std::vector<float> inputLowData(constDataSize), inputHighData(constDataSize), outputLowData(constDataSize), outputHighData(constDataSize);
        for (int i = 0; i < constDataSize; i++) {
            inputLowData[i] = 0;
            inputHighData[i] = 255;
            outputLowData[i] = prec == InferenceEngine::Precision::I8 ? -128 : 0;
            outputHighData[i] = prec == InferenceEngine::Precision::I8 ? 127 : 255;
        }
        return ngraph::builder::makeFakeQuantize(in, ngPrc, quantLevels, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);
    };

    auto dataFq = makeFakeQuantizeNode(paramOuts[0], inputShape, fqPrec);

    std::vector<size_t> weightsShape = {convOutChannels, inputShape[1]};
    std::vector<float> weightsData;
    weightsShape.insert(weightsShape.end(), kernel.begin(), kernel.end());
    auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShape, weightsData, weightsData.empty());
    auto weightsFq = makeFakeQuantizeNode(weightsNode, weightsShape, InferenceEngine::Precision::I8);

    ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
    auto conv = std::make_shared<ngraph::opset5::Convolution>(dataFq, weightsFq, stride, padBegin, padEnd, dilation,
                                                              padType);

    if (fusingCase == FusingCase::case1) {
        auto clamp1 = ngraph::builder::makeActivation(conv, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-200.0f, 200.0f});
        auto clamp2 = ngraph::builder::makeActivation(clamp1, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-100.0f, 100.0f});
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clamp2)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvFusing");
    } else if (fusingCase == FusingCase::case2) {
        auto convOutShape = conv->get_shape();
        std::vector<size_t> scalesShape(convOutShape.size(), 1);
        scalesShape[1] = convOutShape[1];

        auto scales = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);
        auto shifts = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);

        auto multiply = std::make_shared<ngraph::opset5::Multiply>(conv, scales);
        auto add = std::make_shared<ngraph::opset5::Add>(multiply, shifts);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvFusing");
    }
}

void ConvFusingTestFP32::SetUp() {
    ConvSpecificParams convSpecificParams;
    QuantParams quantParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    FusingCase fusingCase;
    std::tie(convSpecificParams, quantParams, fusingCase, netPrecision, inputShape, targetDevice) = this->GetParam();

    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels) = convSpecificParams;

    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    InferenceEngine::Precision fqPrec;
    std::tie(quantLevels, quantGranularity, fqPrec) = quantParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> weightsShape = {convOutChannels, inputShape[1]};
    weightsShape.insert(weightsShape.end(), kernel.begin(), kernel.end());
    int N = convOutChannels * inputShape[1] * kernel[0] * kernel[1];
    std::vector<float> weightsData(N);
    for (int i = 0; i < N; i++) {
        weightsData[i] = i % 10;
    }
    auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShape, weightsData, weightsData.empty());

    ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
    auto conv = std::make_shared<ngraph::opset5::Convolution>(paramOuts[0], weightsNode, stride, padBegin, padEnd, dilation,
                                                              padType);

    if (fusingCase == FusingCase::case1) {
        auto clamp1 = ngraph::builder::makeActivation(conv, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-200.0f, 200.0f});
        auto clamp2 = ngraph::builder::makeActivation(clamp1, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-100.0f, 100.0f});
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clamp2)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvFusing");
    } else if (fusingCase == FusingCase::case2) {
        auto clampOutShape = conv->get_shape();
        std::vector<size_t> scalesShape(clampOutShape.size(), 1);
        scalesShape[1] = clampOutShape[1];

        auto scales = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);
        auto shifts = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);

        auto multiply = std::make_shared<ngraph::opset5::Multiply>(conv, scales);
        auto add = std::make_shared<ngraph::opset5::Add>(multiply, shifts);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
        function = std::make_shared<ngraph::Function>(results, params, "ConvFusing");
    }
}

TEST_P(ConvFusingTest, CompareWithRefs) {
    Run();
}

TEST_P(ConvFusingTestFP32, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<size_t> numOutChannels = {3, 24, 48};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> inputShapes2D = {{1, 3, 3, 3}};
const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t >> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};

const auto conv2DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels)
);

const std::vector<size_t> levels = {256};
const std::vector<QuantizationGranularity> granularity = {Pertensor};

const auto quantParams_u8i8 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::U8)
);

INSTANTIATE_TEST_CASE_P(smoke_FusingConv_u8i8, ConvFusingTest,
                        ::testing::Combine(
                                conv2DParams,
                                quantParams_u8i8,
                                ::testing::Values(1, 2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvFusingTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_FusingConv_fp32, ConvFusingTestFP32,
                        ::testing::Combine(
                                conv2DParams,
                                quantParams_u8i8,
                                ::testing::Values(1, 2),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConvFusingTestFP32::getTestCaseName);

} // namespace

