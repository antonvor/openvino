// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "subgraph_tests/include/matmul_fusing.hpp"
#include "ngraph_functions/builders.hpp"

using ngraph::helpers::QuantizationGranularity;

namespace LayerTestsDefinitions {

    std::string MatmulFusingTest::getTestCaseName(const testing::TestParamInfo<matmulFusingTestParams> &obj) {
        QuantParams quantParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        std::string targetDevice;
        size_t lastDim;
        std::tie(lastDim, quantParams, netPrecision, inputShape, targetDevice) = obj.param;

        size_t quantLevels;
        QuantizationGranularity quantGranularity;
        InferenceEngine::Precision fqPrec;
        std::tie(quantLevels, quantGranularity, fqPrec) = quantParams;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "lastDim" << lastDim << "_";
        result << "Levels=" << quantLevels << "_";
        result << "QuantGranularity=" << quantGranularity << "_";
        result << "fq0PRC=" << fqPrec.name() << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;

        return result.str();
    }

    void MatmulFusingTest::SetUp() {
        QuantParams quantParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        size_t lastDim;
        std::tie(lastDim, quantParams, netPrecision, inputShape, targetDevice) = this->GetParam();

        size_t quantLevels;
        QuantizationGranularity quantGranularity;
        InferenceEngine::Precision fqPrec;
        std::tie(quantLevels, quantGranularity, fqPrec) = quantParams;

        auto inputShape2(inputShape);
        inputShape2[inputShape2.size() - 2] = inputShape[inputShape.size() - 1];
        inputShape2[inputShape2.size() - 1] = lastDim;

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

        std::vector<size_t> weightsShape = inputShape2;
        std::vector<float> weightsData;
        auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShape, weightsData, weightsData.empty());
        auto weightsFq = makeFakeQuantizeNode(weightsNode, weightsShape, InferenceEngine::Precision::I8);

        auto matmul = std::make_shared<ngraph::opset5::MatMul>(dataFq, weightsFq);
//        auto clamp1 = ngraph::builder::makeActivation(matmul, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-200.0f, 200.0f});

        auto matmulOutShape = matmul->get_shape();
        std::vector<size_t> scalesShape(matmulOutShape.size(), 1);
        scalesShape[1] = matmulOutShape[1];

        auto scales = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);
        auto shifts = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);

        auto multiply = std::make_shared<ngraph::opset5::Multiply>(matmul, scales);
        auto add = std::make_shared<ngraph::opset5::Add>(multiply, shifts);

//        auto clamp2 = ngraph::builder::makeActivation(clamp1, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-1.0f, 1.0f});

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
        function = std::make_shared<ngraph::Function>(results, params, "MatmulFusing");
    }

    TEST_P(MatmulFusingTest, CompareWithRefs) {
        Run();
    };

    void MatmulFusingTestFP32::SetUp() {
        QuantParams quantParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape;
        size_t lastDim;
        std::tie(lastDim, quantParams, netPrecision, inputShape, targetDevice) = this->GetParam();

        size_t quantLevels;
        QuantizationGranularity quantGranularity;
        InferenceEngine::Precision fqPrec;
        std::tie(quantLevels, quantGranularity, fqPrec) = quantParams;

        auto inputShape2(inputShape);
        inputShape2[inputShape2.size() - 2] = inputShape[inputShape.size() - 1];
        inputShape2[inputShape2.size() - 1] = lastDim;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));



        std::vector<size_t> weightsShape = inputShape2;
        std::vector<float> weightsData;
        auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShape, weightsData, weightsData.empty());

        auto matmul = std::make_shared<ngraph::opset5::MatMul>(paramOuts[0], weightsNode);
        auto clamp1 = ngraph::builder::makeActivation(matmul, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-200.0f, 200.0f});

//        auto matmulOutShape = matmul->get_shape();
//        std::vector<size_t> scalesShape(matmulOutShape.size(), 1);
//        scalesShape[1] = matmulOutShape[1];
//
//        auto scales = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);
//        auto shifts = ngraph::builder::makeConstant(ngPrc, scalesShape, std::vector<size_t>{}, true);
//
//        auto multiply = std::make_shared<ngraph::opset5::Multiply>(matmul, scales);
//        auto add = std::make_shared<ngraph::opset5::Add>(multiply, shifts);

        auto clamp2 = ngraph::builder::makeActivation(clamp1, ngPrc, ngraph::helpers::ActivationTypes::Clamp, {}, {-1.0f, 1.0f});

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clamp2)};
        function = std::make_shared<ngraph::Function>(results, params, "MatmulFusing");
    }

    TEST_P(MatmulFusingTestFP32, CompareWithRefs) {
        Run();
    };

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };
    const std::vector<std::vector<size_t >> inputShapes2D = {{5, 6}};


    const std::vector<size_t> levels = {256};
    const std::vector<QuantizationGranularity> granularity = {Pertensor};

    const auto quantParams_u8i8 = ::testing::Combine(
            ::testing::ValuesIn(levels),
            ::testing::ValuesIn(granularity),
            ::testing::Values(InferenceEngine::Precision::U8)
    );

    INSTANTIATE_TEST_CASE_P(smoke_FusingMatmul_u8i8, MatmulFusingTest,
                            ::testing::Combine(
                                    ::testing::Values(4),
                                    quantParams_u8i8,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::ValuesIn(inputShapes2D),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MatmulFusingTest::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_FusingMatmul_fp32, MatmulFusingTestFP32,
                            ::testing::Combine(
                                    ::testing::Values(4),
                                    quantParams_u8i8,
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::ValuesIn(inputShapes2D),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MatmulFusingTestFP32::getTestCaseName);

} // namespace

