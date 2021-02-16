// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/conv_dw_conv_fusing.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string ConvDWConvFusingSubgraphTest::getTestCaseName(testing::TestParamInfo<ConvDWConvFusingParams> obj) {
    std::ostringstream result;
    ConvDWConvPatternParams convDWConvPatternParams;
    SizeVector inputShape;
    std::tie(convDWConvPatternParams, inputShape) = obj.param;

    size_t numOutChannels, dwStride;
    bool withBias, withDWBias;
    std::tie(numOutChannels, dwStride, withBias, withDWBias) = convDWConvPatternParams;

    result << "O=" << numOutChannels << "_";
    result << "dwStride=" << dwStride << "_";
    result << "withBias=" << withBias << "_";
    result << "withDWBias=" << withDWBias;

    return result.str();
}

void ConvDWConvFusingSubgraphTest::CheckConvCount(size_t expectedConvCount) {
    InferenceEngine::CNNNetwork execGraphInfo = executableNetwork.GetExecGraphInfo();
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(nullptr, function);
    size_t actualConvCount = 0;
    for (const auto &node : function->get_ops()) {
        const auto & rtInfo = node->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
            IE_ASSERT(nullptr != value);
            return value->get();
        };
        if (getExecValue(ExecGraphInfoSerialization::LAYER_TYPE) == "Convolution") {
            actualConvCount++;
        }
    }

    ASSERT_EQ(expectedConvCount, actualConvCount);
}

void ConvDWConvFusingSubgraphTest::SetUp() {
    targetDevice = CommonTestUtils::DEVICE_CPU;
    ConvDWConvPatternParams convDWConvPatternParams;
    SizeVector inputShape;

    std::tie(convDWConvPatternParams, inputShape) = this->GetParam();
    size_t numOutChannels, dwStride;
    bool withBias, withDWBias;

    std::tie(numOutChannels, dwStride, withBias, withDWBias) = convDWConvPatternParams;

    auto inputParams = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

    auto conv1 = ngraph::builder::makeConvolution(paramOuts[0], ngraph::element::f32, {1, 1}, {1, 1}, {0, 0},
                                                  {0, 0}, {1, 1}, ngraph::op::PadType::EXPLICIT, numOutChannels, withBias);

    auto conv2 = ngraph::builder::makeGroupConvolution(conv1, ngraph::element::f32, {3, 3}, {dwStride, dwStride}, {1, 1}, {1, 1}, {1, 1},
                                                       ngraph::op::PadType::EXPLICIT, numOutChannels, numOutChannels, withDWBias);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(conv2)};
    function = std::make_shared<ngraph::Function>(results, inputParams, "convolutionDWConvolutionFusingPattern");
}

TEST_P(ConvDWConvFusingSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
//    CheckConvCount(2);
};

const SizeVector inputShape2D{1, 96, 224, 224};

std::vector<ConvDWConvPatternParams> ConvDWConvPatternParams2DSet = {
        ConvDWConvPatternParams{96, 1, true, true},
        ConvDWConvPatternParams{96, 1, false, true},
        ConvDWConvPatternParams{96, 2, true, true},
        ConvDWConvPatternParams{96, 2, false, true},
        // todo: [antonvor] dw convolution without biases not supported
//        ConvDWConvPatternParams{96, 1, true, false},
//        ConvDWConvPatternParams{96, 1, false, false},
//        ConvDWConvPatternParams{96, 2, true, false},
//        ConvDWConvPatternParams{96, 2, false, false},
};

const auto ConvDWConvFusingParamsSet = ::testing::Combine(
        ::testing::ValuesIn(ConvDWConvPatternParams2DSet),
        ::testing::Values(inputShape2D));

INSTANTIATE_TEST_CASE_P(smoke_ConvDWConvFusing2D, ConvDWConvFusingSubgraphTest, ConvDWConvFusingParamsSet, ConvDWConvFusingSubgraphTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions
