// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <debug.h>
#include "ie_core.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/multiply_add.hpp"

namespace LayerTestsDefinitions {
std::string MultiplyAddLayerTest::getTestCaseName(const testing::TestParamInfo<MultiplyAddParamsTuple> &obj) {
    std::vector<size_t> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inputShapes, netPrecision, targetName) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void MultiplyAddLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(inputShape, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> constShape(inputShape.size(), 1);
    constShape[1] = inputShape[1];

    auto const_mul = ngraph::builder::makeConstant(ngPrc, constShape, std::vector<float>{-1.0f, 2.0f, -3.0f});
    auto mul = std::make_shared<ngraph::opset3::Multiply>(paramOuts[0], const_mul);
    auto const_add = ngraph::builder::makeConstant(ngPrc, constShape, std::vector<float>{1.0f, -2.0f, 3.0f});
    auto add = std::make_shared<ngraph::opset3::Add>(mul, const_add);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(add)};
    function = std::make_shared<ngraph::Function>(results, params, "multiplyAdd");
}

TEST_P(MultiplyAddLayerTest, CompareWithRefs) {
    Run();
};
} // namespace LayerTestsDefinitions
