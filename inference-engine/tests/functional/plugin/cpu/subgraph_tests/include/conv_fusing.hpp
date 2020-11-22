// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

typedef std::tuple<
        InferenceEngine::SizeVector,    // kernel
        InferenceEngine::SizeVector,    // stride
        std::vector<ptrdiff_t>,         // padBegin
        std::vector<ptrdiff_t>,         // padEnd
        InferenceEngine::SizeVector,    // dilation
        size_t                          // convOutChannels
        > ConvSpecificParams;

typedef std::tuple<
        size_t,
        ngraph::helpers::QuantizationGranularity,
        InferenceEngine::Precision> QuantParams;

enum FusingCase {
    case1 = 1,
    case2 = 2
};

typedef std::tuple<
        ConvSpecificParams,
        QuantParams,
        FusingCase,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        LayerTestsUtils::TargetDevice> convFusingTestParams;

namespace LayerTestsDefinitions {

class ConvFusingTest : public testing::WithParamInterface<convFusingTestParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convFusingTestParams> &obj);

protected:
    void SetUp() override;
};

class ConvFusingTestFP32 : public ConvFusingTest {
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
