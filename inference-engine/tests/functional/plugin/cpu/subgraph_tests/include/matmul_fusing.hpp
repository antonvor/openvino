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
        size_t,
        ngraph::helpers::QuantizationGranularity,
        InferenceEngine::Precision> QuantParams;

typedef std::tuple<
        size_t,
        QuantParams,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        LayerTestsUtils::TargetDevice> matmulFusingTestParams;

namespace LayerTestsDefinitions {

    class MatmulFusingTest : public testing::WithParamInterface<matmulFusingTestParams>, virtual public LayerTestsUtils::LayerTestsCommon {
    public:
        static std::string getTestCaseName(const testing::TestParamInfo<matmulFusingTestParams> &obj);

    protected:
        void SetUp() override;
    };

    class MatmulFusingTestFP32 : public MatmulFusingTest {
    protected:
        void SetUp() override;
    };

}  // namespace LayerTestsDefinitions
