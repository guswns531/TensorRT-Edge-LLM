/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "builder/visualBuilder.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::builder;

TEST(VisualBuilderConfigTest, SmallProfileIsDisabledByDefault)
{
    VisualBuilderConfig const config;

    EXPECT_EQ(config.smallProfileMaxImageTokens, 0);
    EXPECT_EQ(config.toJson().at("small_profile_max_image_tokens"), 0);
}

TEST(VisualBuilderConfigTest, SmallProfileRoundTripsThroughJson)
{
    VisualBuilderConfig config;
    config.smallProfileMaxImageTokens = 840;

    VisualBuilderConfig const parsed = VisualBuilderConfig::fromJson(config.toJson());

    EXPECT_EQ(parsed.smallProfileMaxImageTokens, 840);
}

TEST(VisualBuilderConfigTest, LegacyJsonDisablesSmallProfile)
{
    nlohmann::json config = VisualBuilderConfig{}.toJson();
    config.erase("small_profile_max_image_tokens");

    VisualBuilderConfig const parsed = VisualBuilderConfig::fromJson(config);

    EXPECT_EQ(parsed.smallProfileMaxImageTokens, 0);
}
