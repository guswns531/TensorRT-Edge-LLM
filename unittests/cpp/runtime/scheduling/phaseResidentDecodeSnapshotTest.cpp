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

#include "runtime/phase/mechanism/phaseResidentDecodeSnapshot.h"
#include "runtime/scheduling/phaseQueueScheduler.h"

#include <gtest/gtest.h>
#include <limits>

using namespace trt_edgellm::rt;

TEST(PhaseResidentDecodeSnapshotTest, FreezesMeasuredReferenceInMicrosecondsAtCommitEpoch)
{
    auto const reference = phaseResidentDecodeReference(2.0F, 7U);
    EXPECT_TRUE(reference.valid);
    EXPECT_DOUBLE_EQ(reference.serviceUs, 2000.0);
    EXPECT_EQ(reference.source, PhaseServiceReferenceSource::kRuntimeCovering);
    EXPECT_EQ(reference.epoch, 7U);
}

TEST(PhaseResidentDecodeSnapshotTest, MissingMeasurementDoesNotCreateAColdReference)
{
    for (std::optional<float> const sample : {std::optional<float>{}, std::optional<float>{0.0F},
             std::optional<float>{-1.0F}, std::optional<float>{std::numeric_limits<float>::infinity()},
             std::optional<float>{std::numeric_limits<float>::quiet_NaN()}})
    {
        auto const reference = phaseResidentDecodeReference(sample, 7U);
        EXPECT_FALSE(reference.valid);
        EXPECT_EQ(reference.source, PhaseServiceReferenceSource::kUnknown);
        EXPECT_DOUBLE_EQ(reference.serviceUs, 0.0);
        EXPECT_EQ(reference.epoch, 7U);
    }
}

TEST(PhaseResidentDecodeSnapshotTest, StageChangesDoNotResetTheTokenCommitClockOrReference)
{
    PhaseResidentDecodeObservation observed;
    observed.requestId = 42U;
    observed.reference = phaseResidentDecodeReference(2.0F, 7U);
    observed.lastTokenCommittedHostNs = 1000000U;
    for (auto const stage : {PhaseResidentDecodeStage::kQueued, PhaseResidentDecodeStage::kInFlight,
             PhaseResidentDecodeStage::kSampling, PhaseResidentDecodeStage::kCapacityWait})
    {
        observed.stage = stage;
        auto const age = phaseResidentDecodeAge(observed, 7000000U);
        ASSERT_TRUE(age.has_value());
        EXPECT_DOUBLE_EQ(*age, 3.0);
        EXPECT_EQ(observed.reference.epoch, 7U);
        EXPECT_DOUBLE_EQ(observed.reference.serviceUs, 2000.0);
    }
    observed.reference = phaseResidentDecodeReference(4.0F, 8U);
    observed.lastTokenCommittedHostNs = 7000000U;
    ASSERT_TRUE(phaseResidentDecodeAge(observed, 7000000U).has_value());
    EXPECT_DOUBLE_EQ(*phaseResidentDecodeAge(observed, 7000000U), 0.0);
    EXPECT_EQ(observed.reference.epoch, 8U);
}

TEST(PhaseResidentDecodeSnapshotTest, RejectsUnavailableClocksAndUnmeasuredDenominators)
{
    PhaseResidentDecodeObservation observed;
    observed.reference = phaseResidentDecodeReference(2.0F, 7U);
    EXPECT_FALSE(phaseResidentDecodeAge(observed, 7000000U).has_value());
    observed.lastTokenCommittedHostNs = 8000000U;
    EXPECT_FALSE(phaseResidentDecodeAge(observed, 7000000U).has_value());
    observed.lastTokenCommittedHostNs = 1000000U;
    observed.reference.source = PhaseServiceReferenceSource::kColdFallback;
    EXPECT_FALSE(phaseResidentDecodeAge(observed, 7000000U).has_value());
    observed.reference.source = PhaseServiceReferenceSource::kStaticProfile;
    EXPECT_FALSE(phaseResidentDecodeAge(observed, 7000000U).has_value());
    observed.reference.source = PhaseServiceReferenceSource::kRuntimeExact;
    observed.reference.valid = false;
    EXPECT_FALSE(phaseResidentDecodeAge(observed, 7000000U).has_value());
    observed.reference.valid = true;
    observed.reference.serviceUs = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(phaseResidentDecodeAge(observed, 7000000U).has_value());
}

TEST(PhaseResidentDecodeSnapshotTest, CountsAllResidentStagesWithoutInventingMissingServiceAges)
{
    PhaseResidentDecodeSnapshot snapshot;
    snapshot.hostSnapshotNs = 7000000U;
    for (auto const stage :
        {PhaseResidentDecodeStage::kUnknown, PhaseResidentDecodeStage::kQueued, PhaseResidentDecodeStage::kInFlight,
            PhaseResidentDecodeStage::kSampling, PhaseResidentDecodeStage::kCapacityWait})
    {
        PhaseResidentDecodeObservation observed;
        observed.requestId = snapshot.requests.size();
        observed.stage = stage;
        observed.candidateReady = stage == PhaseResidentDecodeStage::kQueued;
        observed.lastTokenCommittedHostNs = 1000000U;
        if (stage != PhaseResidentDecodeStage::kUnknown)
        {
            observed.reference
                = phaseResidentDecodeReference(stage == PhaseResidentDecodeStage::kSampling ? 1.0F : 2.0F, 7U);
        }
        snapshot.requests.push_back(observed);
    }
    auto const summary = phaseSummarizeResidentDecode(snapshot);
    for (size_t const count : summary.stageCounts)
    {
        EXPECT_EQ(count, 1U);
    }
    EXPECT_EQ(summary.measuredReferences, 4U);
    EXPECT_EQ(summary.candidateReady, 1U);
    ASSERT_TRUE(summary.maxServiceAge.has_value());
    EXPECT_DOUBLE_EQ(*summary.maxServiceAge, 6.0);
}

TEST(PhaseResidentDecodeSnapshotTest, EmptyOrUnmeasuredSnapshotsHaveNoMaximumAge)
{
    PhaseResidentDecodeSnapshot snapshot;
    EXPECT_FALSE(phaseSummarizeResidentDecode(snapshot).maxServiceAge.has_value());
    snapshot.hostSnapshotNs = 7000000U;
    PhaseResidentDecodeObservation observed;
    observed.stage = PhaseResidentDecodeStage::kSampling;
    observed.lastTokenCommittedHostNs = 1000000U;
    snapshot.requests.push_back(observed);
    auto const summary = phaseSummarizeResidentDecode(snapshot);
    EXPECT_EQ(summary.stageCounts[3], 1U);
    EXPECT_EQ(summary.measuredReferences, 0U);
    EXPECT_FALSE(summary.maxServiceAge.has_value());
}

TEST(PhaseResidentDecodeSnapshotTest, QueueMembershipIsUncappedAndIndependentOfCandidateEligibility)
{
    PhaseQueueSchedulerConfig config;
    config.maxDecodeBatchSize = 2;
    config.eligibilityPolicy = [](PhaseWorkItem const& item, bool) { return item.requestId != 1U; };
    PhaseQueueScheduler scheduler(config);
    for (uint64_t id = 1U; id <= 5U; ++id)
    {
        scheduler.enqueueDecode({id, 128});
    }
    std::vector<uint64_t> const expected{1U, 2U, 3U, 4U, 5U};
    EXPECT_EQ(scheduler.queuedDecodeRequestIds(), expected);
    auto const candidate = scheduler.queueSnapshot(true);
    ASSERT_EQ(candidate.decodeRequestIds.size(), 2U);
    EXPECT_EQ(candidate.decodeRequestIds, (std::vector<uint64_t>{2U, 3U}));
    EXPECT_EQ(scheduler.queuedDecodeRequestIds(), expected);

    auto const dispatched = scheduler.next();
    ASSERT_EQ(dispatched.decodeBatch.size(), 2U);
    EXPECT_EQ(dispatched.decodeBatch[0].requestId, 2U);
    EXPECT_EQ(dispatched.decodeBatch[1].requestId, 3U);
    EXPECT_EQ(scheduler.queuedDecodeRequestIds(), (std::vector<uint64_t>{1U, 4U, 5U}));
}
