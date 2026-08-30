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

// Included inside the phase server composition root. Keep this fragment
// statement-only so option parsing remains in the established translation unit.
        rt::PhaseQueueSchedulerConfig semanticSchedulerConfig;
        semanticSchedulerConfig.maxPrefillBatchSize = config.maxSupportedPrefillBatchSize;
        semanticSchedulerConfig.maxDecodeBatchSize = config.maxSupportedDecodeBatchSize;
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SCHEDULER"))
        {
            std::string const mode(value);
            ELLM_CHECK(mode == "disabled" || mode == "shadow" || mode == "active",
                "TRT_EDGELLM_GLOBAL_SCHEDULER must be disabled, shadow, or active");
            semanticSchedulerConfig.globalSchedulerMode = mode == "active" ? rt::PhaseGlobalSchedulerMode::kActive
                : mode == "shadow"                                         ? rt::PhaseGlobalSchedulerMode::kShadow
                                                                           : rt::PhaseGlobalSchedulerMode::kDisabled;
        }
        if (std::getenv("TRT_EDGELLM_GLOBAL_LEGACY_COMPATIBILITY") != nullptr)
        {
            semanticSchedulerConfig.globalSelectionMode = rt::PhaseGlobalSelectionMode::kLegacyCompatibility;
        }
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_DECODE_TPOT_TARGET_US"))
        {
            semanticSchedulerConfig.globalDecodeTpotTargetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_OVERLAP_MIN_SAMPLES"))
        {
            semanticSchedulerConfig.globalCostModelConfig.overlapMinSamples = static_cast<size_t>(std::stoull(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_DECODE_ROW_REPLACEMENT_COST_MS"))
        {
            semanticSchedulerConfig.decodeRowReplacementCostMs = std::stof(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxPrefillBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_CONTINUATION_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxContinuationPrefillBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_DECODE_BATCH"))
        {
            semanticSchedulerConfig.maxDecodeBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_OVERLAP_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxOverlapPrefillBatchSize = std::stoi(value);
        }
        ELLM_CHECK(semanticSchedulerConfig.maxPrefillBatchSize > 0
                && semanticSchedulerConfig.maxPrefillBatchSize <= config.maxSupportedPrefillBatchSize,
            "Semantic prefill batch cap is outside the engine profile");
        ELLM_CHECK(semanticSchedulerConfig.maxContinuationPrefillBatchSize >= 0
                && semanticSchedulerConfig.maxContinuationPrefillBatchSize
                    <= semanticSchedulerConfig.maxPrefillBatchSize,
            "Semantic continuation prefill batch cap is outside the configured prefill limit");
        ELLM_CHECK(semanticSchedulerConfig.maxDecodeBatchSize > 0
                && semanticSchedulerConfig.maxDecodeBatchSize <= config.maxSupportedDecodeBatchSize,
            "Semantic decode batch cap is outside the engine profile");
        semanticSchedulerConfig.maxPrefillChunkTokens = 128;
        semanticSchedulerConfig.maxOverlapPrefillTokens = 128;
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_OVERLAP_PREFILL_TOKENS"))
        {
            semanticSchedulerConfig.maxOverlapPrefillTokens = std::stoi(value);
        }
        semanticSchedulerConfig.maxPrefillBatchTokens = prefillBatchTokenBudget;
        semanticSchedulerConfig.enableRaggedPrefillBatching = config.packedPrefill;
        semanticSchedulerConfig.enablePackedPrefillTokenLayout = config.packedPrefill;
        semanticSchedulerConfig.prefillCompletionBonusTokens = 128;
        semanticSchedulerConfig.enableWavefrontPrefillBatching
            = std::getenv("TRT_EDGELLM_DISABLE_WAVEFRONT_PREFILL") == nullptr;
        semanticSchedulerConfig.maxPrefillCohortSize = semanticSchedulerConfig.maxPrefillBatchSize;
        semanticSchedulerConfig.maxPrefillCohortTurns = 8;
        semanticSchedulerConfig.enableAdaptivePrefillChunking = false;
        semanticSchedulerConfig.enableDecodeTpotTelemetry
            = std::getenv("TRT_EDGELLM_THROUGHPUT_MAX_ENCODED_VISION") != nullptr
            || std::getenv("TRT_EDGELLM_STEPWISE_ADMISSION") != nullptr
            || std::getenv("TRT_EDGELLM_PHASE_MEMORY_BROKER") != nullptr;
        semanticSchedulerConfig.minPrefillChunkTokens = semanticSchedulerConfig.maxPrefillChunkTokens;
        semanticSchedulerConfig.enableMetricsPolicy = std::getenv("TRT_EDGELLM_DISABLE_METRICS_POLICY") == nullptr;
        semanticSchedulerConfig.enablePrefillTtftHardGuard
            = std::getenv("TRT_EDGELLM_PREFILL_TTFT_HARD_GUARD") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER"))
        {
            semanticSchedulerConfig.globalSafeProbeSlackMultiplier = std::stof(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL"))
        {
            semanticSchedulerConfig.globalSafeProbeInterval = static_cast<size_t>(std::stoull(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT"))
        {
            semanticSchedulerConfig.globalExperimentalOverlapPercent = std::stoi(value);
        }
        semanticSchedulerConfig.elideVacuousGlobalDecisions
            = std::getenv("TRT_EDGELLM_DISABLE_VACUOUS_GLOBAL_ELISION") == nullptr;
        semanticSchedulerConfig.enableExternalDrainPreference
            = std::getenv("TRT_EDGELLM_PHASE_MEMORY_BROKER") != nullptr
            && std::getenv("TRT_EDGELLM_DISABLE_PHASE_MEMORY_DRAIN") == nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_DRAIN_DWELL_DISPATCHES"))
        {
            semanticSchedulerConfig.externalDrainPreferenceMinDwellDispatches = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_DRAIN_MAX_CONSECUTIVE_DISPATCHES"))
        {
            semanticSchedulerConfig.externalDrainPreferenceMaxConsecutiveDispatches
                = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_PREFILL_DECODE_PRESSURE_LIMIT"))
        {
            semanticSchedulerConfig.externalPrefillDrainDecodePressureLimit = std::stof(value);
        }
        semanticSchedulerConfig.minMetricsSamples = 2;
        semanticSchedulerConfig.prefillQueueWaitTargetUs = 5000.0;
        semanticSchedulerConfig.decodeQueueWaitTargetUs = 2000.0;
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_QUEUE_TARGET_US"))
        {
            semanticSchedulerConfig.prefillQueueWaitTargetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_DECODE_QUEUE_TARGET_US"))
        {
            semanticSchedulerConfig.decodeQueueWaitTargetUs = std::stod(value);
        }
        bool const forceDynamicDecode = std::getenv("TRT_EDGELLM_ENABLE_DYNAMIC_DECODE") != nullptr;
        semanticSchedulerConfig.enableDynamicDecodeBatching = forceDynamicDecode
            || (config.packedPrefill && std::getenv("TRT_EDGELLM_DISABLE_DYNAMIC_DECODE") == nullptr);
        semanticSchedulerConfig.enableMeasuredDecodeBatching
            = std::getenv("TRT_EDGELLM_ENABLE_MEASURED_DECODE_BATCHING") != nullptr;
        semanticSchedulerConfig.enableMeasuredPrefillBatching
            = std::getenv("TRT_EDGELLM_ENABLE_MEASURED_PREFILL_BATCHING") != nullptr;
        semanticSchedulerConfig.enableDecodeComponentObservation = semanticSchedulerConfig.enableDynamicDecodeBatching
            && std::getenv("TRT_EDGELLM_DISABLE_RUNTIME_COST_OBSERVATION") == nullptr;
        bool const asymmetricDecodeStaging = maxStableSlots > semanticSchedulerConfig.maxDecodeBatchSize
            && semanticSchedulerConfig.maxDecodeBatchSize > semanticSchedulerConfig.maxPrefillBatchSize;
        semanticSchedulerConfig.enableDecodeCohortBatching = semanticSchedulerConfig.enableDynamicDecodeBatching
            && asymmetricDecodeStaging && std::getenv("TRT_EDGELLM_DISABLE_DECODE_COHORT") == nullptr;
        if (std::getenv("TRT_EDGELLM_ENABLE_DECODE_COHORT") != nullptr)
        {
            semanticSchedulerConfig.enableDecodeCohortBatching = true;
        }
        semanticSchedulerConfig.decodeBatchCosts = {
            {1, 2048, 6.238F},
            {2, 2048, 6.294F},
            {3, 2048, 6.370F},
            {4, 2048, 6.321F},
            {5, 2048, 6.402F},
            {6, 2048, 6.360F},
            {7, 2048, 6.467F},
            {8, 2048, 6.467F},
            {9, 2048, 6.680F},
            {10, 2048, 6.617F},
            {11, 2048, 6.767F},
            {12, 2048, 6.712F},
            {13, 2048, 6.924F},
            {14, 2048, 6.790F},
            {15, 2048, 6.904F},
            {16, 2048, 6.862F},
            {17, 1024, 6.927F},
            {18, 1024, 6.988F},
            {19, 1024, 7.016F},
            {20, 1024, 7.129F},
            {21, 1024, 7.105F},
            {22, 1024, 7.082F},
            {23, 1024, 7.106F},
            {24, 1024, 7.178F},
            {25, 1024, 7.291F},
            {26, 1024, 7.278F},
            {27, 1024, 7.341F},
            {28, 1024, 7.264F},
            {29, 1024, 7.374F},
            {30, 1024, 7.435F},
            {31, 1024, 7.523F},
            {32, 1024, 7.484F},
        };
        if (config.maxSupportedDecodeBatchSize > 32)
        {
            // P8/D64 undercommitted-pool profile, measured from decode-only CUDA-event samples.
            semanticSchedulerConfig.decodeBatchCosts.insert(semanticSchedulerConfig.decodeBatchCosts.end(),
                {{40, 2048, 7.949F}, {47, 2048, 8.126F}, {63, 2048, 9.861F}, {64, 2048, 9.764F}});
        }
        if (char const* value = std::getenv("TRT_EDGELLM_SCHEDULER_COST_JSON"))
        {
            loadSchedulerCostModel(value, semanticSchedulerConfig);
        }
