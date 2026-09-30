// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

std::ifstream trialInput(trialPath);
nlohmann::json trial;
trialInput >> trial;
int32_t const rows = trial.at("rows");
int32_t const split = trial.at("split");
int32_t const outputs = trial.at("output_tokens");
int32_t const rounds = trial.at("rounds");
int32_t const splitTurns = trial.at("split_turns");
int32_t const branchTurn = trial.value("branch_turn", -1);
int32_t const inspectTurn = trial.value("inspect_turn", -1);
bool const graphReplay = trial.value("graph_replay", true);
std::vector<int32_t> const graphBatches
    = trial.value("graph_batches", std::vector<int32_t>{1, split, rows - split, rows});
ELLM_CHECK(rows > 1 && rows <= decodeBatchCapacity && split > 0 && split < rows
        && outputs >= 3 && outputs <= 256 && rounds > 0 && rounds <= 10 && splitTurns > 0
        && inspectTurn < outputs - 1 && branchTurn >= -1 && branchTurn < outputs - 2
        && std::all_of(graphBatches.begin(), graphBatches.end(), [rows](int32_t batch) {
               return batch > 0 && batch <= rows;
           }),
    "Invalid async decode trial bounds");
ELLM_CHECK(!enablePrefixReuse && serverConfig.eosTokenIds.empty(), "Trial requires fixed output without prefix reuse");
auto parsed = parsePhaseIpcInput(trial.at("request").dump(), outputs);
ELLM_CHECK(parsed.valid && parsed.request.imageBuffers.empty(), "Trial requires a valid text request");
rt::LLMGenerationRequest::FormattedRequest formatted;
ELLM_CHECK(chatTemplate.apply(parsed.request, formatted, {}), "Trial chat template failed");
auto const prompt = tokenizer.encode(formatted.formattedCompleteRequest, false);
ELLM_CHECK(!prompt.empty() && prompt.size() + outputs <= static_cast<size_t>(config.maxKVCacheCapacity),
    "Trial prompt exceeds KV capacity");
if (graphReplay)
{
    semanticCoordinator.primeDecodeGraphs(graphBatches, decodeStream,
        [&](int32_t batch, cudaStream_t stream) {
            std::vector<int32_t> token{0};
            std::vector<rt::IndependentPhaseRequestView> views(static_cast<size_t>(batch));
            for (auto& view : views)
            {
                view.generatedTokens = &token;
            }
            stageTokens(views, *decodeIO, decodeMap, stream, false);
        });
}
semanticCoordinator.setGraphCaptureEnabled(false);
semanticCoordinator.setMetricsCollectionEnabled(true);
float* inspectionLogits{};
if (inspectTurn >= 0)
{
    void* pinned{};
    CUDA_CHECK(cudaMallocHost(&pinned, static_cast<size_t>(config.outputVocabSize) * sizeof(float)));
    inspectionLogits = static_cast<float*>(pinned);
}
uint64_t lastCommit{};
semanticServer.setTimelineCallback([&](rt::PhaseTimelineEvent const& event) {
    if (event.stage == rt::PhaseTimelineStage::kDecodeTokenCommitted
        || event.stage == rt::PhaseTimelineStage::kPrefillTokenCommitted)
    {
        lastCommit = event.timestampNs;
    }
});
nlohmann::json episodes = nlohmann::json::array();
uint64_t nextId{1000000};
for (int32_t round = -1; round < rounds; ++round)
{
    for (int32_t order{}; order < 3; ++order)
    {
        int32_t const variant = (round + 1 + order) % 3;
        std::vector<uint64_t> ids;
        std::vector<int32_t> slots;
        std::map<uint64_t, std::vector<int32_t>> tokens;
        auto collect = [&]() {
            while (auto event = semanticServer.tryPopToken())
            {
                tokens[event->requestId].push_back(event->tokenId);
            }
        };
        semanticServer.setDecodeDispatchBlocked(true);
        for (int32_t row{}; row < rows; ++row)
        {
            ids.push_back(nextId++);
            auto const submission = semanticServer.submit(ids.back(), prompt, outputs);
            ELLM_CHECK(submission.status == rt::IndependentPhaseServerStatus::kAdmitted, "Trial admission failed");
            slots.push_back(submission.kvSlotId);
            size_t polls{};
            // Canonical BS1 prefill removes prefill batch shape from this comparison.
            while (semanticCoordinator.scheduler().decodeQueueSize() != static_cast<size_t>(row + 1))
            {
                ELLM_CHECK(++polls < 10000000U, "Trial prefill did not drain");
                semanticServer.poll();
                collect();
            }
            ELLM_CHECK(ownership.length(slots.back()) == static_cast<int32_t>(prompt.size()),
                "Trial prefix KV length mismatch");
        }
        semanticServer.setDecodeDispatchBlocked(false);
        collect();
        nlohmann::json dispatches = nlohmann::json::array();
        nlohmann::json inspected = nlohmann::json::array();
        uint64_t const begin = rt::phaseTimelineNowNs();
        uint64_t firstDrain{}, secondDrain{};
        for (int32_t turn{}; turn < outputs - 1; ++turn)
        {
            int32_t batch = rows;
            if (variant == 2 && (branchTurn < 0 || turn == branchTurn))
            {
                batch = 1;
            }
            else if (variant == 1 && (branchTurn < 0 ? turn < splitTurns : turn == branchTurn))
            {
                batch = split;
            }
            for (int32_t offset{}; offset < rows; offset += batch)
            {
                int32_t const count = std::min(batch, rows - offset);
                rt::PhaseGlobalActionCandidate candidate;
                int32_t const width = semanticSchedulerConfig.runtimeDecodeContextBucketTokens;
                candidate.key = {rt::PhaseGlobalActionKind::kDecode, count, 0, 1,
                    (static_cast<int32_t>(prompt.size()) + turn + width - 1) / width, 0};
                candidate.key.executionVariant = semanticCoordinator.scheduler().executionVariantFor(candidate.key, 0);
                candidate.primaryRequestIds.assign(ids.begin() + offset, ids.begin() + offset + count);
                candidate.primaryStableSlotIds.assign(slots.begin() + offset, slots.begin() + offset + count);
                candidate.requestIds = candidate.primaryRequestIds;
                uint64_t const previousCommit = lastCommit;
                size_t const metricCount = semanticCoordinator.metrics().size();
                ELLM_CHECK(semanticServer.dispatchGlobalAction(candidate), "Trial action dispatch failed");
                size_t polls{};
                while (semanticCoordinator.busy() || semanticServer.hasPendingSampling())
                {
                    ELLM_CHECK(++polls < 10000000U, "Trial decode did not drain");
                    semanticServer.pollCompletions();
                }
                uint64_t const drained = rt::phaseTimelineNowNs();
                ELLM_CHECK(semanticCoordinator.metrics().size() == metricCount + 1, "Trial metric missing");
                auto const& metric = semanticCoordinator.metrics().back();
                auto const& timing = metric.decodeHostExecution;
                auto actualIds = metric.decodeRequestIds;
                auto expectedIds = candidate.requestIds;
                std::sort(actualIds.begin(), actualIds.end());
                std::sort(expectedIds.begin(), expectedIds.end());
                ELLM_CHECK(metric.decodeBatchSize == count && actualIds == expectedIds
                        && metric.globalCandidateParity && metric.globalActionFidelity,
                    "Trial action membership changed");
                std::vector<int32_t> actualRows;
                for (uint64_t id : metric.decodeRequestIds)
                {
                    actualRows.push_back(static_cast<int32_t>(std::find(ids.begin(), ids.end(), id) - ids.begin()));
                }
                if (turn == inspectTurn)
                {
                    auto const row = std::find(actualRows.begin(), actualRows.end(), 0);
                    if (row != actualRows.end())
                    {
                        size_t const rowBytes = static_cast<size_t>(config.outputVocabSize) * sizeof(float);
                        auto const* logits = static_cast<std::byte const*>(decodeIO->outputLogits.rawPointer());
                        CUDA_CHECK(cudaMemcpyAsync(inspectionLogits,
                            logits + static_cast<size_t>(row - actualRows.begin()) * rowBytes, rowBytes,
                            cudaMemcpyDeviceToHost, decodeStream));
                        CUDA_CHECK(cudaStreamSynchronize(decodeStream));
                        std::array<int32_t, 8> topIds{};
                        topIds.fill(-1);
                        for (int32_t token{}; token < config.outputVocabSize; ++token)
                        {
                            for (size_t rank{}; rank < topIds.size(); ++rank)
                            {
                                if (topIds[rank] < 0 || inspectionLogits[token] > inspectionLogits[topIds[rank]])
                                {
                                    std::move_backward(topIds.begin() + rank, topIds.end() - 1, topIds.end());
                                    topIds[rank] = token;
                                    break;
                                }
                            }
                        }
                        nlohmann::json top = nlohmann::json::array();
                        for (int32_t token : topIds)
                        {
                            top.push_back({{"token", token}, {"logit", inspectionLogits[token]}});
                        }
                        inspected.push_back({{"turn", turn}, {"actual_row", row - actualRows.begin()},
                            {"top", std::move(top)}});
                    }
                }
                dispatches.push_back({{"turn", turn}, {"batch", count}, {"offset", offset},
                    {"actual_rows", actualRows},
                    {"gpu_ms", metric.decodeGpuMs}, {"graph", timing.graphReplay},
                    {"prepare_ns", timing.prepareStartHostNs}, {"enqueue_ns", timing.executeStartHostNs},
                    {"commit_ns", lastCommit}, {"drained_ns", drained},
                    {"service_ms", static_cast<double>(drained - timing.prepareStartHostNs) / 1.0e6},
                    {"commit_to_prepare_ms", static_cast<double>(timing.prepareStartHostNs - previousCommit) / 1.0e6}});
                collect();
            }
            if (turn == 0)
            {
                firstDrain = rt::phaseTimelineNowNs();
            }
            if (turn == 1)
            {
                secondDrain = rt::phaseTimelineNowNs();
            }
        }
        size_t completed{};
        while (semanticServer.tryPopCompletion())
        {
            ++completed;
        }
        ELLM_CHECK(completed == static_cast<size_t>(rows) && semanticServer.empty()
                && ownership.availableSlots() == maxStableSlots,
            "Trial requests or ownership did not drain");
        std::vector<std::vector<int32_t>> orderedTokens;
        for (uint64_t id : ids)
        {
            ELLM_CHECK(tokens[id].size() == static_cast<size_t>(outputs), "Trial token capture incomplete");
            orderedTokens.push_back(tokens[id]);
        }
        if (round >= 0)
        {
            episodes.push_back({{"round", round}, {"order", order}, {"variant", variant}, {"slots", slots},
                {"tokens", orderedTokens}, {"dispatches", std::move(dispatches)}, {"logits", std::move(inspected)},
                {"first_tick_ms", static_cast<double>(firstDrain - begin) / 1.0e6},
                {"two_tick_ms", static_cast<double>(secondDrain - begin) / 1.0e6}});
        }
    }
}
if (inspectionLogits != nullptr)
{
    CUDA_CHECK(cudaFreeHost(inspectionLogits));
}
std::ofstream output(trial.at("output").get<std::string>());
ELLM_CHECK(output.good(), "Cannot open async trial report");
output << nlohmann::json{{"config", trial}, {"prompt_ids", prompt}, {"episodes", episodes}}.dump(2) << '\n';
output.flush();
ELLM_CHECK(output.good(), "Cannot write async trial report");
