// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace powerserve::ggml {

bool cluster_profile_enabled();
int cluster_profile_report_interval();

void cluster_profile_reset(size_t n_layers);
void cluster_profile_record_prefill(size_t tokens, int64_t ns);
void cluster_profile_record_global_cluster_build(int64_t ns);
void cluster_profile_record_decode_token(int64_t ns);

void cluster_profile_record_layer_op(size_t layer_id, const char *name, int64_t ns);
void cluster_profile_record_cluster_update(size_t layer_id, int64_t copy_ns, int64_t update_ns);
void cluster_profile_record_cluster_select(size_t layer_id, int64_t ns);
void cluster_profile_record_cluster_fetch(size_t layer_id, bool cache_hit, int64_t ns);
void cluster_profile_record_cluster_attn_compute(size_t layer_id, int64_t ns);
void cluster_profile_record_cluster_selection(size_t layer_id, size_t selected_clusters, size_t selected_tokens);
void cluster_profile_record_layer_cluster_stats(size_t layer_id, size_t cluster_count, size_t total_tokens, size_t max_tokens);

void cluster_profile_maybe_report(const std::string &model_id, bool force = false);

} // namespace powerserve::ggml
