#include "backend/opencl/opencl_backend.hpp"
#include "backend/cpu_buffer.hpp"

#include "core/logger.hpp"

#include <cstdlib>

namespace powerserve::opencl {

std::pair<Tensor, Tensor> OpenCLBackend::get_cache_tensors(size_t L) const {
    POWERSERVE_ASSERT(m_kv && m_kv->allocated());
    POWERSERVE_ASSERT(L < m_kv->key.size());

    Shape shape{m_kv->kv_dim, m_kv->max_seq_len, 1, 1};
    Tensor k_cache(DataType::FP32, shape);
    Tensor v_cache(DataType::FP32, shape);
    k_cache.m_data = m_kv->key[L];
    v_cache.m_data = m_kv->value[L];

    return {k_cache, v_cache};
}

static inline bool kv_dbg_enabled() {
    static int on = []() -> int {
        const char* e = std::getenv("POWERSERVE_KV_DBG");
        return e ? std::atoi(e) : 0;
    }();
    return on != 0;
}

void OpenCLBackend::reset_kv_batch_size(const size_t batch_size) const {
    if (!m_kv) {
        POWERSERVE_LOG_ERROR("reset_kv_batch_size called but KVCache not allocated");
        return;
    }
    if (batch_size == 0) {
        POWERSERVE_LOG_ERROR("KVCache v0 expects batch_size > 0");
        return;
    }
    if (m_kv->batch_size == batch_size && m_kv->positions.size() == batch_size) {
        return;
    }
    std::vector<size_t> new_positions(batch_size, 0);
    const size_t copy_count = std::min(batch_size, m_kv->positions.size());
    for (size_t i = 0; i < copy_count; ++i) {
        new_positions[i] = m_kv->positions[i];
    }
    m_kv->positions = std::move(new_positions);
    m_kv->batch_size = batch_size;
}

void OpenCLBackend::add_cache(const Tensor *k,
                              const Tensor *v,
                              size_t L,
                              const std::vector<int> &pos,
                              size_t head_id) {
    (void)head_id;

    if (!m_kv) { POWERSERVE_LOG_ERROR("add_cache: KVCache not allocated"); return; }
    if (!k || !v) { POWERSERVE_LOG_ERROR("add_cache: null tensor"); return; }

    if (pos.empty()) {
        POWERSERVE_LOG_ERROR("add_cache v0 expects non-empty pos");
        return;
    }

    if (pos.size() != m_kv->batch_size) {
        POWERSERVE_LOG_ERROR("add_cache v0 expects pos.size()==batch_size, got pos.size()={} batch_size={}",
                             pos.size(), m_kv->batch_size);
        return;
    }

    if (L >= m_kv->key.size()) {
        POWERSERVE_LOG_ERROR("add_cache: invalid layer {}", L);
        return;
    }

    if (k->m_dtype != DataType::FP32 || v->m_dtype != DataType::FP32) {
        POWERSERVE_LOG_ERROR("add_cache v0 only supports FP32");
        return;
    }

    const size_t kv_dim = m_kv->kv_dim;
    const size_t batch_size = pos.size();

    if (k->m_shape[0] != kv_dim || k->m_shape[1] != batch_size || k->m_shape[2] != 1 || k->m_shape[3] != 1 ||
        v->m_shape[0] != kv_dim || v->m_shape[1] != batch_size || v->m_shape[2] != 1 || v->m_shape[3] != 1) {
        POWERSERVE_LOG_ERROR("add_cache shape mismatch: expect {{kv_dim, batch_size,1,1}} (kv_dim={}, batch_size={})",
                             kv_dim, batch_size);
        return;
    }

    const size_t token_bytes = kv_dim * sizeof(float);
    const size_t batch_stride_bytes = token_bytes;

    Shape sTok{kv_dim, 1, 1, 1};

    try {
        auto &k_parent = *m_kv->key[L];
        auto &v_parent = *m_kv->value[L];
        auto &k_src_parent = const_cast<Tensor *>(k)->get<OpenCLBuffer>();
        auto &v_src_parent = const_cast<Tensor *>(v)->get<OpenCLBuffer>();

        for (size_t b = 0; b < batch_size; ++b) {
            if (pos[b] < 0) {
                POWERSERVE_LOG_ERROR("add_cache: invalid pos[{}]={}", b, pos[b]);
                return;
            }
            const size_t slot = static_cast<size_t>(pos[b]);
            if (slot >= m_kv->max_seq_len) {
                POWERSERVE_LOG_ERROR("KVCache overflow: slot {} max_seq_len {}", slot, m_kv->max_seq_len);
                return;
            }

            const size_t dst_offset = slot * token_bytes;
            const size_t src_offset = b * batch_stride_bytes;

            if (dst_offset + token_bytes > k_parent.get_size()) {
                POWERSERVE_LOG_ERROR("[KV][ADD_CACHE] K view out of range: L={} slot={} offset={} token_bytes={} parent_size={}",
                                     L, slot, dst_offset, token_bytes, k_parent.get_size());
                return;
            }
            if (dst_offset + token_bytes > v_parent.get_size()) {
                POWERSERVE_LOG_ERROR("[KV][ADD_CACHE] V view out of range: L={} slot={} offset={} token_bytes={} parent_size={}",
                                     L, slot, dst_offset, token_bytes, v_parent.get_size());
                return;
            }

            auto k_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(k_parent, sTok, dst_offset);
            auto v_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(v_parent, sTok, dst_offset);
            auto k_src_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(k_src_parent, sTok, src_offset);
            auto v_src_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(v_src_parent, sTok, src_offset);
            if (!k_view || !v_view || !k_src_view || !v_src_view) {
                POWERSERVE_LOG_ERROR("add_cache: create_buffer_view failed (L={}, slot={}, b={})", L, slot, b);
                return;
            }

            Tensor t_dst_k(DataType::FP32, sTok);
            Tensor t_dst_v(DataType::FP32, sTok);
            Tensor t_src_k(DataType::FP32, sTok);
            Tensor t_src_v(DataType::FP32, sTok);
            t_dst_k.m_data = k_view;
            t_dst_v.m_data = v_view;
            t_src_k.m_data = k_src_view;
            t_src_v.m_data = v_src_view;

            this->copy(&t_dst_k, &t_src_k);
            this->copy(&t_dst_v, &t_src_v);

            if (kv_dbg_enabled() && b == 0) {
                Tensor host_k(DataType::FP32, Shape{8,1,1,1});
                host_k.m_data = powerserve::CPUBuffer::create_buffer<float>(Shape{8,1,1,1});
                Tensor k_first8(DataType::FP32, Shape{8,1,1,1});
                auto k_first8_view = powerserve::opencl::OpenCLBuffer::create_buffer_view<float>(*k_view, Shape{8,1,1,1}, /*offset=*/0);
                k_first8.m_data = k_first8_view;
                this->copy(&host_k, &k_first8);

                auto &hb = host_k.get<powerserve::CPUBuffer>();
                float *p = (float*)hb.m_data;
                POWERSERVE_LOG_INFO("[KV][ADD_CACHE] L={} slot={} K_first8: {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                                    L, slot, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
            }

            if (b < m_kv->positions.size()) {
                m_kv->positions[b] = slot + 1;
            }
        }
    } catch (const std::bad_cast &e) {
        POWERSERVE_LOG_ERROR("add_cache expects OpenCLBuffer in KVCache: {}", e.what());
        return;
    }
}

void OpenCLBackend::ensure_kv_cache_allocated_v0(size_t batch_size) {
    const int n_layers_i = m_llm.n_layers;
    const int seq_len_i  = m_llm.seq_len;
    const int kv_dim_i   = m_llm.kv_dim;

    if (n_layers_i <= 0 || seq_len_i <= 0 || kv_dim_i <= 0) {
        POWERSERVE_LOG_WARN("KVCache v0 skipped: invalid llm config n_layers={}, seq_len={}, kv_dim={}",
                            n_layers_i, seq_len_i, kv_dim_i);
        return;
    }

    const size_t n_layers    = static_cast<size_t>(n_layers_i);
    const size_t max_seq_len = static_cast<size_t>(seq_len_i);
    const size_t kv_dim      = static_cast<size_t>(kv_dim_i);

    if (!m_kv) m_kv = std::make_unique<powerserve::opencl::OpenCLKV>();

    if (m_kv->spec_matches(n_layers, kv_dim, max_seq_len, batch_size)) {
        return;
    }

    m_kv->kv_dim = kv_dim;
    m_kv->max_seq_len = max_seq_len;
    m_kv->batch_size = batch_size;
    m_kv->positions.assign(batch_size, 0);
    m_kv->key.clear();
    m_kv->value.clear();
    m_kv->key.resize(n_layers);
    m_kv->value.resize(n_layers);

    const Shape sKV{kv_dim, max_seq_len, 1, 1};
    for (size_t L = 0; L < n_layers; ++L) {
        m_kv->key[L]   = this->create_buffer(sKV, DataType::FP32);
        m_kv->value[L] = this->create_buffer(sKV, DataType::FP32);
        if (!m_kv->key[L] || !m_kv->value[L]) {
            POWERSERVE_LOG_ERROR("KVCache v0 alloc failed at layer {}", L);
            m_kv.reset();
            return;
        }
    }

    POWERSERVE_LOG_INFO("KVCache v0 allocated: layers={}, kv_dim={}, max_seq_len={}, batch_size={}",
                        n_layers, kv_dim, max_seq_len, batch_size);
}

} // namespace powerserve::opencl
