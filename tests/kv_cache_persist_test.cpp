#include "core/logger.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace powerserve;

static const char *TEST_FILE_PATH = "/data/local/tmp/ziqian/powerserve/powerserve_kv_cache_persist.bin";
static constexpr size_t kFileSizeMB = 128;
static constexpr size_t kChunkBytes = 4 * 1024 * 1024;
static constexpr size_t kKVTokenWriteBytes = 512;
static constexpr size_t kReadAllocSizeBytes = 4 * 1024 * 1024;

static double now_ms() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count() / 1000.0;
}

static void print_stats(const char *name, std::vector<double> values_ms) {
    if (values_ms.empty()) {
        POWERSERVE_LOG_INFO("{}: no samples", name);
        return;
    }
    std::sort(values_ms.begin(), values_ms.end());

    double sum = 0.0;
    for (double v : values_ms) sum += v;
    const double mean = sum / values_ms.size();
    const double p50 = values_ms[values_ms.size() / 2];
    const double p95 = values_ms[(values_ms.size() * 95) / 100];
    const double min_v = values_ms.front();
    const double max_v = values_ms.back();

    POWERSERVE_LOG_INFO(
        "{}: n={} mean={:.3f} ms p50={:.3f} ms p95={:.3f} ms min={:.3f} ms max={:.3f} ms",
        name,
        values_ms.size(),
        mean,
        p50,
        p95,
        min_v,
        max_v
    );
}

static bool preallocate_file(int fd, int64_t size_bytes) {
    const int rc = posix_fallocate(fd, 0, size_bytes);
    if (rc != 0) {
        POWERSERVE_LOG_WARN("posix_fallocate failed ({}), fallback to ftruncate", rc);
    } else {
        return true;
    }
    return ftruncate(fd, size_bytes) == 0;
}

static bool sync_file(int fd) {
    return fdatasync(fd) == 0;
}

static bool pwrite_all(int fd, const uint8_t *data, size_t bytes, int64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const size_t cur = std::min(kChunkBytes, bytes - done);
        const ssize_t written = pwrite(fd, data + done, cur, offset + static_cast<int64_t>(done));
        if (written <= 0) {
            return false;
        }
        done += static_cast<size_t>(written);
    }
    return true;
}

static bool pread_all(int fd, uint8_t *data, size_t bytes, int64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const size_t cur = std::min(kChunkBytes, bytes - done);
        const ssize_t n_read = pread(fd, data + done, cur, offset + static_cast<int64_t>(done));
        if (n_read <= 0) {
            return false;
        }
        done += static_cast<size_t>(n_read);
    }
    return true;
}

int main() {
    const int64_t file_size = static_cast<int64_t>(kFileSizeMB) * 1024 * 1024;

    POWERSERVE_LOG_INFO("==== KV cache persist test ====");
    POWERSERVE_LOG_INFO("file: {}", TEST_FILE_PATH);
    POWERSERVE_LOG_INFO("file_size={} MB read_chunk={} MB", kFileSizeMB, kReadAllocSizeBytes / (1024 * 1024));

    const int flags = O_CREAT | O_RDWR | O_TRUNC;
    const mode_t mode = 0644;
    const int fd = open(TEST_FILE_PATH, flags, mode);
    if (fd < 0) {
        POWERSERVE_LOG_ERROR("open failed: {}", std::strerror(errno));
        return 1;
    }

    if (!preallocate_file(fd, file_size)) {
        POWERSERVE_LOG_ERROR("preallocate failed: {}", std::strerror(errno));
        close(fd);
        return 2;
    }

    {
        std::vector<uint8_t> k_buf(kKVTokenWriteBytes, 0x11);
        std::vector<uint8_t> v_buf(kKVTokenWriteBytes, 0x22);

        // Simulate one layer one pos: K write (512B) + V write (512B).
        const int64_t k_offset = 0;
        const int64_t v_offset = static_cast<int64_t>(kKVTokenWriteBytes);
        POWERSERVE_LOG_INFO("---- testing small write once: 2 x 512B (K+V) ----");
        const double t0 = now_ms();
        const bool k_ok = pwrite_all(fd, k_buf.data(), k_buf.size(), k_offset);
        const bool v_ok = pwrite_all(fd, v_buf.data(), v_buf.size(), v_offset);
        const double t1 = now_ms();
        if (!k_ok || !v_ok) {
            POWERSERVE_LOG_ERROR("small write failed: {}", std::strerror(errno));
            close(fd);
            return 5;
        }

        const bool sync_ok = sync_file(fd);
        const double t2 = now_ms();
        if (!sync_ok) {
            POWERSERVE_LOG_ERROR("small write sync failed: {}", std::strerror(errno));
            close(fd);
            return 6;
        }

        POWERSERVE_LOG_INFO(
            "small once: write_pair(2x512B)={:.3f} ms sync={:.3f} ms total={:.3f} ms",
            t1 - t0,
            t2 - t1,
            t2 - t0
        );
    }

    {
        std::vector<double> alloc_ms;
        std::vector<double> read_ms;
        std::vector<double> free_ms;

        // Seed the full file so sequential reads are deterministic.
        std::vector<uint8_t> seed(kReadAllocSizeBytes, 0x5A);
        for (int64_t offset = 0; offset < file_size; offset += static_cast<int64_t>(kReadAllocSizeBytes)) {
            const size_t cur_size = static_cast<size_t>(
                std::min<int64_t>(static_cast<int64_t>(kReadAllocSizeBytes), file_size - offset)
            );
            if (!pwrite_all(fd, seed.data(), cur_size, offset)) {
                POWERSERVE_LOG_ERROR("failed to seed file at offset {}: {}", offset, std::strerror(errno));
                close(fd);
                return 7;
            }
        }
        if (!sync_file(fd)) {
            POWERSERVE_LOG_ERROR("failed to sync seeded file: {}", std::strerror(errno));
            close(fd);
            return 7;
        }

        const size_t n_chunks = static_cast<size_t>(
            (file_size + static_cast<int64_t>(kReadAllocSizeBytes) - 1) / static_cast<int64_t>(kReadAllocSizeBytes)
        );
        alloc_ms.reserve(n_chunks);
        read_ms.reserve(n_chunks);
        free_ms.reserve(n_chunks);

        POWERSERVE_LOG_INFO("---- testing full scan once: 4MB chunks from file head to tail (chunks={}) ----", n_chunks);
        for (size_t c = 0; c < n_chunks; ++c) {
            const int64_t offset = static_cast<int64_t>(c) * static_cast<int64_t>(kReadAllocSizeBytes);
            const size_t cur_size = static_cast<size_t>(
                std::min<int64_t>(static_cast<int64_t>(kReadAllocSizeBytes), file_size - offset)
            );

            const double t0 = now_ms();
            uint8_t *read_buf = static_cast<uint8_t *>(std::malloc(cur_size));
            const double t1 = now_ms();
            if (!read_buf) {
                POWERSERVE_LOG_ERROR("malloc failed at chunk {}", c);
                close(fd);
                return 8;
            }

            const bool read_ok = pread_all(fd, read_buf, cur_size, offset);
            const double t2 = now_ms();

            const double t3 = now_ms();
            std::free(read_buf);
            const double t4 = now_ms();

            if (!read_ok) {
                POWERSERVE_LOG_ERROR("full scan read failed at chunk {}: {}", c, std::strerror(errno));
                close(fd);
                return 9;
            }

            alloc_ms.push_back(t1 - t0);
            read_ms.push_back(t2 - t1);
            free_ms.push_back(t4 - t3);

            POWERSERVE_LOG_INFO(
                "chunk {:02d}: alloc={:.3f} ms read={:.3f} ms free={:.3f} ms total={:.3f} ms",
                static_cast<int>(c),
                t1 - t0,
                t2 - t1,
                t4 - t3,
                t4 - t0
            );
        }

        print_stats("alloc_only_4MB", alloc_ms);
        print_stats("read_only_4MB", read_ms);
        print_stats("free_only_4MB", free_ms);
    }

    close(fd);
    return 0;
}
