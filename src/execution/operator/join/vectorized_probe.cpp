#include <vector>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstdio>

// SIMD headers
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace duckdb {

void VectorizedProbe_AVX2(const int32_t *probe_keys, size_t probe_count,
                          const int32_t *hash_table_keys, size_t table_count,
                          std::vector<size_t> &matched_indices) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    size_t total_matches = 0;

#ifdef __AVX2__
    for (size_t i = 0; i < probe_count; ++i) {
        int32_t probe_key = probe_keys[i];
        __m256i probe_vec = _mm256_set1_epi32(probe_key);

        for (size_t j = 0; j + 8 <= table_count; j += 8) {
            __m256i table_vec = _mm256_loadu_si256((const __m256i*)(hash_table_keys + j));
            __m256i cmp = _mm256_cmpeq_epi32(probe_vec, table_vec);
            int mask = _mm256_movemask_epi8(cmp);

            if (mask != 0) {
                for (int k = 0; k < 8; ++k) {
                    if (mask & (0xF << (k * 4))) {
                        matched_indices.push_back(j + k);
                        total_matches++;
                    }
                }
            }
        }

        // Scalar tail
        for (size_t j = table_count - (table_count % 8); j < table_count; ++j) {
            if (probe_key == hash_table_keys[j]) {
                matched_indices.push_back(j);
                total_matches++;
            }
        }
    }

    auto end = high_resolution_clock::now();
    double elapsed_ms = duration<double, std::milli>(end - start).count();
    printf("[SIMD-AVX2] Probe complete. Total matches: %zu in %.2f ms\n", total_matches, elapsed_ms);
#else
    // Scalar fallback
    for (size_t i = 0; i < probe_count; ++i) {
        int32_t probe_key = probe_keys[i];
        for (size_t j = 0; j < table_count; ++j) {
            if (probe_key == hash_table_keys[j]) {
                matched_indices.push_back(j);
                total_matches++;
            }
        }
    }
    printf("[Scalar] Probe complete. Total matches: %zu\n", total_matches);
#endif
}

#ifdef __ARM_NEON
void VectorizedProbe_NEON(const int32_t *probe_keys, size_t probe_count,
                          const int32_t *hash_table_keys, size_t table_count,
                          std::vector<size_t> &matched_indices) {
    printf("[NEON] Probing %zu probe keys vs %zu build keys\n", probe_count, table_count);
    for (size_t i = 0; i < probe_count; ++i) {
        int32_t probe_key = probe_keys[i];
        int32x4_t probe_vec = vdupq_n_s32(probe_key);

        size_t j = 0;
        for (; j + 4 <= table_count; j += 4) {
            int32x4_t table_vec = vld1q_s32(&hash_table_keys[j]);
            uint32x4_t cmp = vceqq_s32(probe_vec, table_vec);
            uint32_t result[4];
            vst1q_u32(result, cmp);  // Store comparison result

            for (int k = 0; k < 4; ++k) {
                if (result[k] == 0xFFFFFFFF) {
                    matched_indices.push_back(j + k);
                }
            }
        }

        // Scalar tail
        for (; j < table_count; ++j) {
            if (probe_key == hash_table_keys[j]) {
                matched_indices.push_back(j);
            }
        }
    }
}
#endif

} // namespace duckdb
