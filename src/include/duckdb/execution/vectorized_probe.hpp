#include <vector>
#include <cstdint>
#include <cstddef>

namespace duckdb {

void VectorizedProbe_AVX2(const int32_t *probe_keys, size_t probe_count,
                          const int32_t *hash_table_keys, size_t table_count,
                          std::vector<size_t> &matched_indices);

void VectorizedProbe_NEON(const int32_t *probe_keys, size_t probe_count,
const int32_t *hash_table_keys, size_t table_count,
std::vector<size_t> &matched_indices);

}
