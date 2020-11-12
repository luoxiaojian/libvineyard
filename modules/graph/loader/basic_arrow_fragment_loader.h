/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef MODULES_GRAPH_LOADER_BASIC_ARROW_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_BASIC_ARROW_FRAGMENT_LOADER_H_

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arrow/util/config.h"
#include "client/client.h"
#include "grape/worker/comm_spec.h"
#include "io/io/local_io_adaptor.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/utils/partitioner.h"
#include "graph/utils/table_shuffler.h"
#include "graph/utils/table_shuffler_beta.h"
#include "graph/vertex_map/arrow_vertex_map.h"

namespace vineyard {
template <typename T>
class OidSet {
  using oid_t = T;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;

 public:
  boost::leaf::result<void> BatchInsert(
      const std::shared_ptr<arrow::Array>& arr) {
    if (vineyard::ConvertToArrowType<oid_t>::TypeValue() != arr->type()) {
      RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                      "OID_T is not same with arrow::Column(" +
                          arr->type()->ToString() + ")");
    }
    auto oid_arr = std::dynamic_pointer_cast<oid_array_t>(arr);
    for (int64_t i = 0; i < oid_arr->length(); i++) {
      oids.insert(oid_arr->GetView(i));
    }
    return boost::leaf::result<void>();
  }

  boost::leaf::result<void> BatchInsert(
      const std::shared_ptr<arrow::ChunkedArray>& chunked_arr) {
    for (auto chunk_idx = 0; chunk_idx < chunked_arr->num_chunks();
         chunk_idx++) {
      BOOST_LEAF_CHECK(BatchInsert(chunked_arr->chunk(chunk_idx)));
    }
    return boost::leaf::result<void>();
  }

  boost::leaf::result<std::shared_ptr<oid_array_t>> ToArrowArray() {
    typename vineyard::ConvertToArrowType<oid_t>::BuilderType builder;

    for (auto& oid : oids) {
      builder.Append(oid);
    }

    std::shared_ptr<oid_array_t> oid_arr;
    ARROW_OK_OR_RAISE(builder.Finish(&oid_arr));
    return oid_arr;
  }

 private:
  std::unordered_set<internal_oid_t> oids;
};

template <typename OID_T, typename VID_T, typename PARTITIONER_T>
class BasicArrowFragmentLoader {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using partitioner_t = PARTITIONER_T;

 public:
  constexpr static const char* ID_COLUMN = "id_column";
  constexpr static const char* SRC_COLUMN = "src_column";
  constexpr static const char* DST_COLUMN = "dst_column";
  constexpr static const char* SRC_LABEL_ID = "src_label_id";
  constexpr static const char* DST_LABEL_ID = "dst_label_id";

  explicit BasicArrowFragmentLoader(const grape::CommSpec& comm_spec)
      : comm_spec_(comm_spec) {}

  void SetPartitioner(const PARTITIONER_T& partitioner) {
    partitioner_ = partitioner;
  }

  void Init(const std::vector<std::shared_ptr<arrow::Table>>& vertex_tables,
            const std::vector<std::vector<std::shared_ptr<arrow::Table>>>&
                edge_tables) {
    vertex_tables_ = vertex_tables;
    edge_tables_ = edge_tables;
    v_label_num_ = vertex_tables.size();
    e_label_num_ = edge_tables.size();
    oid_lists_.resize(v_label_num_);
  }

  std::vector<std::vector<std::shared_ptr<oid_array_t>>>& GetOidLists() {
    return oid_lists_;
  }

  auto ShuffleVertexTables(bool deduplicate_oid)
      -> boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> {
    std::vector<std::shared_ptr<arrow::Table>> local_v_tables(v_label_num_);

    for (label_id_t v_label = 0; v_label < v_label_num_; v_label++) {
      auto e = boost::leaf::try_handle_all(
          [&, this]() -> boost::leaf::result<GSError> {
            auto& vertex_table = vertex_tables_[v_label];
            auto metadata = vertex_table->schema()->metadata()->Copy();
            auto meta_idx = metadata->FindKey(ID_COLUMN);
            CHECK_OR_RAISE(meta_idx != -1);
            auto id_column_idx = std::stoi(metadata->value(meta_idx));

            // TODO(guanyi.gl): Failure occurred before MPI calling will make
            // processes hanging. We have to resolve this kind of issue.
            auto id_column_type = vertex_table->column(id_column_idx)->type();

            if (vineyard::ConvertToArrowType<oid_t>::TypeValue() !=
                id_column_type) {
              RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                              "OID_T is not same with arrow::Column(" +
                                  id_column_type->ToString() + ")");
            }
#if 0
            BOOST_LEAF_AUTO(tmp_table,
                            ShufflePropertyVertexTable<partitioner_t>(
                                comm_spec_, partitioner_, vertex_table));
#else
            BOOST_LEAF_AUTO(tmp_table,
                            beta::ShufflePropertyVertexTable<partitioner_t>(
                                comm_spec_, partitioner_, vertex_table));
#endif
            /**
             * Keep the oid column in vertex data table for HTAP, rather, we
             * record the id column name (primary key) in schema's metadata.
             *
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
            ARROW_OK_OR_RAISE(tmp_table->RemoveColumn(
                id_column_idx, &local_v_tables[v_label]));
#else
            ARROW_OK_ASSIGN_OR_RAISE(local_v_tables[v_label],
                                     tmp_table->RemoveColumn(id_column_idx));
#endif
            */

            /**
             * Move the id_column to the last column first, to avoid effecting
             * the original analytical apps (the Project API).
             *
             * Note that this operation happens on table after shuffle.
             */
            auto id_field = tmp_table->schema()->field(id_column_idx);
            auto id_column = tmp_table->column(id_column_idx);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
            CHECK_ARROW_ERROR(
                tmp_table->RemoveColumn(id_column_idx, &tmp_table));
            CHECK_ARROW_ERROR(tmp_table->AddColumn(
                tmp_table->num_columns(), id_field, id_column, &tmp_table));
#else
            CHECK_ARROW_ERROR_AND_ASSIGN(
                tmp_table, tmp_table->RemoveColumn(id_column_idx));
            CHECK_ARROW_ERROR_AND_ASSIGN(
                tmp_table, tmp_table->AddColumn(tmp_table->num_columns(),
                                                id_field, id_column));
#endif
            id_column_idx = tmp_table->num_columns() - 1;

            local_v_tables[v_label] = tmp_table;
            metadata->Append("primary_key", local_v_tables[v_label]
                                                ->schema()
                                                ->field(id_column_idx)
                                                ->name());
            local_v_tables[v_label] =
                local_v_tables[v_label]->ReplaceSchemaMetadata(metadata);

            CHECK_OR_RAISE(tmp_table->field(id_column_idx)->type() ==
                           vineyard::ConvertToArrowType<oid_t>::TypeValue());
            CHECK_OR_RAISE(tmp_table->column(id_column_idx)->num_chunks() <= 1);
            auto local_oid_array = std::dynamic_pointer_cast<oid_array_t>(
                tmp_table->column(id_column_idx)->chunk(0));
            BOOST_LEAF_AUTO(
                oids_group_by_worker,
                FragmentAllGatherArray<oid_t>(comm_spec_, local_oid_array));
            // Deduplicate oids. this procedure is necessary when the oids are
            // inferred from efile
            if (deduplicate_oid) {
              for (size_t i = 0; i < oids_group_by_worker.size(); i++) {
                OidSet<oid_t> oid_set;
                BOOST_LEAF_CHECK(oid_set.BatchInsert(oids_group_by_worker[i]));
                BOOST_LEAF_AUTO(deduplicated_oid_array, oid_set.ToArrowArray());
                oids_group_by_worker[i] = deduplicated_oid_array;
              }
            }
            oid_lists_[v_label] = oids_group_by_worker;

            return AllGatherError(comm_spec_);
          },
          [this](GSError& e) { return AllGatherError(e, comm_spec_); },
          [this](const boost::leaf::error_info& unmatched) {
            GSError e(ErrorCode::kIOError, "Unmatched error");
            return AllGatherError(e, comm_spec_);
          });
      if (e.error_code != ErrorCode::kOk) {
        return boost::leaf::new_error(e);
      }
    }

    return local_v_tables;
  }

  auto ShuffleEdgeTables(
      std::function<bool(fid_t, label_id_t, internal_oid_t, vid_t&)> mapper)
      -> boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>> {
    std::vector<std::shared_ptr<arrow::Table>> local_e_tables(e_label_num_);
    vineyard::IdParser<vid_t> id_parser;
    std::shared_ptr<arrow::Field> src_gid_field =
        std::make_shared<arrow::Field>(
            "src", vineyard::ConvertToArrowType<vid_t>::TypeValue());
    std::shared_ptr<arrow::Field> dst_gid_field =
        std::make_shared<arrow::Field>(
            "dst", vineyard::ConvertToArrowType<vid_t>::TypeValue());

    id_parser.Init(comm_spec_.fnum(), v_label_num_);

    for (label_id_t e_label = 0; e_label < e_label_num_; e_label++) {
      auto& edge_table_list = edge_tables_[e_label];
      auto e = boost::leaf::try_handle_all(
          [&, this]() -> boost::leaf::result<GSError> {
            std::vector<std::shared_ptr<arrow::Table>> processed_table_list(
                edge_table_list.size());
            int src_column_idx = -1, dst_column_idx = -1;
            for (size_t edge_table_index = 0;
                 edge_table_index != edge_table_list.size();
                 ++edge_table_index) {
              auto& edge_table = edge_table_list[edge_table_index];
              auto metadata = edge_table->schema()->metadata();
              auto meta_idx_src = metadata->FindKey(SRC_COLUMN);
              auto meta_idx_dst = metadata->FindKey(DST_COLUMN);
              CHECK_OR_RAISE(meta_idx_src != -1);
              CHECK_OR_RAISE(meta_idx_dst != -1);
              auto cur_src_column_idx =
                  std::stoi(metadata->value(meta_idx_src));
              auto cur_dst_column_idx =
                  std::stoi(metadata->value(meta_idx_dst));
              if (src_column_idx == -1) {
                src_column_idx = cur_src_column_idx;
              } else {
                if (src_column_idx != cur_src_column_idx) {
                  RETURN_GS_ERROR(ErrorCode::kIOError,
                                  "Edge tables' schema not consistent");
                }
              }
              if (dst_column_idx == -1) {
                dst_column_idx = cur_dst_column_idx;
              } else {
                if (dst_column_idx != cur_dst_column_idx) {
                  RETURN_GS_ERROR(ErrorCode::kIOError,
                                  "Edge tables' schema not consistent");
                }
              }

              auto src_column_type = edge_table->column(src_column_idx)->type();
              auto dst_column_type = edge_table->column(dst_column_idx)->type();

              if (!src_column_type->Equals(dst_column_type) ||
                  !src_column_type->Equals(
                      vineyard::ConvertToArrowType<oid_t>::TypeValue())) {
                RETURN_GS_ERROR(ErrorCode::kInvalidValueError,
                                "OID_T is not same with arrow::Column(" +
                                    src_column_type->ToString() + ")");
              }

              auto meta_idx_src_label_index = metadata->FindKey(SRC_LABEL_ID);
              auto meta_idx_dst_label_index = metadata->FindKey(DST_LABEL_ID);
              CHECK_OR_RAISE(meta_idx_src_label_index != -1);
              CHECK_OR_RAISE(meta_idx_dst_label_index != -1);

              auto src_label_id = static_cast<label_id_t>(
                  std::stoi(metadata->value(meta_idx_src_label_index)));
              auto dst_label_id = static_cast<label_id_t>(
                  std::stoi(metadata->value(meta_idx_dst_label_index)));

              BOOST_LEAF_AUTO(src_gid_array,
                              parseOidChunkedArray(
                                  src_label_id,
                                  edge_table->column(src_column_idx), mapper));
              BOOST_LEAF_AUTO(dst_gid_array,
                              parseOidChunkedArray(
                                  dst_label_id,
                                  edge_table->column(dst_column_idx), mapper));

          // replace oid columns with gid
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
              ARROW_OK_OR_RAISE(edge_table->SetColumn(
                  src_column_idx, src_gid_field, src_gid_array, &edge_table));
              ARROW_OK_OR_RAISE(edge_table->SetColumn(
                  dst_column_idx, dst_gid_field, dst_gid_array, &edge_table));
#else
              ARROW_OK_ASSIGN_OR_RAISE(
                  edge_table,
                  edge_table->SetColumn(src_column_idx, src_gid_field,
                                        src_gid_array));
              ARROW_OK_ASSIGN_OR_RAISE(
                  edge_table,
                  edge_table->SetColumn(dst_column_idx, dst_gid_field,
                                        dst_gid_array));
#endif

              processed_table_list[edge_table_index] = edge_table;
            }
            auto table = ConcatenateTables(processed_table_list);
#if 0
            auto r = ShufflePropertyEdgeTable<vid_t>(
                comm_spec_, id_parser, src_column_idx, dst_column_idx, table);
#else
            auto r = beta::ShufflePropertyEdgeTable<vid_t>(
                comm_spec_, id_parser, src_column_idx, dst_column_idx, table);
#endif
            BOOST_LEAF_CHECK(r);
            local_e_tables[e_label] = r.value();
            return AllGatherError(comm_spec_);
          },
          [this](GSError& e) { return AllGatherError(e, comm_spec_); },
          [this](const boost::leaf::error_info& unmatched) {
            GSError e(ErrorCode::kIOError, "Unmatched error");
            return AllGatherError(e, comm_spec_);
          });
      if (e.error_code != ErrorCode::kOk) {
        return boost::leaf::new_error(e);
      }
    }

    return local_e_tables;
  }

  std::shared_ptr<arrow::Table> ConcatenateTables(
      std::vector<std::shared_ptr<arrow::Table>>& tables) {
    if (tables.size() == 1) {
      return tables[0];
    }
    auto col_names = tables[0]->ColumnNames();
    for (size_t i = 1; i < tables.size(); ++i) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
      CHECK_ARROW_ERROR(tables[i]->RenameColumns(col_names, &tables[i]));
#else
      CHECK_ARROW_ERROR_AND_ASSIGN(tables[i],
                                   tables[i]->RenameColumns(col_names));
#endif
    }
    std::shared_ptr<arrow::Table> table;
    CHECK_ARROW_ERROR_AND_ASSIGN(table, arrow::ConcatenateTables(tables));
    return table;
  }

 private:
  auto parseOidChunkedArray(
      label_id_t label_id,
      const std::shared_ptr<arrow::ChunkedArray>& oid_arrays_in,
      std::function<bool(fid_t, label_id_t, internal_oid_t, vid_t&)>&
          oid2gid_mapper)
      -> boost::leaf::result<std::shared_ptr<arrow::ChunkedArray>> {
    size_t chunk_num = oid_arrays_in->num_chunks();
    std::vector<std::shared_ptr<arrow::Array>> chunks_out(chunk_num);

#if 0
    for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
      std::shared_ptr<oid_array_t> oid_array =
          std::dynamic_pointer_cast<oid_array_t>(oid_arrays_in->chunk(chunk_i));
      typename vineyard::ConvertToArrowType<vid_t>::BuilderType builder;
      size_t size = oid_array->length();
      ARROW_OK_OR_RAISE(builder.Resize(size));

      for (size_t i = 0; i != size; ++i) {
        internal_oid_t oid = oid_array->GetView(i);
        fid_t fid = partitioner_.GetPartitionId(oid_t(oid));
        CHECK_OR_RAISE(oid2gid_mapper(fid, label_id, oid, builder[i]));
      }
      ARROW_OK_OR_RAISE(builder.Advance(size));
      ARROW_OK_OR_RAISE(builder.Finish(&chunks_out[chunk_i]));
    }
#else
    int thread_num =
        (std::thread::hardware_concurrency() + comm_spec_.local_num() - 1) /
        comm_spec_.local_num();
    std::vector<std::thread> parse_threads(thread_num);

    std::atomic<size_t> cur(0);
    std::vector<arrow::Status> statuses(thread_num, arrow::Status::OK());
    for (int i = 0; i < thread_num; ++i) {
      parse_threads[i] = std::thread(
          [&](int tid) {
            while (true) {
              auto got = cur.fetch_add(1);
              if (got >= chunk_num) {
                break;
              }
              std::shared_ptr<oid_array_t> oid_array =
                  std::dynamic_pointer_cast<oid_array_t>(
                      oid_arrays_in->chunk(got));
              typename ConvertToArrowType<vid_t>::BuilderType builder;
              size_t size = oid_array->length();

              arrow::Status status = builder.Resize(size);
              if (!status.ok()) {
                statuses[tid] = status;
                return;
              }

              for (size_t k = 0; k != size; ++k) {
                internal_oid_t oid = oid_array->GetView(k);
                fid_t fid = partitioner_.GetPartitionId(oid_t(oid));
                if (!oid2gid_mapper(fid, label_id, oid, builder[k])) {
                  LOG(ERROR) << "Mapping vertex " << oid << " failed.";
                }
              }

              status = builder.Advance(size);
              if (!status.ok()) {
                statuses[tid] = status;
                return;
              }
              status = builder.Finish(&chunks_out[got]);
              if (!status.ok()) {
                statuses[tid] = status;
                return;
              }
            }
          },
          i);
    }
    for (auto& thrd : parse_threads) {
      thrd.join();
    }
    for (auto& status : statuses) {
      if (!status.ok()) {
        RETURN_GS_ERROR(ErrorCode::kArrowError, status.ToString());
      }
    }
#endif

    return std::make_shared<arrow::ChunkedArray>(chunks_out);
  }

  grape::CommSpec comm_spec_;
  label_id_t v_label_num_;
  label_id_t e_label_num_;
  std::vector<std::shared_ptr<arrow::Table>> vertex_tables_;
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> edge_tables_;

  std::vector<std::vector<std::shared_ptr<oid_array_t>>>
      oid_lists_;  // v_label/fid/oid_array

  partitioner_t partitioner_;
};
}  // namespace vineyard
#endif  // MODULES_GRAPH_LOADER_BASIC_ARROW_FRAGMENT_LOADER_H_
