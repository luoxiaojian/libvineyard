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

#ifndef MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
#define MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "client/client.h"
#include "grape/worker/comm_spec.h"
#include "io/io/io_factory.h"
#include "io/io/local_io_adaptor.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/fragment/arrow_fragment_group.h"
#include "graph/fragment/graph_schema.h"
#include "graph/fragment/property_graph_types.h"
#include "graph/loader/basic_arrow_fragment_loader.h"
#include "graph/utils/error.h"
#include "graph/utils/partitioner.h"
#include "graph/vertex_map/arrow_vertex_map.h"

#define HASH_PARTITION

namespace vineyard {

template <typename OID_T = property_graph_types::OID_TYPE,
          typename VID_T = property_graph_types::VID_TYPE>
class ArrowFragmentLoader {
  using oid_t = OID_T;
  using vid_t = VID_T;
  using label_id_t = property_graph_types::LABEL_ID_TYPE;
  using internal_oid_t = typename InternalType<oid_t>::type;
  using oid_array_t = typename vineyard::ConvertToArrowType<oid_t>::ArrayType;
  using vertex_map_t = ArrowVertexMap<internal_oid_t, vid_t>;
  const int id_column = 0;
  const int src_column = 0;
  const int dst_column = 1;
#ifdef HASH_PARTITION
  using partitioner_t = HashPartitioner<oid_t>;
#else
  using partitioner_t = SegmentedPartitioner<oid_t>;
#endif
  using basic_loader_t = BasicArrowFragmentLoader<oid_t, vid_t, partitioner_t>;

 public:
  ArrowFragmentLoader(vineyard::Client& client,
                      const grape::CommSpec& comm_spec,
                      const std::vector<std::string>& efiles,
                      const std::vector<std::string>& vfiles,
                      bool directed = true)
      : client_(client),
        comm_spec_(comm_spec),
        efiles_(efiles),
        vfiles_(vfiles),
        vertex_label_num_(vfiles.size()),
        edge_label_num_(efiles.size()),
        directed_(directed),
        basic_arrow_fragment_loader_(comm_spec) {}

  ArrowFragmentLoader(
      vineyard::Client& client, const grape::CommSpec& comm_spec,
      label_id_t vertex_label_num, label_id_t edge_label_num,
      std::vector<std::shared_ptr<arrow::Table>> const& partial_v_tables,
      std::vector<std::shared_ptr<arrow::Table>> const& partial_e_tables,
      bool directed = true)
      : client_(client),
        comm_spec_(comm_spec),
        vertex_label_num_(vertex_label_num),
        edge_label_num_(edge_label_num),
        partial_v_tables_(partial_v_tables),
        partial_e_tables_(partial_e_tables),
        directed_(directed),
        basic_arrow_fragment_loader_(comm_spec) {}

  ~ArrowFragmentLoader() = default;

  boost::leaf::result<vineyard::ObjectID> LoadFragment() {
    BOOST_LEAF_CHECK(initPartitioner());
    BOOST_LEAF_CHECK(initBasicLoader());
    BOOST_LEAF_AUTO(frag_id, shuffleAndBuild());
    return frag_id;
  }

  boost::leaf::result<vineyard::ObjectID> LoadFragmentAsFragmentGroup() {
    BOOST_LEAF_AUTO(frag_id, LoadFragment());
    BOOST_LEAF_AUTO(group_id,
                    constructFragmentGroup(client_, frag_id, comm_spec_,
                                           vertex_label_num_, edge_label_num_));
    return group_id;
  }

 protected:
  boost::leaf::result<void> initPartitioner() {
#ifdef HASH_PARTITION
    partitioner_.Init(comm_spec_.fnum());
#else
    std::vector<std::shared_ptr<arrow::Table>> vtables;
    {
      BOOST_LEAF_AUTO(tmp, loadVertexTables(vfiles_, 0, 1));
      vtables = tmp;
    }
    std::vector<oid_t> oid_list;

    for (auto& table : vtables) {
      std::shared_ptr<arrow::ChunkedArray> oid_array_chunks =
          table->column(id_column);
      size_t chunk_num = oid_array_chunks->num_chunks();

      for (size_t chunk_i = 0; chunk_i != chunk_num; ++chunk_i) {
        std::shared_ptr<oid_array_t> array =
            std::dynamic_pointer_cast<oid_array_t>(
                oid_array_chunks->chunk(chunk_i));
        int64_t length = array->length();
        for (int64_t i = 0; i < length; ++i) {
          oid_list.emplace_back(oid_t(array->GetView(i)));
        }
      }
    }

    partitioner_.Init(comm_spec_.fnum(), oid_list);
#endif
    return boost::leaf::result<void>();
  }

  boost::leaf::result<void> initBasicLoader() {
    std::vector<std::shared_ptr<arrow::Table>> partial_v_tables;
    std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables;
    if (!partial_v_tables_.empty() && !partial_e_tables_.empty()) {
      partial_v_tables = partial_v_tables_;
      partial_e_tables = partial_e_tables_;
    } else {
      BOOST_LEAF_AUTO(tmp_v, loadVertexTables(vfiles_, comm_spec_.worker_id(),
                                              comm_spec_.worker_num()));
      BOOST_LEAF_AUTO(tmp_e, loadEdgeTables(efiles_, comm_spec_.worker_id(),
                                            comm_spec_.worker_num()));
      partial_v_tables = tmp_v;
      partial_e_tables = tmp_e;
    }
    basic_arrow_fragment_loader_.Init(partial_v_tables, partial_e_tables);
    basic_arrow_fragment_loader_.SetPartitioner(partitioner_);

    return boost::leaf::result<void>();
  }

  boost::leaf::result<vineyard::ObjectID> shuffleAndBuild() {
    BOOST_LEAF_AUTO(local_v_tables,
                    basic_arrow_fragment_loader_.ShuffleVertexTables());
    auto oid_lists = basic_arrow_fragment_loader_.GetOidLists();

    BasicArrowVertexMapBuilder<typename InternalType<oid_t>::type, vid_t>
        vm_builder(client_, comm_spec_.fnum(), vertex_label_num_, oid_lists);
    auto vm = vm_builder.Seal(client_);
    auto vm_ptr =
        std::dynamic_pointer_cast<vertex_map_t>(client_.GetObject(vm->id()));
    auto mapper = [&vm_ptr](fid_t fid, label_id_t label, internal_oid_t oid,
                            vid_t& gid) {
      CHECK(vm_ptr->GetGid(fid, label, oid, gid));
      return true;
    };
    BOOST_LEAF_AUTO(local_e_tables,
                    basic_arrow_fragment_loader_.ShuffleEdgeTables(mapper));
    BasicArrowFragmentBuilder<oid_t, vid_t> frag_builder(client_, vm_ptr);

    {
      // Make sure the sequence of tables in local_v_tables and local_e_tables
      // are correspond to their label_index.
      std::vector<std::shared_ptr<arrow::Table>> rearranged_v_tables;
      rearranged_v_tables.resize(local_v_tables.size());
      for (auto table : local_v_tables) {
        auto meta = table->schema()->metadata();
        auto label_index_field = meta->FindKey("label_index");
        CHECK_NE(label_index_field, -1);
        auto label_index = std::stoi(meta->value(label_index_field));
        CHECK_LT(label_index, rearranged_v_tables.size());
        rearranged_v_tables[label_index] = table;
      }
      local_v_tables = rearranged_v_tables;

      std::vector<std::shared_ptr<arrow::Table>> rearranged_e_tables;
      rearranged_e_tables.resize(local_e_tables.size());
      for (auto table : local_e_tables) {
        auto meta = table->schema()->metadata();
        auto label_index_field = meta->FindKey("label_index");
        CHECK_NE(label_index_field, -1);
        auto label_index = std::stoi(meta->value(label_index_field));
        CHECK_LT(label_index, rearranged_e_tables.size());
        rearranged_e_tables[label_index] = table;
      }
      local_e_tables = rearranged_e_tables;
    }

    PropertyGraphSchema schema;
    schema.set_fnum(comm_spec_.fnum());

    for (auto table : local_v_tables) {
      std::unordered_map<std::string, std::string> kvs;
      table->schema()->metadata()->ToUnorderedMap(&kvs);
      std::string type = kvs["type"];
      std::string label = kvs["label"];

      auto entry = schema.CreateEntry(label, type);
      // entry->add_primary_keys(1, table->schema()->field_names());

      // N.B. ID column is already been removed.
      for (int64_t i = 0; i < table->num_columns(); ++i) {
        entry->AddProperty(table->schema()->field(i)->name(),
                           table->schema()->field(i)->type());
      }
    }
    for (auto table : local_e_tables) {
      std::unordered_map<std::string, std::string> kvs;
      table->schema()->metadata()->ToUnorderedMap(&kvs);
      std::string type = kvs["type"];
      std::string label = kvs["label"];
      auto entry = schema.CreateEntry(label, type);

      std::string sub_label = kvs["sub_label_num"];
      if (!sub_label.empty()) {
        int sub_label_num = std::stoi(sub_label);
        for (int i = 0; i < sub_label_num; ++i) {
          std::string src_label = kvs["src_label_" + std::to_string(i)];
          std::string dst_label = kvs["dst_label_" + std::to_string(i)];

          if (!src_label.empty() && !dst_label.empty()) {
            entry->AddRelation(src_label, dst_label);
          }
        }
      }
      // N.B. Skip first two ID columns.
      for (int64_t i = 2; i < table->num_columns(); ++i) {
        entry->AddProperty(table->schema()->field(i)->name(),
                           table->schema()->field(i)->type());
      }
    }
    frag_builder.SetPropertyGraphSchema(std::move(schema));

    BOOST_LEAF_CHECK(frag_builder.Init(comm_spec_.fid(), comm_spec_.fnum(),
                                       std::move(local_v_tables),
                                       std::move(local_e_tables), directed_));
    auto frag = std::dynamic_pointer_cast<ArrowFragment<oid_t, vid_t>>(
        frag_builder.Seal(client_));
    VINEYARD_CHECK_OK(client_.Persist(frag->id()));
    return frag->id();
  }

  boost::leaf::result<vineyard::ObjectID> constructFragmentGroup(
      vineyard::Client& client, vineyard::ObjectID frag_id,
      grape::CommSpec comm_spec, label_id_t v_label_num,
      label_id_t e_label_num) {
    vineyard::ObjectID group_object_id;
    uint64_t instance_id = client.instance_id();

    if (comm_spec.worker_id() == 0) {
      std::vector<uint64_t> gathered_instance_ids(comm_spec.worker_num());
      std::vector<vineyard::ObjectID> gathered_object_ids(
          comm_spec.worker_num());

      MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR,
                 &gathered_instance_ids[0], sizeof(uint64_t), MPI_CHAR, 0,
                 comm_spec.comm());

      MPI_Gather(&frag_id, sizeof(vineyard::ObjectID), MPI_CHAR,
                 &gathered_object_ids[0], sizeof(vineyard::ObjectID), MPI_CHAR,
                 0, comm_spec.comm());

      ArrowFragmentGroupBuilder builder;
      builder.set_total_frag_num(comm_spec.fnum());
      builder.set_vertex_label_num(v_label_num);
      builder.set_edge_label_num(e_label_num);
      for (fid_t i = 0; i < comm_spec.fnum(); ++i) {
        builder.AddFragmentObject(
            i, gathered_object_ids[comm_spec.FragToWorker(i)],
            gathered_instance_ids[comm_spec.FragToWorker(i)]);
      }

      auto group_object =
          std::dynamic_pointer_cast<ArrowFragmentGroup>(builder.Seal(client));
      group_object_id = group_object->id();
      VY_OK_OR_RAISE(client.Persist(group_object_id));

      MPI_Bcast(&group_object_id, sizeof(vineyard::ObjectID), MPI_CHAR, 0,
                comm_spec.comm());

    } else {
      MPI_Gather(&instance_id, sizeof(uint64_t), MPI_CHAR, NULL,
                 sizeof(uint64_t), MPI_CHAR, 0, comm_spec.comm());
      MPI_Gather(&frag_id, sizeof(vineyard::ObjectID), MPI_CHAR, NULL,
                 sizeof(vineyard::ObjectID), MPI_CHAR, 0, comm_spec.comm());

      MPI_Bcast(&group_object_id, sizeof(vineyard::ObjectID), MPI_CHAR, 0,
                comm_spec.comm());
    }
    return group_object_id;
  }

  boost::leaf::result<std::vector<std::shared_ptr<arrow::Table>>>
  loadVertexTables(const std::vector<std::string>& files, int index,
                   int total_parts) {
    label_id_t label_num = static_cast<label_id_t>(files.size());
    std::vector<std::shared_ptr<arrow::Table>> tables(label_num);

    auto io_deleter = [](vineyard::LocalIOAdaptor* adaptor) {
      VINEYARD_CHECK_OK(adaptor->Close());
      delete adaptor;
    };
    try {
      for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
        std::unique_ptr<vineyard::LocalIOAdaptor,
                        std::function<void(vineyard::LocalIOAdaptor*)>>
            io_adaptor(new vineyard::LocalIOAdaptor(files[label_id] +
                                                    "#header_row=true"),
                       io_deleter);
        VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
        VY_OK_OR_RAISE(io_adaptor->Open());
        std::shared_ptr<arrow::Table> table;
        VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
        BOOST_LEAF_CHECK(SyncSchema(table, comm_spec_));
        auto meta = std::make_shared<arrow::KeyValueMetadata>();
        meta->Append("type", "VERTEX");
        meta->Append("label_index", std::to_string(label_id));

        meta->Append(basic_loader_t::ID_COLUMN, "0");
        auto adaptor_meta = io_adaptor->GetMeta();
        for (auto const& kv : adaptor_meta) {
          meta->Append(kv.first, kv.second);
        }
        // If label name is not in meta, we assign a default label '_'
        if (adaptor_meta.count("label") == 0) {
          return boost::leaf::new_error(
              ErrorCode::kIOError,
              "Metadata of input vertex files should contain label name");
        }
        tables[label_id] = table->ReplaceSchemaMetadata(meta);

        vertex_label_to_index_[adaptor_meta.find("label")->second] = label_id;
      }
    } catch (std::exception& e) {
      return boost::leaf::new_error(ErrorCode::kIOError, std::string(e.what()));
    }
    return tables;
  }

  boost::leaf::result<std::vector<std::vector<std::shared_ptr<arrow::Table>>>>
  loadEdgeTables(const std::vector<std::string>& files, int index,
                 int total_parts) {
    label_id_t label_num = static_cast<label_id_t>(files.size());

    std::vector<std::vector<std::shared_ptr<arrow::Table>>> tables(label_num);

    auto io_deleter = [](vineyard::LocalIOAdaptor* adaptor) {
      VINEYARD_CHECK_OK(adaptor->Close());
      delete adaptor;
    };

    try {
      for (label_id_t label_id = 0; label_id < label_num; ++label_id) {
        std::vector<std::string> sub_label_files;
        boost::split(sub_label_files, files[label_id], boost::is_any_of(";"));

        for (size_t j = 0; j < sub_label_files.size(); ++j) {
          std::unique_ptr<vineyard::LocalIOAdaptor,
                          std::function<void(vineyard::LocalIOAdaptor*)>>
              io_adaptor(new vineyard::LocalIOAdaptor(sub_label_files[j] +
                                                      "#header_row=true"),
                         io_deleter);
          VY_OK_OR_RAISE(io_adaptor->SetPartialRead(index, total_parts));
          VY_OK_OR_RAISE(io_adaptor->Open());
          std::shared_ptr<arrow::Table> table;
          VY_OK_OR_RAISE(io_adaptor->ReadTable(&table));
          BOOST_LEAF_CHECK(SyncSchema(table, comm_spec_));

          std::shared_ptr<arrow::KeyValueMetadata> meta(
              new arrow::KeyValueMetadata());
          meta->Append("type", "EDGE");
          meta->Append("label_index", std::to_string(label_id));
          meta->Append(basic_loader_t::SRC_COLUMN, std::to_string(src_column));
          meta->Append(basic_loader_t::DST_COLUMN, std::to_string(dst_column));
          meta->Append("sub_label_num", std::to_string(sub_label_files.size()));

          auto adaptor_meta = io_adaptor->GetMeta();
          auto search = adaptor_meta.find("label");
          if (search == adaptor_meta.end()) {
            return boost::leaf::new_error(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain label name");
          }
          meta->Append("label", search->second);

          search = adaptor_meta.find("src_label");
          if (search == adaptor_meta.end()) {
            return boost::leaf::new_error(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain src label name");
          }
          meta->Append(
              basic_loader_t::SRC_LABEL_INDEX,
              std::to_string(vertex_label_to_index_.at(search->second)));

          search = adaptor_meta.find("dst_label");
          if (search == adaptor_meta.end()) {
            return boost::leaf::new_error(
                ErrorCode::kIOError,
                "Metadata of input edge files should contain dst label name");
          }
          meta->Append(
              basic_loader_t::DST_LABEL_INDEX,
              std::to_string(vertex_label_to_index_.at(search->second)));

          tables[label_id].emplace_back(table->ReplaceSchemaMetadata(meta));
        }
      }
    } catch (std::exception& e) {
      return boost::leaf::new_error(ErrorCode::kIOError, std::string(e.what()));
    }
    return tables;
  }

  arrow::Status swapColumn(std::shared_ptr<arrow::Table> in, int lhs_index,
                           int rhs_index, std::shared_ptr<arrow::Table>* out) {
    if (lhs_index == rhs_index) {
      out = &in;
      return arrow::Status::OK();
    }
    if (lhs_index > rhs_index) {
      return arrow::Status::Invalid("lhs index must smaller than rhs index.");
    }
    auto field = in->schema()->field(rhs_index);
    auto column = in->column(rhs_index);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(in->RemoveColumn(rhs_index, &in));
    CHECK_ARROW_ERROR(in->AddColumn(lhs_index, field, column, out));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(in, in->RemoveColumn(rhs_index));
    CHECK_ARROW_ERROR_AND_ASSIGN(*out, in->AddColumn(lhs_index, field, column));
#endif
    return arrow::Status::OK();
  }

  void SerializeSchema(const std::shared_ptr<arrow::Schema>& schema,
                       std::shared_ptr<arrow::Buffer>* out) {
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(arrow::ipc::SerializeSchema(
        *schema, nullptr, arrow::default_memory_pool(), out));
#elif defined(ARROW_VERSION) && ARROW_VERSION < 2000000
    CHECK_ARROW_ERROR_AND_ASSIGN(
        *out, arrow::ipc::SerializeSchema(*schema, nullptr,
                                          arrow::default_memory_pool()));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(
        *out,
        arrow::ipc::SerializeSchema(*schema, arrow::default_memory_pool()));
#endif
  }

  void DeserializeSchema(const std::shared_ptr<arrow::Buffer>& buffer,
                         std::shared_ptr<arrow::Schema>* out) {
    arrow::io::BufferReader reader(buffer);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(arrow::ipc::ReadSchema(&reader, nullptr, out));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(*out,
                                 arrow::ipc::ReadSchema(&reader, nullptr));
#endif
  }

  std::shared_ptr<arrow::Schema> TypeLoosen(
      const std::vector<std::shared_ptr<arrow::Schema>>& schemas) {
    size_t field_num = 0;
    for (size_t i = 0; i < schemas.size(); ++i) {
      if (schemas[i] != nullptr) {
        field_num = schemas[i]->num_fields();
        break;
      }
    }
    // Perform type lossen.
    // timestamp -> int64 -> double -> utf8   binary (not supported)
    std::vector<std::vector<std::shared_ptr<arrow::Field>>> fields(field_num);
    for (size_t i = 0; i < field_num; ++i) {
      for (size_t j = 0; j < schemas.size(); ++j) {
        if (schemas[j] == nullptr) {
          continue;
        }
        fields[i].push_back(schemas[j]->field(i));
      }
    }
    std::vector<std::shared_ptr<arrow::Field>> lossen_fields(field_num);

    for (size_t i = 0; i < field_num; ++i) {
      // find the max frequency using linear traversal
      auto res = fields[i][0]->type();
      if (res->Equals(arrow::timestamp(arrow::TimeUnit::SECOND))) {
        res = arrow::int64();
      }
      if (res->Equals(arrow::int64())) {
        for (size_t j = 1; j < fields[i].size(); ++j) {
          if (fields[i][j]->type()->Equals(arrow::float64())) {
            res = arrow::float64();
          }
        }
      }

      if (res->Equals(arrow::float64())) {
        for (size_t j = 1; j < fields[i].size(); ++j) {
          if (fields[i][j]->type()->Equals(arrow::utf8())) {
            res = arrow::utf8();
          }
        }
      }
      lossen_fields[i] = fields[i][0]->WithType(res);
    }
    auto final_schema = std::make_shared<arrow::Schema>(lossen_fields);
    return final_schema;
  }

  // This method used when several workers is loading a file in parallel, each
  // worker will read a chunk of the origin file into a arrow::Table.
  // We may get different table schemas as some chunks may have zero rows
  // or some chunks' data doesn't have any floating numbers, but others might
  // have. We could use this method to gather their schemas, and find out most
  // common fields, construct a new schema and broadcast back. Note: We perform
  // type loosen, int64 -> double. timestamp -> int64.
  boost::leaf::result<void> SyncSchema(std::shared_ptr<arrow::Table>& table,
                                       grape::CommSpec comm_spec) {
    std::shared_ptr<arrow::Schema> final_schema;
    int final_serialized_schema_size;
    std::shared_ptr<arrow::Buffer> schema_buffer;
    int size = 0;
    if (table != nullptr) {
      SerializeSchema(table->schema(), &schema_buffer);
      size = static_cast<int>(schema_buffer->size());
    }
    if (comm_spec.worker_id() == 0) {
      std::vector<int> recvcounts(comm_spec.worker_num());

      MPI_Gather(&size, sizeof(int), MPI_CHAR, &recvcounts[0], sizeof(int),
                 MPI_CHAR, 0, comm_spec.comm());
      std::vector<int> displs(comm_spec.worker_num());
      int total_len = 0;
      displs[0] = 0;
      total_len += recvcounts[0];

      for (size_t i = 1; i < recvcounts.size(); i++) {
        total_len += recvcounts[i];
        displs[i] = displs[i - 1] + recvcounts[i - 1];
      }
      if (total_len == 0) {
        RETURN_GS_ERROR(ErrorCode::kIOError, "All schema is empty");
      }
      char* total_string = static_cast<char*>(malloc(total_len * sizeof(char)));
      if (size == 0) {
        MPI_Gatherv(NULL, 0, MPI_CHAR, total_string, &recvcounts[0], &displs[0],
                    MPI_CHAR, 0, comm_spec.comm());

      } else {
        MPI_Gatherv(schema_buffer->data(), schema_buffer->size(), MPI_CHAR,
                    total_string, &recvcounts[0], &displs[0], MPI_CHAR, 0,
                    comm_spec.comm());
      }
      std::vector<std::shared_ptr<arrow::Buffer>> buffers(
          comm_spec.worker_num());
      for (size_t i = 0; i < buffers.size(); ++i) {
        buffers[i] = std::make_shared<arrow::Buffer>(
            reinterpret_cast<unsigned char*>(total_string + displs[i]),
            recvcounts[i]);
      }
      std::vector<std::shared_ptr<arrow::Schema>> schemas(
          comm_spec.worker_num());
      for (size_t i = 0; i < schemas.size(); ++i) {
        if (recvcounts[i] == 0) {
          continue;
        }
        DeserializeSchema(buffers[i], &schemas[i]);
      }

      final_schema = TypeLoosen(schemas);

      SerializeSchema(final_schema, &schema_buffer);
      final_serialized_schema_size = static_cast<int>(schema_buffer->size());

      MPI_Bcast(&final_serialized_schema_size, sizeof(int), MPI_CHAR, 0,
                comm_spec.comm());
      MPI_Bcast(const_cast<char*>(
                    reinterpret_cast<const char*>(schema_buffer->data())),
                final_serialized_schema_size, MPI_CHAR, 0, comm_spec.comm());
      free(total_string);
    } else {
      MPI_Gather(&size, sizeof(int), MPI_CHAR, 0, sizeof(int), MPI_CHAR, 0,
                 comm_spec.comm());
      if (size == 0) {
        MPI_Gatherv(NULL, 0, MPI_CHAR, NULL, NULL, NULL, MPI_CHAR, 0,
                    comm_spec.comm());
      } else {
        MPI_Gatherv(schema_buffer->data(), size, MPI_CHAR, NULL, NULL, NULL,
                    MPI_CHAR, 0, comm_spec.comm());
      }

      MPI_Bcast(&final_serialized_schema_size, sizeof(int), MPI_CHAR, 0,
                comm_spec.comm());
      char* recv_buf = static_cast<char*>(
          malloc(final_serialized_schema_size * sizeof(char)));
      MPI_Bcast(recv_buf, final_serialized_schema_size, MPI_CHAR, 0,
                comm_spec.comm());
      auto buffer = std::make_shared<arrow::Buffer>(
          reinterpret_cast<unsigned char*>(recv_buf),
          final_serialized_schema_size);
      DeserializeSchema(buffer, &final_schema);
      free(recv_buf);
    }
    if (table == nullptr) {
      VY_OK_OR_RAISE(vineyard::EmptyTableBuilder::Build(final_schema, table));
    } else {
      BOOST_LEAF_AUTO(tmp_table, CastTableToSchema(table, final_schema));
      table = tmp_table;
    }
    return boost::leaf::result<void>();
  }

  // Inspired by arrow::compute::Cast
  boost::leaf::result<void> CastIntToDouble(
      const std::shared_ptr<arrow::Array> in,
      std::shared_ptr<arrow::DataType> to_type,
      std::shared_ptr<arrow::Array>* out) {
    CHECK_OR_RAISE(in->type()->Equals(arrow::int64()));
    CHECK_OR_RAISE(to_type->Equals(arrow::float64()));
    using in_type = int64_t;
    using out_type = double;
    auto in_data = in->data()->GetValues<in_type>(1);
    std::vector<out_type> out_data(in->length());
    for (int64_t i = 0; i < in->length(); ++i) {
      out_data[i] = static_cast<out_type>(*in_data++);
    }
    arrow::DoubleBuilder builder;
    ARROW_OK_OR_RAISE(builder.AppendValues(out_data));
    ARROW_OK_OR_RAISE(builder.Finish(out));
    ARROW_OK_OR_RAISE((*out)->ValidateFull());
    return boost::leaf::result<void>();
  }

  // Timestamp value are stored as as number of seconds, milliseconds,
  // microseconds or nanoseconds since UNIX epoch.
  // CSV reader can only produce timestamp in seconds.
  boost::leaf::result<void> CastDateToInt(
      const std::shared_ptr<arrow::Array> in,
      std::shared_ptr<arrow::DataType> to_type,
      std::shared_ptr<arrow::Array>* out) {
    CHECK_OR_RAISE(
        in->type()->Equals(arrow::timestamp(arrow::TimeUnit::SECOND)));
    CHECK_OR_RAISE(to_type->Equals(arrow::int64()));
    auto array_data = in->data()->Copy();
    array_data->type = to_type;
    *out = arrow::MakeArray(array_data);
    ARROW_OK_OR_RAISE((*out)->ValidateFull());
    return boost::leaf::result<void>();
  }

  boost::leaf::result<std::shared_ptr<arrow::Table>> CastTableToSchema(
      const std::shared_ptr<arrow::Table>& table,
      const std::shared_ptr<arrow::Schema>& schema) {
    if (table->schema()->Equals(schema)) {
      return table;
    }
    CHECK_OR_RAISE(table->num_columns() == schema->num_fields());
    std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
    for (int64_t i = 0; i < table->num_columns(); ++i) {
      auto col = table->column(i);
      if (!table->field(i)->type()->Equals(schema->field(i)->type())) {
        auto from_type = table->field(i)->type();
        auto to_type = schema->field(i)->type();
        std::vector<std::shared_ptr<arrow::Array>> chunks;
        for (int64_t j = 0; j < col->num_chunks(); ++j) {
          auto array = col->chunk(j);
          std::shared_ptr<arrow::Array> new_array;
          if (from_type->Equals(arrow::int64()) &&
              to_type->Equals(arrow::float64())) {
            BOOST_LEAF_CHECK(CastIntToDouble(array, to_type, &new_array));
            chunks.push_back(new_array);
          } else if (from_type->Equals(
                         arrow::timestamp(arrow::TimeUnit::SECOND)) &&
                     to_type->Equals(arrow::int64())) {
            BOOST_LEAF_CHECK(CastDateToInt(array, to_type, &new_array));
            chunks.push_back(new_array);
          } else {
            RETURN_GS_ERROR(ErrorCode::kDataTypeError,
                            "Unexpected type: " + to_type->ToString() +
                                "; Origin type: " + from_type->ToString());
          }
          LOG(INFO) << "Cast " << from_type->ToString() << " To "
                    << to_type->ToString();
        }
        auto chunk_array =
            std::make_shared<arrow::ChunkedArray>(chunks, to_type);
        new_columns.push_back(chunk_array);
      } else {
        new_columns.push_back(col);
      }
    }
    return arrow::Table::Make(schema, new_columns);
  }

  std::map<std::string, label_id_t> vertex_label_to_index_;

  vineyard::Client& client_;
  grape::CommSpec comm_spec_;
  std::vector<std::string> efiles_, vfiles_;

  label_id_t vertex_label_num_, edge_label_num_;
  std::vector<std::shared_ptr<arrow::Table>> partial_v_tables_;
  std::vector<std::vector<std::shared_ptr<arrow::Table>>> partial_e_tables_;
  partitioner_t partitioner_;

  bool directed_;
  basic_loader_t basic_arrow_fragment_loader_;
};

}  // namespace vineyard

#endif  // MODULES_GRAPH_LOADER_ARROW_FRAGMENT_LOADER_H_
