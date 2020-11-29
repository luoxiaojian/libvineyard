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

#ifndef MODULES_BASIC_DS_ARROW_UTILS_H_
#define MODULES_BASIC_DS_ARROW_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/buffer.h"
#include "arrow/table.h"
#include "arrow/table_builder.h"
#include "arrow/util/config.h"
#include "boost/leaf/all.hpp"
#include "glog/logging.h"

#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"

#include "basic/ds/types.h"
#include "common/util/status.h"

namespace vineyard {

namespace arrow_types {
using OIDT = std::string;
using VIDT = uint64_t;
using EIDT = uint64_t;
}  // namespace arrow_types

struct RefString;

#define CHECK_ARROW_ERROR(expr) \
  VINEYARD_CHECK_OK(::vineyard::Status::ArrowError(expr))

#define CHECK_ARROW_ERROR_AND_ASSIGN(lhs, expr) \
  do {                                          \
    auto status = (expr);                       \
    CHECK_ARROW_ERROR(status.status());         \
    lhs = std::move(status).ValueOrDie();       \
  } while (0)

#define RETURN_ON_ARROW_ERROR(expr)                  \
  do {                                               \
    auto status = (expr);                            \
    if (!status.ok()) {                              \
      return ::vineyard::Status::ArrowError(status); \
    }                                                \
  } while (0)

#define RETURN_ON_ARROW_ERROR_AND_ASSIGN(lhs, expr)           \
  do {                                                        \
    auto result = (expr);                                     \
    if (!result.status().ok()) {                              \
      return ::vineyard::Status::ArrowError(result.status()); \
    }                                                         \
    lhs = std::move(result).ValueOrDie();                     \
  } while (0)

template <typename T>
struct ConvertToArrowType {};

#define CONVERT_TO_ARROW_TYPE(type, array_type, builder_type, type_value)      \
  template <>                                                                  \
  struct ConvertToArrowType<type> {                                            \
    using ArrayType = array_type;                                              \
    using BuilderType = builder_type;                                          \
    static std::shared_ptr<arrow::DataType> TypeValue() { return type_value; } \
  };

CONVERT_TO_ARROW_TYPE(bool, arrow::BooleanArray, arrow::BooleanBuilder,
                      arrow::boolean())
CONVERT_TO_ARROW_TYPE(int8_t, arrow::Int8Array, arrow::Int8Builder,
                      arrow::int8())
CONVERT_TO_ARROW_TYPE(uint8_t, arrow::UInt8Array, arrow::UInt8Builder,
                      arrow::uint8())
CONVERT_TO_ARROW_TYPE(int16_t, arrow::Int16Array, arrow::Int16Builder,
                      arrow::int16())
CONVERT_TO_ARROW_TYPE(uint16_t, arrow::UInt16Array, arrow::UInt16Builder,
                      arrow::uint16())
CONVERT_TO_ARROW_TYPE(int32_t, arrow::Int32Array, arrow::Int32Builder,
                      arrow::int32())
CONVERT_TO_ARROW_TYPE(uint32_t, arrow::UInt32Array, arrow::UInt32Builder,
                      arrow::uint32())
CONVERT_TO_ARROW_TYPE(int64_t, arrow::Int64Array, arrow::Int64Builder,
                      arrow::int64())
CONVERT_TO_ARROW_TYPE(uint64_t, arrow::UInt64Array, arrow::UInt64Builder,
                      arrow::uint64())
CONVERT_TO_ARROW_TYPE(float, arrow::FloatArray, arrow::FloatBuilder,
                      arrow::float32())
CONVERT_TO_ARROW_TYPE(double, arrow::DoubleArray, arrow::DoubleBuilder,
                      arrow::float64())
CONVERT_TO_ARROW_TYPE(RefString, arrow::LargeStringArray,
                      arrow::LargeStringBuilder, arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(std::string, arrow::LargeStringArray,
                      arrow::LargeStringBuilder, arrow::large_utf8())
CONVERT_TO_ARROW_TYPE(arrow::TimestampType, arrow::TimestampArray,
                      arrow::TimestampBuilder,
                      arrow::timestamp(arrow::TimeUnit::MILLI))

std::shared_ptr<arrow::DataType> FromAnyType(AnyType type);

/**
 * @brief PodArrayBuilder is designed for constructing Arrow arrays of POD data
 * type
 *
 * @tparam T
 */
template <typename T>
class PodArrayBuilder : public arrow::FixedSizeBinaryBuilder {
 public:
  explicit PodArrayBuilder(
      arrow::MemoryPool* pool = arrow::default_memory_pool())
      : arrow::FixedSizeBinaryBuilder(arrow::fixed_size_binary(sizeof(T)),
                                      pool) {}

  T* MutablePointer(int64_t i) {
    return reinterpret_cast<T*>(
        arrow::FixedSizeBinaryBuilder::GetMutableValue(i));
  }
};

/**
 * Similar to arrow's `GetRecordBatchSize`, but considering the schema.
 *
 * Used for pre-allocate buffer for NewStreamWriter's `WriteRecordBatch`.
 */
Status GetRecordBatchStreamSize(const arrow::RecordBatch& batch, size_t* size);

Status SerializeRecordBatchesToAllocatedBuffer(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer);

Status SerializeRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeRecordBatches(
    const std::shared_ptr<arrow::Buffer>& buffer,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches);

Status RecordBatchesToTable(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::Table>* table);

Status CombineRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
    std::shared_ptr<arrow::RecordBatch>* batch);

Status TableToRecordBatches(
    std::shared_ptr<arrow::Table> table,
    std::vector<std::shared_ptr<arrow::RecordBatch>>* batches);

Status SerializeTableToAllocatedBuffer(std::shared_ptr<arrow::Table> table,
                                       std::shared_ptr<arrow::Buffer>* buffer);

Status SerializeTable(std::shared_ptr<arrow::Table> table,
                      std::shared_ptr<arrow::Buffer>* buffer);

Status DeserializeTable(std::shared_ptr<arrow::Buffer> buffer,
                        std::shared_ptr<arrow::Table>* table);

struct EmptyTableBuilder {
  static Status Build(const std::shared_ptr<arrow::Schema>& schema,
                      std::shared_ptr<arrow::Table>& table) {
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

    for (int i = 0; i < schema->num_fields(); i++) {
      std::shared_ptr<arrow::Array> dummy;
      auto type = schema->field(i)->type();

      if (type == arrow::uint64()) {
        arrow::UInt64Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::int64()) {
        arrow::Int64Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::uint32()) {
        arrow::UInt32Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::int32()) {
        arrow::Int32Builder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::float32()) {
        arrow::FloatBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::float64()) {
        arrow::DoubleBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::large_utf8()) {
        arrow::LargeStringBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else if (type == arrow::null()) {
        arrow::NullBuilder builder;
        RETURN_ON_ARROW_ERROR(builder.Finish(&dummy));
      } else {
        return Status::NotImplemented("Unsupported type: " + type->ToString());
      }
      columns.push_back(std::make_shared<arrow::ChunkedArray>(dummy));
    }
    table = arrow::Table::Make(schema, columns);
    return Status::OK();
  }
};

/**
 * @brief Concatenate multiple arrow tables into one.
 */
std::shared_ptr<arrow::Table> ConcatenateTables(
    std::vector<std::shared_ptr<arrow::Table>>& tables);

/**
 * @brief Convert type name in string to arrow type.
 *
 */
inline std::shared_ptr<arrow::DataType> type_name_to_arrow_type(
    const std::string& name) {
  if (name == "bool") {
    return vineyard::ConvertToArrowType<bool>::TypeValue();
  } else if (name == "int8_t" || name == "int8" || name == "byte") {
    return vineyard::ConvertToArrowType<int8_t>::TypeValue();
  } else if (name == "uint8_t" || name == "uint8" || name == "char") {
    return vineyard::ConvertToArrowType<uint8_t>::TypeValue();
  } else if (name == "int16_t" || name == "int16" || name == "half") {
    return vineyard::ConvertToArrowType<int16_t>::TypeValue();
  } else if (name == "uint16_t" || name == "uint16") {
    return vineyard::ConvertToArrowType<uint16_t>::TypeValue();
  } else if (name == "int32_t" || name == "int32" || name == "int") {
    return vineyard::ConvertToArrowType<int32_t>::TypeValue();
  } else if (name == "uint32_t" || name == "uint32") {
    return vineyard::ConvertToArrowType<uint32_t>::TypeValue();
  } else if (name == "int64_t" || name == "int64" || name == "long") {
    return vineyard::ConvertToArrowType<int64_t>::TypeValue();
  } else if (name == "uint64_t" || name == "uint64") {
    return vineyard::ConvertToArrowType<uint64_t>::TypeValue();
  } else if (name == "float") {
    return vineyard::ConvertToArrowType<float>::TypeValue();
  } else if (name == "double") {
    return vineyard::ConvertToArrowType<double>::TypeValue();
  } else if (name == "string" || name == "std::string" || name == "str" ||
             name == "std::__1::string") {
    return vineyard::ConvertToArrowType<std::string>::TypeValue();
  } else if (name == "null" || name == "NULL") {
    return arrow::null();
  } else {
    LOG(ERROR) << "Unsupported data type: " << name;
    return nullptr;
  }
}

boost::leaf::result<std::shared_ptr<arrow::Table>> SyncSchema(
    const std::shared_ptr<arrow::Table>& table,
    const grape::CommSpec& comm_spec) {
  std::shared_ptr<arrow::Schema> local_schema =
      table != nullptr ? table->schema() : nullptr;
  std::vector<std::shared_ptr<arrow::Schema>> schemas;

  GlobalAllGatherv(local_schema, schemas, comm_spec);
  BOOST_LEAF_AUTO(normalized_schema, TypeLoosen(schemas));

  if (table == nullptr) {
    std::shared_ptr<arrow::Table> table_out;
    VY_OK_OR_RAISE(
        vineyard::EmptyTableBuilder::Build(normalized_schema, table_out));
    return table_out;
  } else {
    return CastTableToSchema(table, normalized_schema);
  }
}

// Inspired by arrow::compute::Cast
boost::leaf::result<std::shared_ptr<arrow::Array>> CastIntToDouble(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
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
  std::shared_ptr<arrow::Array> out;
  ARROW_OK_OR_RAISE(builder.Finish(&out));
  ARROW_OK_OR_RAISE(out->ValidateFull());
  return out;
}

// Timestamp value are stored as as number of seconds, milliseconds,
// microseconds or nanoseconds since UNIX epoch.
// CSV reader can only produce timestamp in seconds.
boost::leaf::result<std::shared_ptr<arrow::Array>> CastDateToInt(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
  CHECK_OR_RAISE(in->type()->Equals(arrow::timestamp(arrow::TimeUnit::SECOND)));
  CHECK_OR_RAISE(to_type->Equals(arrow::int64()));
  auto array_data = in->data()->Copy();
  array_data->type = to_type;
  auto out = arrow::MakeArray(array_data);
  ARROW_OK_OR_RAISE(out->ValidateFull());
  return out;
}

boost::leaf::result<std::shared_ptr<arrow::Array>> CastStringToBigString(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
  auto array_data = in->data()->Copy();
  auto offset = array_data->buffers[1];
  using from_offset_type = typename arrow::StringArray::offset_type;
  using to_string_offset_type = typename arrow::LargeStringArray::offset_type;
  auto raw_value_offsets_ =
      offset == NULLPTR
          ? NULLPTR
          : reinterpret_cast<const from_offset_type*>(offset->data());
  std::vector<to_string_offset_type> to_offset(offset->size() /
                                               sizeof(from_offset_type));
  for (size_t i = 0; i < to_offset.size(); ++i) {
    to_offset[i] = raw_value_offsets_[i];
  }
  std::shared_ptr<arrow::Buffer> buffer;
  arrow::TypedBufferBuilder<to_string_offset_type> buffer_builder;
  buffer_builder.Append(to_offset.data(), to_offset.size());
  buffer_builder.Finish(&buffer);
  array_data->type = to_type;
  array_data->buffers[1] = buffer;
  auto out = arrow::MakeArray(array_data);
  ARROW_OK_OR_RAISE(out->ValidateFull());
  return out;
}

boost::leaf::result<std::shared_ptr<arrow::Array>> CastNullToOthers(
    const std::shared_ptr<arrow::Array>& in,
    const std::shared_ptr<arrow::DataType>& to_type) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  ARROW_OK_OR_RAISE(
      arrow::MakeBuilder(arrow::default_memory_pool(), to_type, &builder));
  ARROW_OK_OR_RAISE(builder->AppendNulls(in->length()));
  std::shared_ptr<arrow::Array> out;
  ARROW_OK_OR_RAISE(builder->Finish(&out));
  ARROW_OK_OR_RAISE(out->ValidateFull());
  return out;
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
        if (from_type->Equals(arrow::int64()) &&
            to_type->Equals(arrow::float64())) {
          BOOST_LEAF_AUTO(new_array, CastIntToDouble(array, to_type));
          chunks.push_back(new_array);
        } else if (from_type->Equals(
                       arrow::timestamp(arrow::TimeUnit::SECOND)) &&
                   to_type->Equals(arrow::int64())) {
          BOOST_LEAF_AUTO(new_array, CastDateToInt(array, to_type));
          chunks.push_back(new_array);
        } else if (from_type->Equals(arrow::utf8()) &&
                   to_type->Equals(arrow::large_utf8())) {
          BOOST_LEAF_AUTO(new_array, CastStringToBigString(array, to_type));
          chunks.push_back(new_array);
        } else if (from_type->Equals(arrow::null())) {
          BOOST_LEAF_AUTO(new_array, CastNullToOthers(array, to_type));
          chunks.push_back(new_array);
        } else {
          RETURN_GS_ERROR(ErrorCode::kDataTypeError,
                          "Unexpected type: " + to_type->ToString() +
                              "; Origin type: " + from_type->ToString());
        }
        VLOG(2) << "Cast " << from_type->ToString() << " To "
                << to_type->ToString();
      }
      auto chunk_array = std::make_shared<arrow::ChunkedArray>(chunks, to_type);
      new_columns.push_back(chunk_array);
    } else {
      new_columns.push_back(col);
    }
  }
  return arrow::Table::Make(schema, new_columns);
}

}  // namespace vineyard

namespace grape {
inline grape::InArchive& operator<<(grape::InArchive& in_archive,
                                    std::shared_ptr<arrow::Schema>& schema) {
  if (schema != nullptr) {
    std::shared_ptr<arrow::Buffer> out;
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(arrow::ipc::SerializeSchema(
        *schema, nullptr, arrow::default_memory_pool(), &out));
#elif defined(ARROW_VERSION) && ARROW_VERSION < 2000000
    CHECK_ARROW_ERROR_AND_ASSIGN(
        out, arrow::ipc::SerializeSchema(*schema, nullptr,
                                         arrow::default_memory_pool()));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(
        out,
        arrow::ipc::SerializeSchema(*schema, arrow::default_memory_pool()));
#endif
    in_archive.AddBytes(out->data(), out->size());
  }
  return in_archive;
}

inline grape::OutArchive& operator>>(grape::OutArchive& out_archive,
                                     std::shared_ptr<arrow::Schema>& schema) {
  if (!out_archive.Empty()) {
    auto buffer = std::make_shared<arrow::Buffer>(
        reinterpret_cast<const uint8_t*>(out_archive.GetBuffer()),
        out_archive.GetSize());
    arrow::io::BufferReader reader(buffer);
#if defined(ARROW_VERSION) && ARROW_VERSION < 17000
    CHECK_ARROW_ERROR(arrow::ipc::ReadSchema(&reader, nullptr, &schema));
#else
    CHECK_ARROW_ERROR_AND_ASSIGN(schema,
                                 arrow::ipc::ReadSchema(&reader, nullptr));
#endif
  }
  return out_archive;
}
}  // namespace grape

#endif  // MODULES_BASIC_DS_ARROW_UTILS_H_
