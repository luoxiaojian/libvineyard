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

#include <stdio.h>

#include <fstream>
#include <string>

#include "glog/logging.h"

#include "client/client.h"

#include "graph/fragment/arrow_fragment.h"
#include "graph/loader/arrow_fragment_loader.h"

using namespace vineyard;  // NOLINT(build/namespaces)

using GraphType = ArrowFragment<property_graph_types::OID_TYPE,
                                property_graph_types::VID_TYPE>;
using LabelType = typename GraphType::label_id_t;

int main(int argc, char** argv) {
  if (argc < 6) {
    printf(
        "usage: ./arrow_fragment_test <ipc_socket> <e_label_num> <efiles...> "
        "<v_label_num> <vfiles...> [directed]\n");
    return 1;
  }
  int index = 1;
  std::string ipc_socket = std::string(argv[index++]);

  int edge_label_num = atoi(argv[index++]);
  std::vector<std::string> efiles;
  for (int i = 0; i < edge_label_num; ++i) {
    efiles.push_back(argv[index++]);
  }

  int vertex_label_num = atoi(argv[index++]);
  std::vector<std::string> vfiles;
  for (int i = 0; i < vertex_label_num; ++i) {
    vfiles.push_back(argv[index++]);
  }

  int directed = 1;
  if (argc > index) {
    directed = atoi(argv[index]);
  }

  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(ipc_socket));

  LOG(INFO) << "Connected to IPCServer: " << ipc_socket;

  grape::InitMPIComm();
  grape::CommSpec comm_spec;
  comm_spec.Init(MPI_COMM_WORLD);

  vineyard::ObjectID fragment_id = InvalidObjectID();
  {
    auto loader =
        std::make_unique<ArrowFragmentLoader<property_graph_types::OID_TYPE,
                                             property_graph_types::VID_TYPE>>(
            client, comm_spec, efiles, vfiles, directed != 0);
    fragment_id = boost::leaf::try_handle_all(
        [&loader]() { return loader->LoadFragment(); },
        [](const GSError& e) {
          LOG(FATAL) << e.error_msg;
          return 0;
        },
        [](const boost::leaf::error_info& unmatched) {
          LOG(FATAL) << "Unmatched error " << unmatched;
          return 0;
        });
  }

  grape::FinalizeMPIComm();

  LOG(INFO) << "[worker-" << comm_spec.worker_id()
            << "] loaded graph to vineyard: " << VYObjectIDToString(fragment_id)
            << " ...";

  LOG(INFO) << "Passed arrow fragment test...";

  return 0;
}
