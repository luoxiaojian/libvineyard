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

#ifndef SRC_SERVER_ASYNC_RPC_SERVER_H_
#define SRC_SERVER_ASYNC_RPC_SERVER_H_

#include <sys/param.h>

#include <string>

#include "boost/asio.hpp"

#include "server/async/socket_server.h"
#include "server/server/vineyard_server.h"

namespace vineyard {

/**
 * @brief A kind of server that supports remote procedure call (RPC)
 *
 */
class RPCServer : public SocketServer {
 public:
  explicit RPCServer(vs_ptr_t vs_ptr);

  ~RPCServer() override;

  void Start() override;

  std::string Endpoint() {
    char hostname[MAXHOSTNAMELEN];
    gethostname(hostname, MAXHOSTNAMELEN);
    return std::string(hostname) + ":" + rpc_spec_.get<std::string>("port");
  }

 private:
  asio::ip::tcp::endpoint getEndpoint(asio::io_context&);

  void doAccept() override;

  const ptree rpc_spec_;
  asio::ip::tcp::acceptor acceptor_;
};

}  // namespace vineyard

#endif  // SRC_SERVER_ASYNC_RPC_SERVER_H_
