#ifndef DIFACTO_NODE_ID_H_
#define DIFACTO_NODE_ID_H_
namespace difacto {

class NodeID {
 public:
  /** \brief node ID for the scheduler */
  const static int kScheduler = 1;
  /**
   * \brief the server node group ID
   *
   * group id can be combined:
   * - kServerGroup + kScheduler means all server nodes and the scheuduler
   * - kServerGroup + kWorkerGroup means all server and worker nodes
   */
  const static int kServerGroup = 2;

  /** \brief the worker node group ID */
  const static int kWorkerGroup = 4;

  static int Encode(int group, int rank) {
    return group + (rank+1) * 8;
  }

};
}  // namespace difacto

#endif  // DIFACTO_NODE_ID_H_
