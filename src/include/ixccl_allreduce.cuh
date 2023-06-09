#ifndef IXCCL_ALLREDUCE_H
#define IXCCL_ALLREDUCE_H

#include "allreduce.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "nccl.h"

/* 1: IXCCL_ALLREDUCE; 2: ixcclRing; 3: ixcclButterfly; 4: ixcclTreeReduction; else: IXCCL_REDUCE */
#define LOCAL_IXCCL_REDUCE_ALGO 2
/* 1: ixcclBtreeBcast; 2: ixcclPipelineBtreeBcast; else IXCCL_BCAST */
#define LOCAL_IXCCL_BCAST_ALGO 2

#define nBlocks 1024

/************************** FUNCTION DECLARATIONS **************************/
__global__ void sum(float *, float *, const int);
__global__ void assign(float *, float *, const int);
ncclResult_t ixcclAllreduce(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                            cudaStream_t);
ncclResult_t ixcclRing(const void *, void *, size_t, int, int, ncclDataType_t, ncclComm_t,
                       cudaStream_t);
ncclResult_t ixcclButterfly(const void *, void *, size_t, int, int, ncclDataType_t, ncclComm_t,
                            cudaStream_t);
ncclResult_t ixcclBtreeBcast(const void *, void *, size_t, int, int, ncclDataType_t, ncclComm_t,
                             cudaStream_t);
ncclResult_t ixcclPipelineBtreeBcast(const void *, void *, size_t, int, int, ncclDataType_t,
                                     ncclComm_t, cudaStream_t);
ncclResult_t ixcclTreeReduction(const void *, void *, size_t, int, int, ncclDataType_t, ncclComm_t,
                                cudaStream_t);
ncclResult_t ixcclHierarchical(void *, void *, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                               ncclComm_t, cudaStream_t, MPI_Comm);

#define CUDACHECK(cmd)                                                                             \
    do {                                                                                           \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define NCCLCHECK(cmd)                                                                             \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define IXCCL_PERF_COUNTER(cmd)                                                                    \
    do {                                                                                           \
        double begin, end;                                                                         \
        for (int run_i = 0; run_i < RUN_ROUND; run_i++) {                                          \
            begin = double(clock());                                                               \
            cmd;                                                                                   \
            CUDACHECK(cudaStreamSynchronize(s));                                                   \
            end = double(clock());                                                                 \
            avg += (end - begin) / RUN_ROUND;                                                      \
        }                                                                                          \
    } while (0)

#define GRID(count, blocks) ((count + blocks - 1) / blocks)

typedef struct MPINCCL_COMM_T {
    MPI_Comm mpi_comm;
    ncclUniqueId id;
    ncclComm_t nccl_comm;
} MPINCCL_COMM;

/************************** HELPER FUNCTIONS **************************/

__global__ void sum(float *send, float *tmp, const int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        send[i] += tmp[i];
}

__global__ void assign(float *send, float *tmp, const int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        send[i] = tmp[i];
}

/************************** WARPPED MPI FUNCTIONS **************************/

ncclResult_t ixcclAllreduce(const void *sendbuff, void *recvbuff, size_t count,
                            ncclDataType_t dtype, ncclRedOp_t op, ncclComm_t comm,
                            cudaStream_t stream)
{
    NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, dtype, op, comm, stream));

    return ncclSuccess;
}

ncclResult_t ixcclMultiStreamAllreduce(const void *sendbuff, void *recvbuff, size_t count,
                                       ncclDataType_t dtype, ncclRedOp_t op, ncclComm_t comm,
                                       int nStream, cudaStream_t *streams)
{

    int countInAllr = count / nStream;

    for (int i = 0; i < nStream; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, countInAllr);
        NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, countInAllr, dtype, op, comm, streams[i]));
    }

    return ncclSuccess;
}

/************************** USER COLLECTIVE FUNCTIONS **************************/

ncclResult_t ixcclRing(const void *sendbuff, void *recvtemp, size_t count, int rank, int size,
                       ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream)
{

    int sendTo = (rank + 1) % size;
    int recvFrom = (rank - 1 + size) % size;
    int count_in_ring = count / size;

    // Reduce-Scatter
    for (int i = 0; i < size - 1; i++) {
        void *sendBase = GET_BASE(float, sendbuff, ((rank - i + size) % size), count_in_ring);
        void *recvBase = GET_BASE(float, sendbuff, ((rank - 1 - i + size) % size), count_in_ring);

        ncclGroupStart();
        NCCLCHECK(ncclSend(sendBase, count_in_ring, dtype, sendTo, comm, stream));
        NCCLCHECK(ncclRecv(recvtemp, count_in_ring, dtype, recvFrom, comm, stream));
        ncclGroupEnd();

        // Sum
        sum<<<GRID(count_in_ring, nBlocks), nBlocks, 0, stream>>>((float *)recvBase,
                                                                  (float *)recvtemp, count_in_ring);
    }

    // All-Gather
    for (int i = 0; i < size - 1; i++) {
        void *sendBase = GET_BASE(float, sendbuff, ((rank + 1 - i + size) % size), count_in_ring);
        void *recvBase = GET_BASE(float, sendbuff, ((rank - i + size) % size), count_in_ring);

        ncclGroupStart();
        NCCLCHECK(ncclSend(sendBase, count_in_ring, dtype, sendTo, comm, stream));
        NCCLCHECK(ncclRecv(recvtemp, count_in_ring, dtype, recvFrom, comm, stream));
        ncclGroupEnd();

        // assign
        assign<<<GRID(count_in_ring, nBlocks), nBlocks, 0, stream>>>(
            (float *)recvBase, (float *)recvtemp, count_in_ring);
    }

    return ncclSuccess;
}

ncclResult_t ixcclMultiStreamRing(const void *sendbuff, void *recvtemp, size_t count, int rank,
                                  int size, ncclDataType_t dtype, ncclComm_t comm, int nStream,
                                  cudaStream_t *streams)
{

    for (int i = 0; i < nStream; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, count / nStream);
        ixcclRing(sendBase, recvtemp, count / nStream, rank, size, dtype, comm, streams[i]);
    }

    return ncclSuccess;
}

ncclResult_t ixcclButterfly(const void *sendbuff, void *recvtemp, size_t count, int rank, int size,
                            ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream)
{

    int n = log2(size);

    for (int i = 0; i < n; i++) {
        int comm_node = rank ^ (1 << i);

        ncclGroupStart();
        NCCLCHECK(ncclSend(sendbuff, count, dtype, comm_node, comm, stream));
        NCCLCHECK(ncclRecv(recvtemp, count, dtype, comm_node, comm, stream));
        ncclGroupEnd();

        sum<<<GRID(count, nBlocks), nBlocks, 0, stream>>>((float *)sendbuff, (float *)recvtemp,
                                                          count);
    }

    return ncclSuccess;
}

ncclResult_t ixcclMultiStreamButterfly(const void *sendbuff, void *recvtemp, size_t count, int rank,
                                       int size, ncclDataType_t dtype, ncclComm_t comm, int nStream,
                                       cudaStream_t *streams)
{
    for (int i = 0; i < nStream; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, count / nStream);
        ixcclButterfly(sendBase, recvtemp, count / nStream, rank, size, dtype, comm, streams[i]);
    }

    return ncclSuccess;
}

/**************** BTREE FUNCTIONS ****************/

ncclResult_t ixcclBtreeBcast(const void *sendbuff, void *tmp, size_t count, int rank, int size,
                             ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream)
{

    int recvFrom, sendToL, sendToR, parentChildType;
    GetBtree(rank, size, &recvFrom, &sendToL, &sendToR, &parentChildType);

    ncclGroupStart();
    if (recvFrom >= 0) {
        NCCLCHECK(ncclRecv(tmp, count, dtype, recvFrom, comm, stream));
        assign<<<GRID(count, nBlocks), nBlocks, 0, stream>>>(TO_ARRAY(sendbuff), TO_ARRAY(tmp),
                                                             count);
    }
    if (sendToL >= 0) {
        NCCLCHECK(ncclSend(sendbuff, count, dtype, sendToL, comm, stream));
    }
    if (sendToR >= 0) {
        NCCLCHECK(ncclSend(sendbuff, count, dtype, sendToR, comm, stream));
    }
    ncclGroupEnd();
    return ncclSuccess;
}

ncclResult_t ixcclPipelineBtreeBcast(const void *sendbuff, void *tmp, size_t count, int rank,
                                     int size, ncclDataType_t dtype, ncclComm_t comm,
                                     cudaStream_t stream)
{
    int recvFrom, sendToL, sendToR, parentChildType;
    int countInTree = count / size;

    GetBtree(rank, size, &recvFrom, &sendToL, &sendToR, &parentChildType);

    for (int i = 0; i < size; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, countInTree);
        void *recvBase = GET_BASE(float, sendbuff, i, countInTree);

        ncclGroupStart();
        if (recvFrom >= 0) {
            NCCLCHECK(ncclRecv(recvBase, countInTree, dtype, recvFrom, comm, stream));
        }
        if (sendToL >= 0) {
            NCCLCHECK(ncclSend(sendbuff, countInTree, dtype, sendToL, comm, stream));
        }
        if (sendToR >= 0) {
            NCCLCHECK(ncclSend(sendbuff, countInTree, dtype, sendToR, comm, stream));
        }
        ncclGroupEnd();
    }
    return ncclSuccess;
}

ncclResult_t ixcclTreeReduction(const void *sendbuff, void *tmp, size_t count, int rank, int size,
                                ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream)
{

    int level = (int)log2(size);
    for (int i = 0; i < level; i++) {
        int commNode = rank ^ (1 << i);
        ncclGroupStart();
        if (rank < commNode) {
            NCCLCHECK(ncclRecv(tmp, count, dtype, commNode, comm, stream));
            sum<<<GRID(count, nBlocks), nBlocks, 0, stream>>>((float *)sendbuff, (float *)tmp,
                                                              count);
        } else {
            NCCLCHECK(ncclSend(sendbuff, count, dtype, commNode, comm, stream));
        }
        ncclGroupEnd();
    }
    return ncclSuccess;
}

/**************** HIERARCHICAL FUNCTIONS ****************/

ncclResult_t ixcclHierarchical(void *sendbuff, void *tmp, size_t count, ncclDataType_t dtype,
                               ncclRedOp_t op, ncclComm_t comm_local, ncclComm_t comm_world_main,
                               cudaStream_t stream, MPI_Comm COMM_LOCAL)
{

    int localRank, localSize;
    MPI_Comm_rank(COMM_LOCAL, &localRank);
    MPI_Comm_size(COMM_LOCAL, &localSize);

#if LOCAL_IXCCL_REDUCE_ALGO == 1
    /* Local NCCL Allreduce */
    NCCLCHECK(ncclAllReduce(sendbuff, sendbuff, count, dtype, op, comm_local, stream));
#elif LOCAL_IXCCL_REDUCE_ALGO == 2
    /* Local ixccl Ring */
    NCCLCHECK(ixcclRing(sendbuff, tmp, count, localRank, localSize, dtype, comm_local, stream));
#elif LOCAL_IXCCL_REDUCE_ALGO == 3
    /* Local ixccl Butterfly */
    NCCLCHECK(
        ixcclButterfly(sendbuff, tmp, count, localRank, localSize, dtype, comm_local, stream));
#elif LOCAL_IXCCL_REDUCE_ALGO == 4
    /* Local ixccl Tree Reduction */
    NCCLCHECK(
        ixcclTreeReduction(sendbuff, tmp, count, localRank, localSize, dtype, comm_local, stream));
#else
    /* Local NCCL Reduce */
    NCCLCHECK(ncclReduce(sendbuff, sendbuff, count, dtype, op, LOCAL_HIER_ROOT, comm_local, stream));
#endif

    /* World Main NCCL Allreduce */
    if (localRank == LOCAL_HIER_ROOT)
        NCCLCHECK(ncclAllReduce(sendbuff, sendbuff, count, dtype, op, comm_world_main, stream));

#if LOCAL_IXCCL_BCAST_ALGO == 1
    /* Local ixccl Btree broadcast */
    NCCLCHECK(
        ixcclBtreeBcast(sendbuff, tmp, count, localRank, localSize, dtype, comm_local, stream));
#elif LOCAL_IXCCL_BCAST_ALGO == 2
    /* Local ixccl Btree broadcast */
    NCCLCHECK(ixcclPipelineBtreeBcast(sendbuff, tmp, count, localRank, localSize, dtype, comm_local,
                                      stream));
#else
    /* Local Broadcast */
    NCCLCHECK(ncclBroadcast(sendbuff, sendbuff, count, dtype, LOCAL_HIER_ROOT, comm_local, stream));
#endif
    return ncclSuccess;
}

ncclResult_t ixcclMultiStreamHier(void *sendbuff, void *tmp, size_t count, ncclDataType_t dtype,
                                  ncclRedOp_t op, ncclComm_t comm_local, ncclComm_t comm_world_main,
                                  MPI_Comm COMM_LOCAL, int nStream, cudaStream_t *streams)
{

    for (int i = 0; i < nStream; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, count / nStream);
        ixcclHierarchical(sendBase, tmp, count / nStream, dtype, op, comm_local, comm_world_main,
                          streams[i], COMM_LOCAL);
    }

    return ncclSuccess;
}

ncclResult_t ixcclPipelineHier(void *sendbuff, void *tmp, size_t count, ncclDataType_t dtype,
                               ncclRedOp_t op, ncclComm_t comm_local, ncclComm_t *comms,
                               MPI_Comm COMM_LOCAL, int nStream, cudaStream_t *streams)
{
    int localRank, localSize;
    MPI_Comm_rank(COMM_LOCAL, &localRank);
    MPI_Comm_size(COMM_LOCAL, &localSize);

    int countInHier = count / nStream;

    for (int i = 0; i < nStream; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, countInHier);
        NCCLCHECK(
            ncclReduce(sendBase, sendBase, countInHier, dtype, op, i, comm_local, streams[i]));
    }
    for (int i = 0; i < nStream; i++) {
        if (localRank == i) {
            void *sendBase = GET_BASE(float, sendbuff, i, countInHier);
            NCCLCHECK(
                ncclAllReduce(sendBase, sendBase, countInHier, dtype, op, comms[i], streams[i]));
        }
    }
    for (int i = 0; i < nStream; i++) {
        void *sendBase = GET_BASE(float, sendbuff, i, countInHier);
        NCCLCHECK(ncclBroadcast(sendBase, sendBase, countInHier, dtype, i, comm_local, streams[i]));
    }

    return ncclSuccess;
}

#endif