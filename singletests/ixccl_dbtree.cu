#include "ixccl_allreduce.cuh"

int main(int argc, char *argv[])
{

    int rank, size, localRank;

    int dataSize = 102760448;

    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    getLocalRank(rank, size, &localRank);
    CUDACHECK(cudaSetDevice(localRank));
    if (rank == 0)
        printf("IXCCL_DBTREE_STANDALONE\n");

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));
    float *sendbuff, *recvbuff;
    CUDACHECK(cudaMalloc(&sendbuff, dataSize * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, dataSize * sizeof(float)));

    // get NCCL unique ID at rank 0 and broadcast it to all others
    if (rank == 0)
        ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // 使用double binary tree算法进行归约操作
    for (int dist = 1; dist < size; dist <<= 1) {
        ncclGroupStart();
        int sendTo = (rank + dist) % size;
        int recvFrom = (rank - dist + size) % size;

        ncclSend(sendbuff, dataSize, ncclFloat, sendTo, comm, s);
        ncclRecv(sendbuff, dataSize, ncclFloat, recvFrom, comm, s);

        ncclGroupEnd();
    }

    // free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    ncclCommDestroy(comm);

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    return 0;
}
