#include "ixccl_allreduce.cuh"

int main(int argc, char *argv[])
{
    int rank, size, localRank;

    int dataSize = 16384;

    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    getLocalRank(rank, size, &localRank);
    CUDACHECK(cudaSetDevice(localRank));
    if (rank == 0)
        printf("IXCCL_SMALL_SIZE_TEST\n");

    // initializing data buffer on device
    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));
    float *sendbuff, *recvbuff;
    CUDACHECK(cudaMalloc(&sendbuff, dataSize * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, dataSize * sizeof(float)));

    // get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0)
        ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    // start testing
    for(int run = 0; run < 100000; run++){
        // ixcclRing(sendbuff, recvbuff, dataSize, rank, size, ncclFloat, comm, s);
        ixcclAllreduce(sendbuff, recvbuff, dataSize, ncclFloat, ncclSum, comm, s);
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