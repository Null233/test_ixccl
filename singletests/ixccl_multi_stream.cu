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
    char hostname[1024];
    getHostName(hostname, 1024);
    printf("Hostname: %s, Rank: %d, localRank: %d, Size: %d\n", hostname, rank, localRank, size);
    CUDACHECK(cudaSetDevice(localRank));
    if (rank == 0)
        printf("IXCCL_MULTIPLE_STREAMS_TEST\n");

    // initializing data buffer on device
    int nStream = size;
    cudaStream_t s[nStream];
    for (int i = 0; i < nStream; i++) {
        CUDACHECK(cudaStreamCreate(&s[i]));
    }
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
    ncclGroupStart();

    for (int s_i = 0; s_i < nStream; s_i++) {
        int peer = size - 1 - rank;
        void *sendbase = GET_BASE(float, sendbuff, s_i, (dataSize / nStream));
        NCCLCHECK(ncclSend(sendbase, (dataSize / nStream), ncclFloat, peer,
                           comm, s[s_i]));
        NCCLCHECK(ncclRecv(recvbuff, (dataSize / nStream), ncclFloat, peer,
                           comm, s[s_i]));
    }

    ncclGroupEnd();

    for (int i = 0; i < nStream; i++) {
        CUDACHECK(cudaStreamSynchronize(s[i]));
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