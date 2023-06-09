#include "ixccl_allreduce.cuh"

int main(int argc, char *argv[])
{
    int rank, size, localRank;

    int dataSize = 102760448;

    // initializing MPI_COMM_WORLD
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    getLocalRank(rank, size, &localRank);

    // initializing COMM_LOCAL and COMM_WORLD_MAIN
    /**
     * We need to use MPI communicators to construct NCCL communicators
     * After the construction of local and inter-node communicators,
     * check if rank in local communicator is the same with localRank
     */
    MPI_Comm COMM_LOCAL;
    MPI_Comm COMM_WORLD_MAIN;
    split_world(rank, localRank, &COMM_LOCAL, &COMM_WORLD_MAIN);
    int localSize, mlocalRank;
    int worldMainSize, worldMainRank;
    MPICHECK(MPI_Comm_rank(COMM_LOCAL, &mlocalRank));
    MPICHECK(MPI_Comm_size(COMM_LOCAL, &localSize));
    if (mlocalRank != localRank)
        printf("!!!!!!  localRank != mpi_localRank  !!!!!!");
    /**
     * COMM_WORLD_MAIN as a inter-node communicator is only used if localRank == 0
     */
    if (localRank == 0)
        MPICHECK(MPI_Comm_rank(COMM_WORLD_MAIN, &worldMainRank));
    if (localRank == 0)
        MPICHECK(MPI_Comm_size(COMM_WORLD_MAIN, &worldMainSize));
    if (rank == 0)
        printf("IXCCL__MULTI_STREAM_HIERARCHICAL\n");

    if (rank == 0) {
#if LOCAL_IXCCL_REDUCE_ALGO == 1
        printf("LOCAL_IXCCL_ALLREDUCE\n");
#elif LOCAL_IXCCL_REDUCE_ALGO == 2
        printf("LOCAL_IXCCL_RING\n");
#elif LOCAL_IXCCL_REDUCE_ALGO == 3
        printf("LOCAL_IXCCL_BUTTERFLY\n");
#elif LOCAL_IXCCL_REDUCE_ALGO == 4
        printf("LOCAL_IXCCL_TREE_REDUCTION\n");
#else
        printf("LOCAL_IXCCL_REDUCE\n");
#endif

#if LOCAL_IXCCL_BCAST_ALGO == 1
        printf("LOCAL_IXCCL_BTREE_BROADCAST\n");
#elif LOCAL_IXCCL_BCAST_ALGO == 2
        printf("LOCAL_IXCCL_PIPELINE_BTREE_BROADCAST\n");
#else
        printf("LOCAL_IXCCL_BROADCAST\n");
#endif
    }

    // initializing device buffer and stream
    CUDACHECK(cudaSetDevice(localRank));
    int nStream = size;
    cudaStream_t s[nStream];
    for (int i = 0; i < nStream; i++) {
        CUDACHECK(cudaStreamCreate(&s[i]));
    }
    float *sendbuff, *recvbuff;
    CUDACHECK(cudaMalloc(&sendbuff, dataSize * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, dataSize * sizeof(float)));

    // get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id_local, id_main;
    /**
     * allocate unique id if localRank == 0 and be broadcasted to all local ranks
     * every node has a duplicate nccl communicator comm_local
     * only main local ranks (localRank = 0) have comm_world_main
     */
    if (localRank == 0)
        ncclGetUniqueId(&id_local);
    MPICHECK(MPI_Bcast((void *)&id_local, sizeof(id_local), MPI_BYTE, 0, COMM_LOCAL));
    if (worldMainRank == 0)
        ncclGetUniqueId(&id_main);
    if (localRank == 0)
        MPICHECK(MPI_Bcast((void *)&id_main, sizeof(id_main), MPI_BYTE, 0, COMM_WORLD_MAIN));

    // initializing NCCL
    ncclComm_t comm_local, comm_world_main;
    NCCLCHECK(ncclCommInitRank(&comm_local, localSize, id_local, localRank));
    if (localRank == 0)
        NCCLCHECK(ncclCommInitRank(&comm_world_main, worldMainSize, id_main, worldMainRank));
    if (rank == 0)
        printf("Finish initializing NCCL\n");

    // starting performance counter
    double start, end, avg = 0;
    for (int run = 0; run < 100; run++) {
        start = double(clock());
        ixcclMultiStreamHier(sendbuff, recvbuff, dataSize, ncclFloat, ncclSum, comm_local,
                             comm_world_main, COMM_LOCAL, nStream, s);
        for (int i = 0; i < nStream; i++) {
            CUDACHECK(cudaStreamSynchronize(s[i]));
        }
        end = double(clock());
        avg += (end - start) / 1000;
    }
    if (rank == 0)
        printf("Takes: %f\n", avg / CLOCKS_PER_SEC * 1000);

    // free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    ncclCommDestroy(comm_local);
    if (localRank == 0)
        ncclCommDestroy(comm_world_main);

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    return 0;
}