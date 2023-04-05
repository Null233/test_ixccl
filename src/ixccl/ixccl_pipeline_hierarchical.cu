#include "ixccl_allreduce.cuh"

int ConstructComms(int rank, int size, int localRank, int localSize, ncclComm_t *comms)
{

    int scale = localSize;
    ncclUniqueId ids[scale];
    MPI_Comm mpi_comms[scale];
    // Uniformly getting ncclUniqueId from rank 0 and broadcasting to all
    if (rank == 0) {
        for (int i = 0; i < scale; i++) {
            ncclGetUniqueId(&ids[i]);
        }
    }
    MPICHECK(MPI_Bcast((void *)&ids[0], scale * sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));

    for (int i = 0; i < scale; i++) {
        int color = localRank == i ? i : MPI_UNDEFINED;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &mpi_comms[i]);
    }
    for (int i = 0; i < scale; i++) {
        if (localRank == i) {
            int world_leader_size, world_leader_rank;
            MPI_Comm_size(mpi_comms[i], &world_leader_size);
            MPI_Comm_rank(mpi_comms[i], &world_leader_rank);
            NCCLCHECK(ncclCommInitRank(&comms[i], world_leader_size, ids[i], world_leader_rank));
        }
    }

    return MPI_SUCCESS;
}

int main(int argc, char *argv[])
{
    int rank, size, localRank;

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
        printf("IXCCL_PIPELINE_HIERARCHICAL\n");

    // initializing device buffer and stream
    CUDACHECK(cudaSetDevice(localRank));
    int nStream = localSize;
    cudaStream_t s[nStream];
    for (int i = 0; i < nStream; i++) {
        CUDACHECK(cudaStreamCreate(&s[i]));
    }
    float **sendbuff, *recvbuff;
    sendbuff = (float **)malloc(DATA_NUM * sizeof(float *));

    /**
     * We need localSize + 1 ids
     * including local communicator, and communicators for each rank locally
     * ids[0] is the id for local communicator
     */

    // initializing NCCL
    ncclUniqueId id_local;
    ncclComm_t comm_local;
    if (localRank == 0)
        ncclGetUniqueId(&id_local);
    MPICHECK(MPI_Bcast((void *)&id_local, sizeof(id_local), MPI_BYTE, 0, COMM_LOCAL));
    NCCLCHECK(ncclCommInitRank(&comm_local, localSize, id_local, localRank));
    ncclComm_t comms[localSize];
    ConstructComms(rank, size, localRank, localSize, comms);
    if (rank == 0)
        printf("FINISH INITIALIZING NCCL\n");

    // starting performance counter
    for (int size_i = 0; size_i < data_sizes.size(); size_i++) {
        // Preparing data
        for (int j = 0; j < DATA_NUM; j++) {
            CUDACHECK(cudaMalloc(&sendbuff[j], data_sizes[size_i] * sizeof(float)));
            CUDACHECK(cudaMalloc(&recvbuff, data_sizes[size_i] * sizeof(float)));
            CUDACHECK(cudaMemcpy(sendbuff[j], &vector<float>(data_sizes[size_i], rand())[0],
                                 data_sizes[size_i], cudaMemcpyHostToDevice));
        }

        double start, end, avg = 0;
        for (int run = 0; run < RUN_ROUND; run++) {
            start = double(clock());
            ixcclPipelineHier(sendbuff[run % DATA_NUM], recvbuff, data_sizes[size_i], ncclFloat,
                              ncclSum, comm_local, comms, COMM_LOCAL, nStream, s);
            for (int i = 0; i < nStream; i++) {
                CUDACHECK(cudaStreamSynchronize(s[i]));
            }
            end = double(clock());
            avg += (end - start) / RUN_ROUND;
        }

        PRINT(printf("DATA SIZE: %-10d takes %.3lfms\n", data_sizes[size_i],
                     double(avg) / CLOCKS_PER_SEC * 1000));
        for (int j = 0; j < DATA_NUM; j++) {
            CUDACHECK(cudaFree(sendbuff[j]));
        }
        CUDACHECK(cudaFree(recvbuff));
    }

    // free device buffers
    for (int j = 0; j < DATA_NUM; j++) {
        CUDACHECK(cudaFree(sendbuff[j]));
    }
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    ncclCommDestroy(comm_local);
    for (int i = 0; i < localSize; i++) {
        if (localRank == i) {
            ncclCommDestroy(comms[i]);
        }
    }

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    return 0;
}