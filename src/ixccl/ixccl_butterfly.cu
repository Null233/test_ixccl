#include "ixccl_allreduce.cuh"

int main(int argc, char *argv[])
{
    int rank, size, localRank;

    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    getLocalRank(rank, size, &localRank);
    CUDACHECK(cudaSetDevice(localRank));
    if (rank == 0)
        printf("IXCCL_BUTTERFLY\n");

    // initializing data buffer on device
    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));
    float **sendbuff, *recvbuff;
    sendbuff = (float **)malloc(DATA_NUM * sizeof(float *));

    // get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    ncclComm_t comm;
    if (rank == 0)
        ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    for (int size_i = 0; size_i < data_sizes.size(); size_i++) {
        // Preparing data
        for (int j = 0; j < DATA_NUM; j++) {
            CUDACHECK(cudaMalloc(&sendbuff[j], data_sizes[size_i] * sizeof(float)));
            CUDACHECK(cudaMalloc(&recvbuff, data_sizes[size_i] * sizeof(float)));
            CUDACHECK(cudaMemcpy(sendbuff[j], &vector<float>(data_sizes[size_i], rand())[0],
                                 data_sizes[size_i], cudaMemcpyHostToDevice));
        }

        double avg = 0;
        IXCCL_PERF_COUNTER(NCCLCHECK(ixcclButterfly(
            sendbuff[run_i % DATA_NUM], recvbuff, data_sizes[size_i], rank, size, ncclFloat, comm, s)));

        PRINT(printf("DATA SIZE: %-10d takes %.3lfms\n", data_sizes[size_i], avg  / CLOCKS_PER_SEC * 1000));
    }

    // free device buffers
    for (int j = 0; j < DATA_NUM; j++) {
        CUDACHECK(cudaFree(sendbuff[j]));
    }
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    ncclCommDestroy(comm);

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    return 0;
}