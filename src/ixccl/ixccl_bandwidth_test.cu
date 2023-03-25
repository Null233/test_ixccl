#include "ixccl_allreduce.cuh"
/**
 * Construct a two dimensional vector, which is a square matrix
 * vector as N * N scale, and uses node_i and node_j as index
 * When node_i == rank, function will get a ncclGetUniqueId and broadcast it to
 * node_j
 */
int ConstructComms(int rank, int size, vector<vector<MPINCCL_COMM>> &comms)
{

    // To split MPI_COMM_WORLD into communicators between any two nodes
    MPI_Group worldGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

#ifdef BROADCAST_FROM_0
    int scale = size * (size - 1) / 2;
    ncclUniqueId ids[scale];
    // Uniformly getting ncclUniqueId from rank 0 and broadcasting to all
    if (rank == 0) {
        for (int i = 0; i < scale; i++) {
            ncclGetUniqueId(&ids[i]);
        }
    }
    MPICHECK(MPI_Bcast((void *)&ids[0], scale * sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            int index_ids = size * i + j - ((i + 1) * (i + 2) / 2);
            comms[i][j].id = ids[index_ids];
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        // Define comms[i][i] to be NULL
        comms[i][i].mpi_comm = MPI_COMM_NULL;
        for (int j = i + 1; j < size; j++) {
            // Construct mpi communicator for any two nodes
            MPI_Group group;
            const int groupN[2] = {i, j};
            MPI_Group_incl(worldGroup, 2, &groupN[0], &group);
            MPI_Comm_create(MPI_COMM_WORLD, group, &(comms[i][j].mpi_comm));

            // when i == rank or j == rank, constructor needs to allocate NCCL
            // communicator
            if (i == rank || j == rank) {
#ifndef BROADCAST_FROM_0
                if (i == rank)
                    ncclGetUniqueId(&(comms[i][j].id));
                MPICHECK(MPI_Bcast((void *)&comms[i][j].id, sizeof(comms[i][j].id), MPI_BYTE, 0,
                                   comms[i][j].mpi_comm));
#endif
                // use mpi_rank and mpi_size to construct NCCL communicator
                int mpiRank, mpiSize;
                MPI_Comm_rank(comms[i][j].mpi_comm, &mpiRank);
                MPI_Comm_size(comms[i][j].mpi_comm, &mpiSize);
                NCCLCHECK(
                    ncclCommInitRank(&(comms[i][j].nccl_comm), mpiSize, comms[i][j].id, mpiRank));
#ifdef DEBUG
                int countInNccl, deviceInNccl, rankInNccl;
                char *hostname = (char *)malloc(1024 * sizeof(char));
                getHostName(hostname, 1024);
                NCCLCHECK(ncclCommCount(comms[i][j].nccl_comm, &countInNccl));
                NCCLCHECK(ncclCommCuDevice(comms[i][j].nccl_comm, &deviceInNccl));
                NCCLCHECK(ncclCommUserRank(comms[i][j].nccl_comm, &rankInNccl));
                printf("Rank: %d, i:%d, j: %d, mpiSize: %d, mpiRank: %d\n", rank, i, j, mpiSize,
                       mpiRank);
                printf("host: %-7s, Rank: %d, i:%d, j: %d, ncclSize: %d, "
                       "ncclRank: %d, ncclDevice: "
                       "%d\n",
                       hostname, rank, i, j, countInNccl, rankInNccl, deviceInNccl);
#endif
            }
        }
    }

    return MPI_SUCCESS;
}

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
        printf("IXCCL_BANDWIDTH_TEST\n");

    // initializing data buffer on device
    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));
    float **sendbuff, *recvbuff;
    sendbuff = (float **)malloc(DATA_NUM * sizeof(float *));

    // get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id_world;
    if (rank == 0)
        ncclGetUniqueId(&id_world);
    MPICHECK(MPI_Bcast((void *)&id_world, sizeof(id_world), MPI_BYTE, 0, MPI_COMM_WORLD));

    // initializing NCCL
    ncclComm_t comm_world;
    NCCLCHECK(ncclCommInitRank(&comm_world, size, id_world, rank));

    // getting NCCL unique ID at rank 0 and broadcast it to all others
    /**
     * Assume having N devices
     * Bandwidth test will be performed between any two devices
     * Communicators will be established between any two devices with only one
     * direction N * (N - 1) / 2 communicators are needed Each communicator
     * needs a ncclUniqueId Each node controls (N - 1 - rank) ncclUniqueId Each
     * node controls (N - 1 - rank) communicators, both ncclComm_t and MPI_Comm
     */
    vector<vector<MPINCCL_COMM>> comms(size, vector<MPINCCL_COMM>(size));
    ConstructComms(rank, size, comms);

    // start testing
    MPI_Barrier(MPI_COMM_WORLD);
#ifdef TEST_SR
    if (rank == 0)
        printf("IXCCL_SEND_RECV_TESTS\n");
    for (int node_i = 0; node_i < size; node_i++) {
        for (int node_j = node_i + 1; node_j < size; node_j++) {
            if (node_i == rank || node_j == rank) {
                int peer = rank == node_i ? 1 : 0;
                for (int size_i = 0; size_i < data_sizes.size(); size_i++) {
                    // Preparing data
                    for (int j = 0; j < DATA_NUM; j++) {
                        CUDACHECK(cudaMalloc(&sendbuff[j], data_sizes[size_i] * sizeof(float)));
                        CUDACHECK(cudaMalloc(&recvbuff, data_sizes[size_i] * sizeof(float)));
                        CUDACHECK(cudaMemcpy(sendbuff[j],
                                             &vector<float>(data_sizes[size_i], rand())[0],
                                             data_sizes[size_i], cudaMemcpyHostToDevice));
                    }

                    double begin, end, avg = 0;
                    for (int run = 0; run < RUN_ROUND; run++) {
                        begin = double(clock());
                        ncclGroupStart();
                        NCCLCHECK(ncclSend(sendbuff[run % DATA_NUM], data_sizes[size_i], ncclFloat,
                                           peer, comms[node_i][node_j].nccl_comm, s));
                        NCCLCHECK(ncclRecv(recvbuff, data_sizes[size_i], ncclFloat, peer,
                                           comms[node_i][node_j].nccl_comm, s));
                        ncclGroupEnd();
                        CUDACHECK(cudaStreamSynchronize(s));
                        end = double(clock());
                        avg += (end - begin) / RUN_ROUND;
                    }

                    if (node_i == rank)
                        printf("Rank %d with %d:\n", node_i, node_j);
                    if (node_i == rank)
                        printf("DATA SIZE: %-10d takes %.3lfms\n", data_sizes[size_i],
                               double(avg) / CLOCKS_PER_SEC * 1000);
                }
                usleep(1000000);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
#ifdef TEST_AR
    if (rank == 0)
        printf("IXCCL_ALLREDUCE_TESTS\n");
    for (int node_i = 0; node_i < size; node_i++) {
        for (int node_j = node_i + 1; node_j < size; node_j++) {
            if (node_i == rank || node_j == rank) {

                for (int size_i = 0; size_i < data_sizes.size(); size_i++) {
                    // Preparing data
                    for (int j = 0; j < DATA_NUM; j++) {
                        CUDACHECK(cudaMalloc(&sendbuff[j], data_sizes[size_i] * sizeof(float)));
                        CUDACHECK(cudaMalloc(&recvbuff, data_sizes[size_i] * sizeof(float)));
                        CUDACHECK(cudaMemcpy(sendbuff[j],
                                             &vector<float>(data_sizes[size_i], rand())[0],
                                             data_sizes[size_i], cudaMemcpyHostToDevice));
                    }

                    double avg = 0;
                    IXCCL_PERF_COUNTER(NCCLCHECK(
                        ixcclAllreduce(sendbuff[i % DATA_NUM], recvbuff, data_sizes[size_i],
                                       ncclFloat, ncclSum, comms[node_i][node_j].nccl_comm, s)));
                    if (node_i == rank)
                        printf("Rank %d with %d:\n", node_i, node_j);
                    if (node_i == rank)
                        printf("DATA SIZE: %-10d takes %.3lfms\n", data_sizes[size_i],
                               double(avg) / CLOCKS_PER_SEC * 1000);
                }
                usleep(1000000);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
#endif

    // free device buffers
    for (int j = 0; j < DATA_NUM; j++) {
        CUDACHECK(cudaFree(sendbuff[j]));
    }
    CUDACHECK(cudaFree(recvbuff));

    // finalizing NCCL
    for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
            if (i == rank || j == rank)
                ncclCommDestroy(comms[i][j].nccl_comm);
        }
    }

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    return 0;
}