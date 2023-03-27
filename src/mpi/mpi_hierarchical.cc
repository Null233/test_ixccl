#include "allreduce.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size, localRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    getLocalRank(rank, size, &localRank);

    MPI_Comm COMM_LOCAL;
    MPI_Comm COMM_WORLD_MAIN;
    split_world(rank, localRank, &COMM_LOCAL, &COMM_WORLD_MAIN);

    if (rank == 0)
        printf("MPI_HIERARCHICAL_ALLREDUCE\n");
    if (rank == 0) {
#if LOCAL_REDUCE_ALGO == 1
        printf("LOCAL_MPI_ALLREDUCE\n");
#elif LOCAL_REDUCE_ALGO == 2
        printf("LOCAL_RING\n");
#elif LOCAL_REDUCE_ALGO == 3
        printf("LOCAL_BUTTERFLY\n");
#elif LOCAL_REDUCE_ALGO == 4
        printf("LOCAL_MPI_TREEREDUCTION\n");
#else
        printf("LOCAL_MPI_REDUCE\n");
#endif

#if LOCAL_BCAST_ALGO == 1
        printf("LOCAL_BTREE_BCAST\n");
#elif LOCAL_BCAST_ALGO == 2
        printf("LOCAL_PIPELINEDBTREE_BCAST\n");
#else
        printf("LOCAL_MPI_BCAST\n");
#endif
    }

    for (int size_i = 0; size_i < data_sizes.size(); size_i++) {
        vector<vector<float>> datas;
        for (int j = 0; j < DATA_NUM; j++) {
            datas.push_back(vector<float>(data_sizes[size_i], rand()));
        }
        float *tmp = (float *)malloc(data_sizes[size_i] * sizeof(float));
        double avg = 0;
        PERF_COUNTER(mpiHierarchicalAllreduce(&datas[i % DATA_NUM][0], tmp, data_sizes[size_i],
                                              rank, size, MPI_FLOAT, MPI_SUM, COMM_LOCAL,
                                              COMM_WORLD_MAIN));
        if (rank == 0)
            printf("DATA SIZE: %-10d takes %.3lfms\n", data_sizes[size_i], double(avg) * 1000);
        free(tmp);
    }

    MPI_Finalize();
    return 0;
}
