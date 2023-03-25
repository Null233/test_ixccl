#include "allreduce.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        printf("MPI_RING\n");

    for (int size_i = 0; size_i < data_sizes.size(); size_i++) {
        vector<vector<float>> datas;
        for (int j = 0; j < DATA_NUM; j++) {
            datas.push_back(vector<float>(data_sizes[size_i], rand()));
        }
        float *tmp = (float *)malloc(data_sizes[size_i] * sizeof(float));
        double avg = 0;
        PERF_COUNTER(mpiRing(&datas[i % DATA_NUM][0], tmp, data_sizes[size_i], rank, size,
                             MPI_FLOAT, MPI_COMM_WORLD));
        if (rank == 0)
            printf("DATA SIZE: %-10d takes %.3lfms\n", data_sizes[size_i],
                   double(avg) / CLOCKS_PER_SEC * 1000);
        free(tmp);
    }

    MPI_Finalize();
    return 0;
}
