#include "allreduce.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        printf("MPI_DBTREE\n");

    vector<float> data(size, rank);

    mpiDBtreeReduction(&data[0], data.size(), rank, size, MPI_FLOAT, MPI_COMM_WORLD);

    printf("Rank: %3d\t", rank);
    for (int i = 0; i < size; i++) {
        printf("%5d ", int(data[i]));
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}