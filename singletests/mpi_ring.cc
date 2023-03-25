#include "allreduce.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        printf("MPI_RING\n");

    int count = 20;
    vector<float> data(count, rank);
    
    float *tmp = (float *)malloc(count * sizeof(float));
    mpiRing(&data[0], tmp, count, rank, size, MPI_FLOAT, MPI_COMM_WORLD);
    if(rank == 0){
        for(int i = 0; i < count; i++)
            printf("%3d ", int(data[i]));
    }

    MPI_Finalize();
    return 0;
}
