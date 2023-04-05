#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int depth = 0;
    int temp = size;
    while (temp >>= 1) depth++;

    int data = rank + 1; // 每个进程的初始数据为进程ID加1，以便观察输出结果

    for (int i = 0; i < depth; i++) {
        int parent = (rank - 1) / 2;
        int left_child = 2 * rank + 1;
        int right_child = 2 * rank + 2;

        if (rank != 0 && i == 0) {
            MPI_Send(&data, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
        } else if (left_child < size) {
            int left_data;
            MPI_Recv(&left_data, 1, MPI_INT, left_child, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            data += left_data;
            if (right_child < size) {
                int right_data;
                MPI_Recv(&right_data, 1, MPI_INT, right_child, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                data += right_data;
            }
            if (rank != 0) {
                MPI_Send(&data, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
            }
        }
    }

    for (int i = depth - 1; i >= 0; i--) {
        int parent = (rank - 1) / 2;
        int left_child = 2 * rank + 1;
        int right_child = 2 * rank + 2;

        if (rank != 0 && i == 0) {
            MPI_Recv(&data, 1, MPI_INT, parent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (left_child < size) {
            MPI_Send(&data, 1, MPI_INT, left_child, 0, MPI_COMM_WORLD);
            if (right_child < size) {
                MPI_Send(&data, 1, MPI_INT, right_child, 0, MPI_COMM_WORLD);
            }
            if (rank != 0) {
                MPI_Recv(&data, 1, MPI_INT, parent, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    std::cout << "Process " << rank << " has data: " << data << std::endl;

    MPI_Finalize();
    return 0;
}
