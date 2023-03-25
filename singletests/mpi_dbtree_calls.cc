#include "allreduce.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Use ncclGetBtree and ncclGetDtree to get the parent and children ranks
    int parent0, parent1, nChildren0 = 0, nChildren1 = 0, children0[2], children1[2],
                          parentChildType0, parentChildType1;

    GetDtree(rank, size, &parent0, &children0[1], &children0[0], &parentChildType0, &parent1,
             &children1[1], &children1[0], &parentChildType1);

    int count = size;

    // vector<double> data(count, rank);
    double data = rank;

    // Use MPI_Send and MPI_Recv to implement a double binary tree algorithm
    int level = (int)log2(size); // The height of the tree
    for (int i = 0; i < level; i++) {
        // Use the first tree for reduce
        if (parent0 != -1) {
            // Send data to parent
            MPI_Send(&data, 1, MPI_DOUBLE, parent0, 0, MPI_COMM_WORLD);
        }
        // Receive data from children and add to sum
        for (int j = 0; j < 2; j++) {
            // if(children0[j]<0) continue;
            double recv;
            MPI_Recv(&recv, 1, MPI_DOUBLE, children0[j], 0, MPI_COMM_WORLD, IGNORE);
            data += recv;
        }
        // Use the second tree for broadcast
        if (parent1 != -1) {
            // Receive data from parent
            MPI_Recv(&data, 1, MPI_DOUBLE, parent1, 0, MPI_COMM_WORLD, IGNORE);
        }
        // Send data to children
        for (int j = 0; j < 2; j++) {
            if(children0[j]<0) continue;
            MPI_Send(&data, 1, MPI_DOUBLE, children1[j], 0, MPI_COMM_WORLD);
        }
    }

    printf("Rank: %d, The sum of all numbers is: %f\n", rank, data);
    printf("Rank: %d, The sum of all numbers is: ", rank);
    // for(int i = 0; i < data.size(); i++){
    //     printf("%.3f ", data[i]);
    // }
    // printf("\n");
    

    MPI_Finalize();
    return 0;
}