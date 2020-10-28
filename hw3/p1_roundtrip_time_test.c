#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int MAX_MSG_LEN = 100;

int main(int argn, char **argv) {
    // initialize the MPI environment
    MPI_Init(NULL, NULL);

    // get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get the rank of the process
    int rank_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);

    // check world size
    if (world_size < 2) {
        if (rank_world == 0) {
            fprintf(stderr, "world size is less than 2, got %d\n", world_size);
        }
        MPI_Finalize();
        return 0;
    }

    // var for storing start & end time
    double start = 0.0, end = 0.0;

    // message array
    char msg[2][MAX_MSG_LEN];

    if (rank_world == 0) {
        // get msg
        sprintf(msg[0], "pack: %d =>", rank_world);

        // time start
        start = MPI_Wtime();

        // send message to rank 1 (specified in hw)
        MPI_Send(msg[0], strlen(msg[0]) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        // recv ack
        // reuse msg
        MPI_Recv(msg[0], MAX_MSG_LEN, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // time stop
        end = MPI_Wtime();

        // get some feedback
        printf("ack: {{%s}}\n", msg[0]);
        printf("elapsed time: %f\n", end - start);
    } else if (rank_world == 1) {
        // recv msg and send back ack
        MPI_Recv(msg[0], MAX_MSG_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // construct msg
        sprintf(msg[1], "ack <= %d, msg: %s", rank_world, msg[0]);

        // send back
        MPI_Send(msg[1], strlen(msg[1]) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        // according to hw specification
        // do nothing
    }

    // finalize
    MPI_Finalize();

    return 0;
}