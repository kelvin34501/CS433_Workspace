#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int MAX_STRING = 100;

int main(int argn, char **argv) {
    // initialize the MPI environment
    MPI_Init(NULL, NULL);

    // get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // print off a hello world message
    // printf("Hello world from processor %s, rank %d out of %d processors\n",
    // processor_name, world_rank, world_size);

    // greet
    char greeting[MAX_STRING];
    if (world_rank != 0) {
        sprintf(greeting, "Greetings from process %d of %d!", world_rank, world_size);
        MPI_Send(greeting, strlen(greeting) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        printf("Greetings from process %d of %d!\n", world_rank, world_size);
        for (int q = 1; q < world_size; q++) {
            MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", greeting);
        }
    }

    // finalize the MPI env
    MPI_Finalize();

    return 0;
}