#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int BUF_SIZE = 1024;

int main(int argn, char **argv) {
    // initialize the MPI environment
    MPI_Init(NULL, NULL);

    // get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get the rank of the process
    int rank_in_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_in_world);

    // buffer to hold string
    char buf[BUF_SIZE];

    // pass 1 -> 2 -> ... -> world_size - 1 -> 0 -> 1 -> ...
    int rank_peer_prev = (rank_in_world + world_size - 1) % world_size;
    int rank_peer_next = (rank_in_world + 1) % world_size;

    // buf to store the temporary chars
    char tmp[BUF_SIZE];

    // flag to check whether the ring has already stopped
    int is_stop = 0;

    // 0 need to read from stdin
    int len_buf;
    if (rank_in_world == 0) {
        memset(buf, 0, sizeof(buf));
        printf("$>");
        fflush(NULL);
        fgets(buf, BUF_SIZE, stdin);
        len_buf = strlen(buf);
        if (buf[len_buf - 1] == '\n') {
            buf[len_buf - 1] = '\0';
        }
        printf("rank %d: get input> %s\n", rank_in_world, buf);
        fflush(NULL);

        // trigger the program
        // is_stop should be zero
        MPI_Send(&is_stop, 1, MPI_INT, rank_peer_next, 0, MPI_COMM_WORLD);

        // send out msg of string
        MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, rank_peer_next, 0, MPI_COMM_WORLD);
    }

    for (;;) {
        // try to recv msg from peer prev
        // printf("%d, %d, %d\n", rank_in_world, rank_peer_prev, is_stop);
        MPI_Recv(&is_stop, 1, MPI_INT, rank_peer_prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("%d, %d, %d\n", rank_in_world, rank_peer_prev, is_stop);

        // check if the ring has already stopped
        if (is_stop) {
            // send msg to next peer to tell it to stop
            MPI_Send(&is_stop, 1, MPI_INT, rank_peer_next, 0, MPI_COMM_WORLD);
            // exit main body
            break;
        }

        // if not is_stop, try to recv msg
        // printf("%d, %d, %s\n", rank_in_world, rank_peer_prev, buf);
        MPI_Recv(buf, BUF_SIZE, MPI_CHAR, rank_peer_prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // check if this is an empty string
        if (strlen(buf) == 0) {
            // then there will be no further msg to pass
            is_stop = 1;
            // tell next one to stop the ring
            MPI_Send(&is_stop, 1, MPI_INT, rank_peer_next, 0, MPI_COMM_WORLD);
            // block until the msg propagate back
            MPI_Recv(&is_stop, 1, MPI_INT, rank_peer_prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // print out your rank
            printf("rank %d write down: game over\n", rank_in_world);
            // exit main body
            break;
        }

        // there is still remainder
        // skip spaces
        int i;
        for (i = 0; i < BUF_SIZE && buf[i] == ' '; i++)
            ;
        // check is we have got no content
        if (buf[i] == '\0') {
            // run out of content
            is_stop = 1;
            // tell next one to stop the ring
            MPI_Send(&is_stop, 1, MPI_INT, rank_peer_next, 0, MPI_COMM_WORLD);
            // block until the msg propagate back
            MPI_Recv(&is_stop, 1, MPI_INT, rank_peer_prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // print out your rank
            printf("rank %d write down: game over\n", rank_in_world);
            // exit main body
            break;
        }

        // still got characters to play with
        int j;
        for (j = 0; i < BUF_SIZE && buf[i] != ' ' && buf[i] != '\0'; i++, j++) {
            tmp[j] = buf[i];
        }
        tmp[j] = '\0';
        printf("rank %d write down: %s\n", rank_in_world, tmp);

        // send out msg show it is not stopped yet
        // is_stop should be zero
        MPI_Send(&is_stop, 1, MPI_INT, rank_peer_next, 0, MPI_COMM_WORLD);

        // send out msg of string
        MPI_Send(buf + i, strlen(buf + i) + 1, MPI_CHAR, rank_peer_next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
