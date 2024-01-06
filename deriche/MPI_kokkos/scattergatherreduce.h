#ifndef SSR_H_INCLUDED
#define SSR_H_INCLUDED

#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <utility>
/* Include benchmark-specific header. */
#include "deriche.h"
//#include "../polybench_helpers.h"

typedef Kokkos::View<DATA_TYPE **> ViewMatrixType;

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

struct Subarray
{
    ViewMatrixType data; // pointer to data
    int h;               // height of array
    int w;               // width of array
};

// dt **allocarray(int n, int m);
// void printarr(dt **data, int n, int m, char *str);

struct Subarray scatter(ViewMatrixType data, int n, int m);
struct Subarray shuffle(struct Subarray in, int h, int w);
struct Subarray reduce(struct Subarray in, int h, int w);

struct Subarray scatter(ViewMatrixType data, int h, int w)
{
    struct Subarray sub;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = (h + (size - 1)) / size;

    // calculate size of local array
    int lower_h, upper_h;
    lower_h = rank * block_size;
    upper_h = MIN((rank + 1) * block_size, h);
    int range_h = upper_h - lower_h;
    int send_size = range_h * w;

    // dt **local = allocarray(range_h, w);
    ViewMatrixType local("local", range_h, w);

    /* communications parameters */
    const int sender = 0;
    int receiver = 1;
    const int ourtag = 2;

    if (rank == 0)
    {
        for (receiver = 1; receiver < size; receiver++)
        {
            // Send subarray to each thread 1,...,n-1
            int lower, upper;
            lower = receiver * block_size;
            upper = MIN((receiver + 1) * block_size, h);
            int range = upper - lower;
            int send_size = range * h;
            // float *local = malloc(sizeof(dt) * send_size);

            int starts[2] = {lower, 0};
            int subsizes[2] = {range, w};
            int bigsizes[2] = {h, w};
            auto send_view = Kokkos::subview(data, std::make_pair(lower, lower + range), std::make_pair(0, w));
            void *send_ptr = send_view.data();

            /*MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
                                    MPI_ORDER_C, MPI_FLOAT, &mysubarray);
            MPI_Type_commit(&mysubarray);

            MPI_Send(&(data[0][0]), 1, mysubarray, receiver, ourtag, MPI_COMM_WORLD);
            MPI_Type_free(&mysubarray); */
            // printf("Rank %d sending data to rank %d \n", rank, receiver);
            MPI_Send(send_ptr, send_view.size(), MPI_FLOAT, receiver, ourtag, MPI_COMM_WORLD);
        }

        parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({lower_h, 0}, {upper_h, w}),
            KOKKOS_LAMBDA(const int i, const int j) {
                local(i, j) = data(i, j);
            });
    }
    else
    {
        void *recv_ptr = local.data();
        // printf("Rank %d waiting for data from master \n", rank);
        MPI_Recv(recv_ptr, local.size(), MPI_FLOAT, sender, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    sub.data = local;
    sub.h = range_h;
    sub.w = w;
    return sub;
}

struct Subarray shuffle(struct Subarray in, int h, int w)
{

    /* communications parameters */
    const int sender = 0;
    int receiver = 1;
    const int ourtag = 3;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype mysubarray;

    int block_size_w = (w + (size - 1)) / size;
    int block_size_h = (h + (size - 1)) / size;
    // calculate width of end array
    int lower_w, upper_w;
    lower_w = rank * block_size_w;
    upper_w = MIN((rank + 1) * block_size_w, w);
    int range_w = upper_w - lower_w;

    for (int i = 0; i < size; i++)
    {
        if (i == rank)
        {
            continue;
        }
        // calculate width of subarray to send
        int lower, upper;
        lower = i * block_size_w;
        upper = MIN((i + 1) * block_size_w, in.w);
        int range = upper - lower;

        int starts[2] = {0, lower};
        int subsizes[2] = {in.h, range};
        int bigsizes[2] = {in.h, in.w};

        auto send_view = Kokkos::subview(in.data, std::make_pair(0, in.h), std::make_pair(lower, lower + range));
        void *send_ptr = send_view.data();

        // MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &mysubarray);
        // MPI_Type_commit(&mysubarray);

        MPI_Request request;
        // printf("Shuffle: Rank %d sending data to rank %d \n", rank, i);
        MPI_Isend(send_ptr, send_view.size(), MPI_FLOAT, i, ourtag, MPI_COMM_WORLD, &request);
        // MPI_Type_free(&mysubarray);
    }

    struct Subarray out;
    ViewMatrixType o("o", h, range_w);
    out.data = o;
    out.h = h;
    out.w = range_w;

    for (int receiver = 0; receiver < size; receiver++)
    {

        // Caluculate height of incomming array: range * range_w
        int lower, upper;
        lower = receiver * block_size_h;
        upper = MIN((receiver + 1) * block_size_h, h);
        int range = upper - lower;
        int send_size = range * range_w;

        if (receiver == rank)
        {
            parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {range, range_w}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    out.data(lower + i, j) = in.data(i, lower_w + j);
                });
            continue;
        }

        ViewMatrixType rec_view("recv", range, range_w);
        void *recv_ptr = rec_view.data();
        // dt **rec_buffer = allocarray(range, range_w);
        // printf("Shuffle: Rank %d waiting for data from %d \n", rank, receiver);
        MPI_Recv(recv_ptr, send_size, MPI_FLOAT, receiver, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // printf("Shuffle: Rank %d received data from %d \n", rank, receiver);
        parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {range, range_w}),
            KOKKOS_LAMBDA(const int _i, const int _j) {
                out.data(lower + _i, _j) = rec_view(_i, _j);
            });
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return out;
}

struct Subarray reduce(struct Subarray in, int h, int w)
{

    /* communications parameters */
    const int sender = 0;
    int receiver = 1;
    const int ourtag = 4;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype mysubarray;

    int block_size_w = (w + (size - 1)) / size;
    int block_size_h = (h + (size - 1)) / size;
    // calculate width of end array
    int lower_w, upper_w;
    lower_w = rank * block_size_w;
    upper_w = MIN((rank + 1) * block_size_w, w);
    int range_w = upper_w - lower_w;

    if (rank != 0)
    {
        // Send data to thread 0
        auto send_view = Kokkos::subview(in.data, std::make_pair(0, in.h), std::make_pair(0, in.w));
        void *send_ptr = send_view.data();
        // printf("Reduce: Rank %d sending data to rank %d \n", rank, receiver);
        MPI_Request request;
        MPI_Isend(send_ptr, send_view.size(), MPI_FLOAT, 0, ourtag, MPI_COMM_WORLD, &request);
        // MPI_Send(&(in.data(0,0)), in.h*in.w, MPI_FLOAT, 0, ourtag, MPI_COMM_WORLD);
        MPI_Barrier( MPI_COMM_WORLD );
    }
    else
    {
        struct Subarray out;
        ViewMatrixType t("t", h, w);
        out.data = t;
        out.h = h;
        out.w = w;

        parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, range_w}),
            KOKKOS_LAMBDA(const int _i, const int _j) {
                out.data(_i, lower_w + _j) = in.data(_i, _j);
            });

        for (int receiver = 1; receiver < size; receiver++)
        {

            // Caluculate width of incomming array: range * range_w
            int lower, upper;
            lower = receiver * block_size_w;
            upper = MIN((receiver + 1) * block_size_h, w);
            int range = upper - lower;
            int send_size = range * h;

            // dt **rec_buffer = allocarray(h, range);
            ViewMatrixType rec_view("recv", h, range);
            void *recv_ptr = rec_view.data();
            // printf("Reduce: Rank %d waiting for data from %d \n", rank, receiver);
            MPI_Recv(recv_ptr, send_size, MPI_FLOAT, receiver, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h, range}),
                KOKKOS_LAMBDA(const int _i, const int _j) {
                    out.data(_i, lower + _j) = rec_view(_i, _j);
                });
        }
        MPI_Barrier( MPI_COMM_WORLD );
        return out;
    }

    struct Subarray empty;
    return empty;
}

#endif