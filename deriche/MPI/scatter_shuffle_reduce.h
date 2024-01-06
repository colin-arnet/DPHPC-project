#include <mpi.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

struct Subarray {
    dt ** data; // pointer to data
    int h;      // height of array
    int w;      // width of array
};

dt **allocarray(int n, int m);
void freearray(dt **a);
void printarr(dt **data, int n, int m, char *str);

struct Subarray scatter (dt **data, int n, int m);
struct Subarray shuffle (struct Subarray in, int h, int w);
struct Subarray reduce (struct Subarray in, int h, int w);

struct Subarray scatter (dt **data, int h, int w) {
    struct Subarray sub;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_size = (h + (size - 1))/ size;

    // calculate size of local array
    int lower_h, upper_h;
    lower_h = rank * block_size;
    upper_h = MIN ((rank + 1) * block_size, h);
    int range_h = upper_h - lower_h;
    int send_size = range_h * w;


    sub.data = allocarray(range_h, w);

    /* communications parameters */
    const int sender  =0;
    int receiver=1;
    const int ourtag  =1;

    MPI_Datatype mysubarray;

    if (rank == 0) {
        for(receiver = 1; receiver < size; receiver++) {
            // Send subarray to each thread 1,...,n-1
            int lower, upper;
            lower = receiver * block_size;
            upper = MIN ((receiver + 1) * block_size, h);
            int range = upper - lower;
            int send_size = range * h;
            //float *local = malloc(sizeof(dt) * send_size);

            int starts[2] = {lower,0};
            int subsizes[2]  = {range, w};
            int bigsizes[2]  = {h, w};
            MPI_Type_create_subarray(2, bigsizes, subsizes, starts,
                                    MPI_ORDER_C, MPI_FLOAT, &mysubarray);
            MPI_Type_commit(&mysubarray);

            MPI_Send(&(data[0][0]), 1, mysubarray, receiver, ourtag, MPI_COMM_WORLD);
            MPI_Type_free(&mysubarray);
        }
        for(int i = lower_h; i < upper_h; i++) {
            for(int j = 0; j < w; j++) {
                sub.data[i][j] = data[i][j];
            }
        }
    } else {
        MPI_Recv(&(sub.data[0][0]), send_size, MPI_FLOAT, sender, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    sub.h = range_h;
    sub.w = w;
    return sub;
}

struct Subarray shuffle (struct Subarray in, int h, int w) {

    /* communications parameters */
    const int sender  =0;
    int receiver=1;
    const int ourtag  =2;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Datatype mysubarray;

    int block_size_w = (w + (size - 1))/ size;
    int block_size_h = (h + (size - 1))/ size;
    // calculate width of end array
    int lower_w, upper_w;
    lower_w = rank * block_size_w;
    upper_w = MIN ((rank + 1) * block_size_w, w);
    int range_w = upper_w - lower_w;


    for  (int i = 0; i < size; i++) {
        if (i == rank) {
            continue;
        }
        //calculate width of subarray to send
        int lower, upper;
        lower = i * block_size_w;
        upper = MIN ((i + 1) * block_size_w, in.w);
        int range = upper - lower;

        int starts[2] = {0, lower};
        int subsizes[2]  = {in.h, range};
        int bigsizes[2]  = {in.h, in.w};

        MPI_Type_create_subarray(2, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &mysubarray);
        MPI_Type_commit(&mysubarray);
        
        MPI_Request request;
        MPI_Isend(&(in.data[0][0]), 1, mysubarray, i, ourtag, MPI_COMM_WORLD, &request);
        //printf("Sent array to  %d \n", i);
        MPI_Type_free(&mysubarray);
    }

    struct Subarray out;
    out.data = allocarray(h, range_w);
    out.h = h;
    out.w = range_w;

    for(int receiver = 0; receiver < size; receiver++) {

        // Caluculate height of incomming array: range * range_w
        int lower, upper;
        lower = receiver * block_size_h;
        upper = MIN ((receiver + 1) * block_size_h, h);
        int range = upper - lower;
        int send_size = range * range_w;

        if (receiver == rank) {
            for(int i = 0; i < range; i++) {
                for(int j = 0; j < range_w; j++) {
                    out.data[lower + i][j] = in.data[i][lower_w + j];
                }
            }
            continue;
        } 

        dt **rec_buffer = allocarray(range, range_w);
        //printf("Waiting for array from %d \n", receiver);
        MPI_Recv(&(rec_buffer[0][0]), send_size, MPI_FLOAT, receiver, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("received array from %d \n", receiver);
        for(int i_ = 0; i_ < range; i_++) {
            for(int j_ = 0; j_ < range_w; j_++) {
                out.data[lower + i_][j_] = rec_buffer[i_][j_];
            }
        }
        freearray(rec_buffer);

    }
    MPI_Barrier( MPI_COMM_WORLD );
    freearray(in.data);
    return out;
}

struct Subarray reduce (struct Subarray in, int h, int w) {

    /* communications parameters */
    const int sender  =0;
    int receiver=1;
    const int ourtag  =3;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Datatype mysubarray;

    int block_size_w = (w + (size - 1))/ size;
    int block_size_h = (h + (size - 1))/ size;
    // calculate width of end array
    int lower_w, upper_w;
    lower_w = rank * block_size_w;
    upper_w = MIN ((rank + 1) * block_size_w, w);
    int range_w = upper_w - lower_w;

    if (rank != 0) {
        // Send data to thread 0
        MPI_Request request;        
        MPI_Isend(&(in.data[0][0]), in.h*in.w, MPI_FLOAT, 0, ourtag, MPI_COMM_WORLD, &request);
        MPI_Barrier( MPI_COMM_WORLD );
        freearray(in.data);
    } else {
        struct Subarray out;
        out.data = allocarray(h, w);
        out.h = h;
        out.w = w;

        for(int i_ = 0; i_ < h; i_++) {
            for(int j_ = 0; j_ < range_w; j_++) {
                out.data[i_][lower_w+j_] = in.data[i_][j_];
            }
        }

        for(int receiver = 1; receiver < size; receiver++) {

            // Caluculate width of incomming array: range * range_w
            int lower, upper;
            lower = receiver * block_size_h;
            upper = MIN ((receiver + 1) * block_size_h, w);
            int range = upper - lower;
            int send_size = range * h;

            dt **rec_buffer = allocarray(h, range);
            //printf("Upper %d lower %d \n", upper, lower);
            //printf("Expecting %d by %d \n", range, h);

            MPI_Recv(&(rec_buffer[0][0]), send_size, MPI_FLOAT, receiver, ourtag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int i_ = 0; i_ < h; i_++) {
                for(int j_ = 0; j_ < range; j_++) {
                    out.data[i_][lower+j_] = rec_buffer[i_][j_];
                }
            }
            freearray(rec_buffer);
        }
        MPI_Barrier( MPI_COMM_WORLD );
        freearray(in.data);
        return out;
    }
    struct Subarray empty;
    return empty;

}

dt **allocarray(int n, int m) {
    dt *data = malloc(n*m*sizeof(dt));
    dt **arr = malloc(n*sizeof(dt *));
    for (int i=0; i<n; i++)
        arr[i] = &(data[i*m]);

    return arr;
}

void freearray(dt **a) {
    free (a[0]);
    free (a);
}

void printarr(dt **data, int n, int m, char *str) {    
    printf("-- %s --\n", str);
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            printf("%3f ", data[i][j]);
        }
        printf("\n");
    }
}