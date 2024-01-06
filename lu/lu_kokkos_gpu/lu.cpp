/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* lu.c: this file is part of PolyBench/C */
#include <Kokkos_Core.hpp>


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "lu.h"


#ifdef DOUBLE_PRECISION
using Scalar = double;
#else
using Scalar = float;
#endif


using ViewMatrixType =  Kokkos::View<Scalar**>;


/* Array initialization. */

static
void init_array (int n, ViewMatrixType::HostMirror A)
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j <= i; j++){
			A(i, j) = (DATA_TYPE)(-j % n) / n + 1;
			for (j = i+1; j < n; j++) {
				A(i, j) = 0;
			}
			A(i,i) = 1;
		}
	}

  /* Make the matrix positive semi-definite. */
  /* not necessary for LU, but using same code as cholesky */
  
  int r,s,t;
  ViewMatrixType B( "y", n, n );
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      B(r,s) = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	B(r,s) += A(r,t) * A(s,t);
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A(r,s) = B(r,s);
}




/* Main computational kernel. The whole function will be timed,
   including the call and return. */
   
static void kernel_lu(int n, ViewMatrixType A) {

	Kokkos::parallel_for( "kernel", n, KOKKOS_LAMBDA (const int i) {
		for (int j = 0; j <i; j++) {
			for (int k = 0; k < j; k++) {
				A(i, j) -= A(i, k) * A(k, j);
			}
			A(i, j) /= A(j, j);
		}

		for (int j = i; j < n; j++) {
			for (int k = 0; k < i; k++) {
				A(i, j) -= A(i,k) * A(k,j);
			}
		}
	});

}


