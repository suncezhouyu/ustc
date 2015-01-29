#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "cal_rho.h"


void cal_rho(cublasHandle_t handle, int NSPIN, int lgd, int bxyz, int LD_pool, int psir_ylm_pool_size, int size, int* block_iw, int* bsize, int* colidx, double* d_DM, double* d_psir_ylm, double* d_psir_DM, int* vindex, double** rho, double* psir_DM_pool)
{
	double r, *rhop, alpha, beta = 1.0;
	int iw1_lo, is, i, j, ia1, ia2, iw2_lo, ib, grid;
	
	for(is=0;is<NSPIN;++is)
	{
		for(i = 0; i<psir_ylm_pool_size; ++i) cudaMemcpy(d_psir_DM, psir_DM_pool, bxyz * LD_pool * sizeof(double), cudaMemcpyHostToDevice);//set d_psir_DM to zero
		for(ia1=0;ia1<size;++ia1) 
		{
			
			alpha = 1.0;
			iw1_lo = block_iw[ia1];

			cublasDsymm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, bxyz, bsize[ia1], &alpha, (d_DM + is * lgd * lgd + iw1_lo * lgd + iw1_lo), lgd, (d_psir_ylm + colidx[ia1] * bxyz), bxyz, &beta, (d_psir_DM + colidx[ia1] * bxyz), bxyz);

			alpha = 2.0;
			for(ia2=ia1+1; ia2<size; ++ia2)
			{
				iw2_lo = block_iw[ia2];
				cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bxyz, bsize[ia2], bsize[ia1], &alpha, (d_psir_ylm + colidx[ia1] * bxyz), bxyz, (d_DM + is * lgd *lgd + iw2_lo * lgd + iw1_lo), lgd, &beta, (d_psir_DM + colidx[ia2] * bxyz), bxyz);

			}   
   		}   
		
		rhop = rho[is];
		for(ib=0;ib<bxyz;++ib)
		{
		    cublasDdot(handle, colidx[size], d_psir_ylm + ib, bxyz, d_psir_DM + ib, bxyz, &r);
				
			grid = vindex[ib];
			rhop[grid] += r;
                        
		}
	}
        
}
