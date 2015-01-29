#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include "cal_rho.h"


void change_major(double *address, int elements_per_period, int period_number)
{
    int p, q, l;
    double temp[elements_per_period * period_number];
    for(p=0;p<period_number;p++)
    {
             for(q=0;q<elements_per_period;q++)
                 temp[q * period_number + p] = address[p * elements_per_period + q];  
    }
    for(l=0;l<elements_per_period * period_number;l++)
        address[l] = temp[l];
} 

using namespace std;

int main(int argc, char** argv)
{
	int NSPIN=0;
	int nbx=0, nby=0, nbz=0;
	int bxyz=0;
	int lgd=0;
	int LD_pool=0;
	int max_size=0;
        int i,j,k,is;
	int psir_ylm_pool_size, all_grid;
	
	int* vindex;
	
	double** DM_pool;
	double*** DM;
	
	int size;
	int* colidx;
	int* block_iw;
	int* bsize;
	
	double* psir_ylm_pool;
	double** psir_ylm;
	
	double* psir_DM_pool;
		
	double** rho;
	double sum;

	double *d_DM, *d_psir_ylm, *d_psir_DM;
	cublasHandle_t handle;
	
	ifstream fPARAMETER("PARAMETER.dat");
	fPARAMETER>>NSPIN;
	fPARAMETER>>nbx;
	fPARAMETER>>nby;
	fPARAMETER>>nbz;
	fPARAMETER>>bxyz;
	fPARAMETER>>lgd;
	fPARAMETER>>LD_pool;
	fPARAMETER>>max_size;
	fPARAMETER.close();
	
	vindex=new int[bxyz];
	
	DM_pool=new double*[NSPIN];
	DM=new double**[NSPIN];
	for(is=0; is<NSPIN; ++is)
	{
		DM_pool[is]=new double[lgd*lgd];
		DM[is]=new double*[lgd];
		for(i=0; i<lgd; ++i)
		{
			DM[is][i]=&DM_pool[is][i*lgd];
		}
	}
	cudaMalloc((void **)&d_DM, NSPIN * lgd * lgd * sizeof(double));
	
	colidx=new int[max_size+1];
	block_iw=new int[max_size];
	bsize=new int[max_size];
	
	psir_ylm_pool=new double[bxyz*LD_pool];
	psir_ylm=new double*[bxyz];
	cudaMalloc((void **)&d_psir_ylm, bxyz * LD_pool * sizeof(double));
	
	psir_DM_pool=new double[bxyz*LD_pool];
	
	cudaMalloc((void **)&d_psir_DM, bxyz * LD_pool * sizeof(double));
	
	for(i=0; i<bxyz; ++i)
	{
		psir_ylm[i]=&psir_ylm_pool[i*LD_pool];
		
		for(j=0;j<LD_pool;j++) psir_DM_pool[i * LD_pool + j] = 0.0;
	}
	
	all_grid=nbx*nby*nbz*bxyz;
	rho=new double*[NSPIN];
	for(is=0; is<NSPIN; ++is)
		rho[is]=new double[all_grid];
	
	ifstream fDM("DM.dat", ios::binary);
	for(is=0; is<NSPIN; ++is)
	{
		for(i=0;i<lgd; ++i)
		{
			fDM.read(reinterpret_cast<char*>(&DM[is][i][i]), sizeof(double)*(lgd-i));//export to device; read the upper triangle
		}
		change_major(&DM[is][0][0], lgd, lgd);
		cudaMemcpy(d_DM + is * lgd * lgd, &DM[is][0][0], lgd * lgd * sizeof(double), cudaMemcpyHostToDevice);
	}
	fDM.close();
	
	ifstream fBAND("BAND.dat", ios::binary);
	ifstream fPSIR_YLM("PSIR_YLM.dat", ios::binary);
	ifstream fVINDEX("VINDEX.dat", ios::binary);
	
	psir_ylm_pool_size=bxyz*LD_pool;

	cublasCreate(&handle);//need cublas

   	for(i=0; i<nbx; ++i)
	{
		for(j=0; j<nby; ++j)
		{
			for(k=0; k<nbz; ++k)
			{
      
				fBAND.read(reinterpret_cast<char*>(&size), sizeof(int));
				if(size<=0) continue;      
				fBAND.read(reinterpret_cast<char*>(colidx), sizeof(int)*(max_size+1));
				fBAND.read(reinterpret_cast<char*>(block_iw), sizeof(int)*(max_size));
 				fBAND.read(reinterpret_cast<char*>(bsize), sizeof(int)*(max_size));

				fPSIR_YLM.read(reinterpret_cast<char*>(psir_ylm_pool), sizeof(double)*(psir_ylm_pool_size));//export to device
				change_major(psir_ylm_pool, LD_pool, bxyz);
				cudaMemcpy(d_psir_ylm, psir_ylm_pool, bxyz * LD_pool * sizeof(double), cudaMemcpyHostToDevice);

				fVINDEX.read(reinterpret_cast<char*>(vindex), sizeof(int)*(bxyz));

				//cudaMemcpy(d_psir_DM, psir_DM_pool, bxyz * LD_pool * sizeof(double), cudaMemcpyHostToDevice);//set d_psir_DM to zero
				cal_rho(handle, NSPIN, lgd, bxyz, LD_pool, psir_ylm_pool_size, size, block_iw, bsize, colidx, d_DM, d_psir_ylm, d_psir_DM, vindex, rho, psir_DM_pool);
   
			}
		}
	}

      	cublasDestroy(handle);//don't need cublas anymore
	cudaFree(d_DM);
	cudaFree(d_psir_DM);
	cudaFree(d_psir_ylm);

	fBAND.close();
	fPSIR_YLM.close();
	fVINDEX.close();
	
	sum=0; // sum of charge
	for(is=0; is<NSPIN; ++is)
		for(i=0; i<all_grid; ++i)
			sum+=rho[is][i];
	cout<<"total charge in the system is:"<<sum<<endl;
        
	delete [] vindex;
	delete [] colidx;
	delete [] block_iw;
	delete [] bsize;
	delete [] psir_DM_pool;
	delete [] psir_ylm_pool;
        
	for(is=0; is<NSPIN; ++is)
	{
		delete [] DM_pool[is];
		delete [] DM[is];
	}
	delete [] DM_pool;
	delete [] DM;
	
	delete [] psir_ylm;
	delete [] rho;
	return 0;
}
