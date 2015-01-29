void cal_rho(cublasHandle_t handle, int NSPIN, int lgd, int bxyz, int LD_pool, int psir_ylm_pool_size, int size, int* block_iw, int* bsize, int* colidx, double* d_DM, double* d_psir_ylm, double* d_psir_DM, int* vindex, double** rho, double* psir_DM_pool);
void change_major(double *address, int elements_per_period, int period_number);
/* void fetch_submatrix(int elements_per_period, int periodicity, int period_number, double *source_address, double *target_address);
void place_submatrix(int elements_per_period, int periodicity, int period_number, double *source_address, double *target_address); */
