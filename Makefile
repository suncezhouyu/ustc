################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:
#
# This source code is subject to NVIDIA ownership rights under U.S. and
# international Copyright laws.
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
# OR PERFORMANCE OF THIS SOURCE CODE.
#
# U.S. Government End Users.  This source code is a "commercial item" as
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
# and is provided to the U.S. Government only as a commercial end item.
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

all: build

build: cal_rho_cublas

main.o:main.cpp
	"/usr/local/cuda-6.0"/bin/nvcc -ccbin g++ -m64 -o main.o -c main.cpp

cal_rho_cublas.o:cal_rho_cublas.cpp
	"/usr/local/cuda-6.0"/bin/nvcc -ccbin g++ -I/usr/local/cuda-6.0/samples/common/inc  -I/usr/local/cuda-6.0/include -m64     -o cal_rho_cublas.o -c cal_rho_cublas.cpp

cal_rho_cublas: main.o cal_rho_cublas.o
	"/usr/local/cuda-6.0"/bin/nvcc -ccbin g++ -m64 -o cal_rho_cublas main.o cal_rho_cublas.o  -lcublas
	

run: build
	./cal_rho_cublas

clean:
	rm -f cal_rho_cublas cal_rho_cublas.o main.o *.txt

clobber: clean
