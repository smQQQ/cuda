#include "header.cuh"
#include <iostream>
using namespace std;

matrix::matrix(int size1)
{
	size = size1;
	mx = new double*[size1];
	for (int i = 0; i < size1; i++)
		mx[i] = new double[size1];

	msfill();
}

matrix::matrix(const matrix& m)
{
	size = m.size;
	mx = new double*[m.size];

	for (int i = 0; i < m.size; i++)
	{
		mx[i] = new double[size];
		memcpy(mx[i], m.mx[i],m.size*sizeof(double));
	}

}

matrix::~matrix()
{
	for (int i = 0; i < size; i++)
		delete[] mx[i];

	delete[] mx;
}


void matrix::swap(double** mx,int a, int b)
{
	double* buff = new double[size];
	for (int i = 0; i < size; i++)
		buff[i] = mx[b][i];

	for (int i = 0; i < size; i++)
		mx[b][i] = mx[a][i];

	for (int i = 0; i < size; i++)
		mx[a][i] = buff[i];

	delete[] buff;
}

int random1()
{
	//	srand(time(NULL));
	return (rand() % 5) + 1;
}

void matrix::msfill()
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			mx[i][j] = random1();
	}
}

void matrix::matrixprintCPU(double** arr)
{
	cout << "=====cpu_print======" << endl;
 	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
			cout << arr[i][j] << "\t";
		cout << endl;
	}
	cout << "===========" << endl;
}

void matrix::matrixprintGPU(double* deviceArray)
{
	double* temp = new double[size*size];
	cudaMemcpy((void*)temp, (void*)deviceArray, sizeof(double) * size * size, cudaMemcpyDeviceToHost);

	cout << "======gpu_print=====" << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			cout << temp[i*size + j] << "\t";
		cout << endl;
	}
	cout << "===========" << endl<<endl;
	delete[] temp;

}

int matrix::maxline(double ** mx,int j)
{
	double max = mx[j][j];
	int col = j;
	for (int lineidx=j; j < size; j++)
	{
		if (fabs(mx[j][lineidx]) > fabs(max))
		{
			max = mx[j][lineidx];
			col = j;
		}
	}
	return col;
}

bigValue matrix::DE_CPU()
{
	double** mcopy = new double*[size];
	for (int i = 0; i < size; i++)
	{
		mcopy[i] = new double[size];
		for (int j = 0; j < size; j++)
			mcopy[i][j] = mx[i][j];
	}

	double res = 1;
	double b;
	int swap_counter=0;
	for (int i = 0; i < size - 1; i++)
	{
		//cout << i << endl; 
		int l = maxline(mcopy,i);
		if (l != i)
		{
			swap(mcopy,l, i);
			swap_counter++;
		}
		
		for (int j = i + 1; j < size; j++)
		{
			b = mcopy[j][i] / mcopy[i][i];
			for (int i2 = 0; i2 < size; i2++)
				mcopy[j][i2] -= mcopy[i][i2]  * b;

		}
	}

	double mult = 1;
	int power = 0;
	

	for (int i = 0; i < size; i++)
	{
		mult *= mcopy[i][i];
		if (abs(mult) > 10 || abs(mult) < 1)
		{
			int valuePow = (floor(log10(abs(mult))));
			power += valuePow;
			mult = mult / pow(10, valuePow);
		}
	}

	if (swap_counter % 2 != 0)
		mult *= -1;

	bigValue bV;
	bV.mant = mult;
	bV.exp = power;

	for (int i = 0; i < size; i++)
		delete[] mcopy[i];

	delete[] mcopy;

	return bV;
}


__global__ void divCalc(double* divMatrix, double* div, int i, int msize)
{
	if((blockIdx.x* blockDim.x) +threadIdx.x < msize - i - 1)
		div[(blockIdx.x *blockDim.x) + threadIdx.x] = divMatrix[((i+1+ (blockIdx.x *blockDim.x) + threadIdx.x)*msize) + i] / divMatrix[i*msize+i];
}

__global__ void findmax(double* devMatrix, int *blockOut, int i, int msize)
{
	extern __shared__ double arr[];

	//  ажда€ нить записывает один из элементов р€да из массива в раздел€ему пам€ть
	if (((blockIdx.x*blockDim.x) + threadIdx.x) < msize - i)
	{
		arr[threadIdx.x] = devMatrix[((blockIdx.x*blockDim.x) + threadIdx.x + i)*msize + i];
		
	}
	__syncthreads();

	int blockLenght;
	int endofblock;

		// —пециальна€ перва€ итераци€
	if (((blockIdx.x*blockDim.x) + threadIdx.x) < msize - i)
	{
		

		if (blockIdx.x == (gridDim.x - 1) && ((msize - i) % blockDim.x) % 2 != 0 && gridDim.x > 1)
		{
			blockLenght = (msize - i) % blockDim.x;
			endofblock = ((msize - i) % blockDim.x) / 2;
		}
		else
		{
			blockLenght = blockDim.x;
			endofblock = blockDim.x / 2;
		}


		if (threadIdx.x < endofblock)
		{
			int left = endofblock - threadIdx.x - 1;
			int right = left + endofblock;
			if (abs(arr[left]) > abs(arr[right]))
				arr[right] = left;
			else
			{
				arr[left] = arr[right];
				arr[right] = right;
			}

			if (threadIdx.x == 0 && blockLenght % 2 != 0)
			{
				if (abs(arr[0]) < abs(arr[blockLenght - 1]))
				{
					arr[0] = arr[blockLenght - 1];
					arr[endofblock] = blockLenght - 1;
				}
			}
		}
	}

		__syncthreads();

		int lastEndOfBlock = endofblock;
		for (int endofblockC = endofblock / 2; endofblockC > 0; lastEndOfBlock = endofblockC, endofblockC /= 2)//переделать цикл / установить новую верхнюю границу в массиве
		{
			if (((blockIdx.x*blockDim.x) + threadIdx.x) < msize - i)
			{
				if (threadIdx.x < endofblockC)
				{
					int left_value = threadIdx.x;
					int right_value = left_value + endofblockC;

					int left_idx = left_value + endofblock;
					int right_idx = right_value + endofblock;


					if (abs(arr[left_value]) < abs(arr[right_value]))
					{
						arr[left_value] = arr[right_value];
						arr[left_idx] = arr[right_idx];
					}
				}

				if (threadIdx.x == 0 && lastEndOfBlock % 2 != 0)
				{

					if (abs(arr[0]) < abs(arr[endofblockC * 2]))
					{
						arr[0] = arr[endofblockC * 2];
						arr[blockLenght / 2] = arr[(blockLenght / 2) + (endofblockC * 2)];
					}
				}
			}
			__syncthreads();

		}

		if (threadIdx.x == 0)
		{
			blockOut[blockIdx.x] = ((arr[blockLenght / 2] + (blockDim.x*blockIdx.x)) + i)*msize + i;
		}
}

__global__ void swap(double* devMatrix, int i, int j, int size)
{
	if (((blockDim.x*blockIdx.x) + threadIdx.x) < size)
	{
		double buff = devMatrix[i*size + (blockDim.x*blockIdx.x + threadIdx.x)];
		devMatrix[i*size + (blockDim.x*blockIdx.x + threadIdx.x)] = devMatrix[j*size + (blockDim.x*blockIdx.x + threadIdx.x)];
		devMatrix[j*size + (blockDim.x*blockIdx.x + threadIdx.x)] = buff;
	}
}


__global__ void TriangleM(double* devMatrix, double* div,int i, int size)
{
	for (int j = i + 1; j < size; j++)
	{
		if (blockIdx.x*blockDim.x + threadIdx.x < size-i)
			devMatrix[j*size + (blockIdx.x*blockDim.x) + threadIdx.x + i] -= devMatrix[i*size + (blockIdx.x*blockDim.x) + threadIdx.x + i]  * div[j-i-1];
	
		__syncthreads();
	}

}

__global__ void MaxOfBlock(double* devMatrix, int* idx, int size)
{
	int lastMidle;
	for (int i = size / 2, lastMidle = i; i>0; lastMidle = i, i /= 2)
	{
		if (threadIdx.x < i)
		{
			if (abs(devMatrix[idx[threadIdx.x]]) < abs(devMatrix[idx[threadIdx.x + i]]))
				idx[threadIdx.x] = idx[threadIdx.x + i];

			if (threadIdx.x == 0 && lastMidle % 2 != 0)
			{
				if (abs(devMatrix[idx[0]]) < abs(devMatrix[idx[lastMidle - 1]]))
					idx[0] = idx[lastMidle - 1];
			}
		}
		__syncthreads();
	}
}

__global__ void getDiag(double* devMatrix, double* diag, int size)
{
	if (blockIdx.x*blockDim.x + threadIdx.x < size)
		diag[blockIdx.x*blockDim.x + threadIdx.x] = devMatrix[size*(blockIdx.x*blockDim.x + threadIdx.x) + blockIdx.x*blockDim.x + threadIdx.x];
}


kernelCfg cfgMaker(int size, cudaDeviceProp p)
{
	kernelCfg cfg;
	cfg.grid_size = ceil((double)(size) / p.maxThreadsPerBlock);
	cfg.block_size = ceil((double)(size) / cfg.grid_size);
	
	return cfg;
}



void MemInfo()
{
	size_t free_byte;

	size_t total_byte;

	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status) {

		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}

	double free_db = (double)free_byte;

	double total_db = (double)total_byte;

	double used_db = total_db - free_db;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

bigValue matrix::DE_GPU()
{
	//MemInfo();
	cudaError_t t;
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);

	int fullLineBlocks = ceil((float)size / p.maxThreadsPerBlock);
	int fullLineThreads = ceil((float)size / fullLineBlocks);
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'cudaMalloc'" << cudaGetErrorString(t) << endl;
	double* devMatrix;
	cudaMalloc((void**)&devMatrix, sizeof(double)*size*size);
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'cudaMalloc'" << cudaGetErrorString(t) << endl;

	double* ms1 = new double[size*size];
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			ms1[i*size + j] = mx[i][j];
	}
	cudaMemcpy((void*)devMatrix, (void*)ms1, sizeof(double) * size * size, cudaMemcpyHostToDevice);
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'cudaMemcpy'" << cudaGetErrorString(t) << endl;
	delete[] ms1;

	int sharedmem;

	int* devBlockResult;

	cudaMalloc((void**)&devBlockResult, sizeof(int)*fullLineBlocks);
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'cudaMalloc'" << cudaGetErrorString(t) << endl;


	int swap_counter = 0;

	for (int i = 0; i < size - 1; i++)
	{

		//cout << i << endl;

		kernelCfg launchCfg = cfgMaker((size - i), p);
		sharedmem = launchCfg.block_size * sizeof(double);

		findmax << < launchCfg.grid_size, launchCfg.block_size, sharedmem >> >(devMatrix, devBlockResult, i, size);
		cudaDeviceSynchronize();
		t = cudaGetLastError();
		if (t != cudaSuccess)
			cout << "cudaFailed in 'findmax'" << cudaGetErrorString(t) << endl;


		if (launchCfg.grid_size != 1)
		{
			MaxOfBlock << < 1, launchCfg.grid_size / 2 >> >(devMatrix, devBlockResult, launchCfg.grid_size);
			cudaDeviceSynchronize();
			t = cudaGetLastError();
			if (t != cudaSuccess)
				cout << "cudaFailed in 'MaxOfBlock'" << cudaGetErrorString(t) << endl;
		}


		int* ll = new int[launchCfg.grid_size];// использовать символ

		cudaMemcpy((void*)ll, (void*)devBlockResult, sizeof(int)*launchCfg.grid_size, cudaMemcpyDeviceToHost);
		t = cudaGetLastError();
		if (t != cudaSuccess)
			cout << "cudaFailed in 'memcpy'" << cudaGetErrorString(t) << endl;

		int l = ll[0];
		l = (l - i) / size;

		if (l != i)
		{
			::swap<<<fullLineBlocks, fullLineThreads >>>(devMatrix, i, l, size); 
			cudaDeviceSynchronize();
			t = cudaGetLastError();
			if (t != cudaSuccess)
				cout << "cudaFailed in 'swap'" << cudaGetErrorString(t) << endl;

			swap_counter++;
		}
		delete[] ll;

		double* div;
		cudaMalloc((void**)&div, sizeof(double)*(size-i-1));
		kernelCfg specCfg = cfgMaker(size - i - 1, p);
		divCalc<<<specCfg.grid_size, specCfg.block_size >>> (devMatrix,div, i, size);

		TriangleM << <launchCfg.grid_size, launchCfg.block_size >> >(devMatrix,div, i, size);
		cudaFree(div);
		cudaDeviceSynchronize();
		t = cudaGetLastError();
		if (t != cudaSuccess)
			cout << "cudaFailed in 'TriangleM'" << cudaGetErrorString(t) << endl;
	}

	cudaFree(devBlockResult);

	double* HostDiag = new double[size];
	double* DevDiag;

	cudaMalloc((void**)&DevDiag, sizeof(double)*size);
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'cudaMalloc'" << cudaGetErrorString(t) << endl;

	getDiag << <fullLineBlocks, fullLineThreads >> >(devMatrix, DevDiag, size);
	cudaDeviceSynchronize();
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'getDiag'" << cudaGetErrorString(t) << endl;

	cudaMemcpy((void*)HostDiag, (void*)DevDiag, sizeof(double) * size, cudaMemcpyDeviceToHost);
	t = cudaGetLastError();
	if (t != cudaSuccess)
		cout << "cudaFailed in 'cudaMemcpy'" << cudaGetErrorString(t) << endl;

	cudaFree(DevDiag);

	cudaFree(devMatrix);

	double mult = 1;
	int power = 0;

	for (int i = 0; i < size; i++)
	{
		mult *= HostDiag[i];
		if (abs(mult) > 10 || abs(mult) < 1)
		{
			int valuePow = (floor(log10(abs(mult))));
			power += valuePow;
			mult = mult / pow(10, valuePow);
		}
	}

	delete[] HostDiag;
	if (swap_counter % 2 != 0)
		mult *= -1;

	bigValue bV;
	bV.mant = mult;
	bV.exp = power;
	

	
	return bV;
}


typedef bigValue(matrix::*DE_func)();
vector<double> matrix::test()
{
	DE_func TestF[] = { &matrix::DE_CPU,&matrix::DE_GPU }; // с методами класс не получитс€
	vector<double> ret;
	bigValue bb[2];
	for (int fc = 0; fc < 2; fc++)
	{
		std::vector<double> time;
		for (int i = 0; i < repeat; i++)
		{
			cout << endl;
			if (fc == 0)
				cout << "CPU" << endl;
			else
				cout << "GPU" << endl;

			auto start_time = std::chrono::system_clock::now();
			bb[fc]=(this->*TestF[fc])();
			auto end_time = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds = end_time - start_time;
			cout << elapsed_seconds.count() << 's' << endl;
			time.push_back(elapsed_seconds.count());
		}
		/*if ((bb[0].mant != bb[1].mant) && (bb[0].exp != bb[1].exp))
			cout << "error" << endl;*/

		cout << bb[fc].mant << "*10^(" << bb[0].exp << ')' << endl;
		ret.push_back((double)std::accumulate(time.begin(), time.end(), 0.0) / (double)repeat);  // av time
	}
	return ret;
}
