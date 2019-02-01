//запихнуть инфу в дебаг массив структур
#include <stdio.h>
#include <windows.h>
#include <math.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <vector>
#include <numeric>
#include <chrono>

using namespace std;

struct bigValue
{
	double mant;
	int exp;
};

struct kernelCfg
{
	int grid_size;
	int block_size;
};

class matrix
{
public:
	double **mx = NULL;
	int size;
	matrix(int);
	matrix(const matrix&);
	bigValue DE_CPU();
	bigValue DE_GPU();
	vector<double> test();
	~matrix();

private:
	int repeat = 1;
	void swap(double**, int, int);
	void msfill();
	int maxline(double **, int);
	void matrixprintCPU(double**);
	void matrixprintGPU(double*);
	
};






