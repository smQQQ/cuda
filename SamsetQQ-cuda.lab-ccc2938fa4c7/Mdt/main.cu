
/*#ifndef __CUDACC__
#define __CUDACC__
#endif*/


#include "header.cuh"
#include <iomanip>
int main()
{
	ofstream file;
	file.open("test_result.txt");
	file << "size" << ' ' << "CPU" << ' ' << "GPU" << endl;
	for (int i = 100; i < 20000; i += 1000)
	{
		cout << i << endl;
		matrix m(i);
		vector<double> time=m.test();
		cout << "---------------" << endl;
		file << i << ' ' << std::fixed <<std::setprecision(4) << time[0] << ' ' << time[1] << endl;
	}
	file.close();
	/*matrix m(2000);
	m.DE_CPU();*/
    return 0;
}
