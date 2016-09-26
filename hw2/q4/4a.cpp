#include <omp.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
	if (argc != 4)
	{
		cerr << "Usage: " << argv[0] << " matrix1 matrix2 thread_number" << endl;
		return -1;
	}
	vector<vector<double>> matrix1;
	vector<vector<double>> matrix2;
	vector<vector<double>> result;
	string line;
	string dimension;
	int matrix1_row;
	int matrix1_column;
	int matrix2_column;
	int chunk;
	double time_elapsed;
	//parse file1 into matrix1
	ifstream file(argv[1]);/*("matrix1")*/;
	getline(file, dimension);
	istringstream di(dimension);
	string subs;
	di >> subs;
	matrix1_row = stoi(subs);
	di >> subs;
	matrix1_column = stoi(subs);
	while (getline(file, line)) {
		istringstream iss(line);
		matrix1.push_back(vector<double>((istream_iterator<double>(iss)), istream_iterator<double>()));
	}
	file.close();

	//parse file2 into matrix2
	ifstream file2(argv[2]);
	getline(file2, dimension);
	istringstream di2(dimension);
	di2 >> subs;
	di2 >> subs;
	matrix2_column = stoi(subs);
	while (getline(file2, line)) {
		istringstream iss2(line);
		matrix2.push_back(vector<double>((istream_iterator<double>(iss2)), istream_iterator<double>()));
	}
	file2.close();

	//set number of threads and chunk size
	int numOfThreads = stoi(argv[3]);//*(argv[3]);
	chunk = (matrix1_row < numOfThreads) ? 1 : matrix1_row / numOfThreads;
	//initialize result matrix to all 0s
	for (int i = 0; i < matrix1_row; i++) {
		vector<double> column;
		for (int j = 0; j < matrix2_column; j++) {
			column.push_back(0);
		}
		result.push_back(column);
	}
	//start counting time
	time_elapsed = omp_get_wtime();
	int tid;
	int i;
	int j;
	int k;
    omp_set_dynamic(0);
    omp_set_num_threads(numOfThreads);
#pragma omp parallel shared(matrix1, matrix2, result, chunk) private(tid, i, j, k)
{
tid = omp_get_thread_num();
#pragma omp for schedule(static, chunk)
	for (i = 0; i < matrix1_row; i++) {
	printf("Thread=%d did row=%d\n",tid,i);
		for (j = 0; j < matrix2_column; j++) {
			for (k = 0; k < matrix1_column; k++) {
				result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
}
	//determine running time after openMP calculations
	time_elapsed = omp_get_wtime() - time_elapsed;
	cout << "Execution time: " << time_elapsed << endl;
	//print results
	printf("******************************************************\n");
	printf("Result Matrix:\n");
	for (i = 0; i < matrix1_row; i++)
	{
		for (j = 0; j < matrix2_column; j++)
			printf("%6.2f   ", result[i][j]);
		printf("\n");
	}
	printf("******************************************************\n");
	printf("Done.\n\n");
}
