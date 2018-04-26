/*
 * try.c
 *
 *  Created on: Mar 30, 2018
 *      Author: Rasika, Satish
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include "cluster.h"
#include "kmeans.h"

#define SIZE 4096
#define MAX_CLUSTER_SIZE 20
#define MAX_CLUSTER_COUNT 400
int num_cols = 0;
int clusterCounter = 0;
int scoreCounter = 0;
double* testScore;
int current_cluster=0;
double** gaussian_centers;
int gc_counter = 0;

//define cluster as a matrix
typedef struct cluster {
	double **data;
	int nrows;
	double testScore;
	double *center;
} cluster;

cluster *cluster_array;

void tokenize(char *line, double *data) {
	char *token;
	int col = 0;

	token = strtok(line, ",");
	while(token) {
		//printf("token: %s\n", token);
		data[col] = strtod(token, NULL);
		//printf("double: %lf\n", data[col]);
		token = strtok(NULL, ",");
		col++;
	}
	return;
}

void findDiffBetweenCenters(double* v, double** centers){
	//double v[16];
	for(int j=0;j<num_cols;j++){
		v[j] = centers[1][j] - centers[0][j];
	}
}

double dot_product(double v[], double u[], int n)
{
    double result = 0.0;
    for (int i = 0; i < n; i++)
        result += v[i]*u[i];
    return result;
}

void transformCluster(double* x, int nrows, double* v, double vectorProduct, double** cluster)
{
	for(int i=0;i<nrows;i++)
	{
		x[i] = dot_product(cluster[i],v,num_cols)/vectorProduct;
	}
}

void preprocess(double* x, int n, double* z)
{
	double sum=0.0, sum1=0.0, average=0.0, variance=0.0, std_deviation=0.0;
	for (int i = 0; i < n; i++)
	{
		sum = sum + x[i];
	}
	average = sum / (float)n;
	/*  Compute  variance  and standard deviation  */
	for (int i = 0; i < n; i++)
	{
		sum1 = sum1 + pow((x[i] - average), 2);
	}
	variance = sum1 / (float)n;
	std_deviation = sqrt(variance);

	//printf("Mean: %f & SD: %f", average, std_deviation);

	for(int i=0; i<n; i++){
		z[i] = (x[i]-average) / std_deviation;
	}
	/*printf("Transformed Vector:");
	for(int i=0; i<n; i++) {
		printf("Row %d: %f", i, z[i]);
		printf("\n");
	}*/
}

int compare(const void * a, const void * b)
{
    return ( *(double*)a - *(double*)b );
}

void swap(double *xp, double *yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void selectionSort(double arr[], int n)
{
    int i, j, min_idx;

    // One by one move boundary of unsorted subarray
    for (i = 0; i < n-1; i++)
    {
        // Find the minimum element in unsorted array
        min_idx = i;
        for (j = i+1; j < n; j++)
          if (arr[j] < arr[min_idx])
            min_idx = j;

        // Swap the found minimum element with the first element
        swap(&arr[min_idx], &arr[i]);
    }
}

// prob distribution function
void pdf(double* z, int n){
	for (int i=0;i<n;i++){
		//z[i] = exp(-1*(pow(z[i],2)/2))/pow(2*M_PI,0.5);
		z[i] = 0.5 * erfc(-0.707106781186547524 * z[i]);
		if(z[i] == 0.0) z[i] = 0.000001;
		if(z[i] == 1.0) z[i] = 0.999999;
	}
}

void andersondarling(double* z, int n){
	selectionSort(z, n);
	pdf(z, n);

	/*for (int j = 0; j < n; j++){
		printf("Row %d: \t%7.3f", j, z[j]);
		printf("\n");
	}*/

	double A = 0.0;
	for (int i = 0; i < n; i++) {
		//A = A + (2*i+1) * (log(z[i]) + log(1.0-z[n-i-1]));
		A -= (2*i+1) * (log(z[i]) + log(1-z[n-i-1]));
	}
	A = (A / n) - n;
	A *= (1 + 0.75/n - 2.25/(n*n));
	cluster_array[current_cluster].testScore = A;
	//printf(" Score : %f", A);
}

void printClusters(double** cluster, int n){
	for (int i = 0; i < n; i++)
	  {
		printf("Row %2d:", i);
		for (int j = 0; j < 2; j++)
			printf("\t%7.3f", cluster[i][j]);
		printf("\n");
	  }
}

void deque_cluster(int pop_index){
	cluster *temp = (cluster *)malloc(MAX_CLUSTER_COUNT * sizeof(cluster));
	if(clusterCounter>1){
		for(int i=0, j=0; i<clusterCounter;i++){
			if(i!=pop_index){
				temp[j] = cluster_array[i];
				j++;
			}
		}
	}
	clusterCounter=clusterCounter-1;
	free(cluster_array);
	cluster_array = temp;
}

void create_csv(char *filename, int rows,int cols){
	FILE *fp;
	int i =0;
	fp=fopen(filename,"w+");
	double **data;
	for(i=0; i<clusterCounter; i++){
		data = (double **)malloc(sizeof(double *) * cluster_array[i].nrows);
		data = cluster_array[i].data;
		for(int j=0;j<cluster_array[i].nrows;j++){
			for(int k=0;k<cols;k++){
				fprintf(fp,"%f ,",data[j][k]);
			}
			fprintf(fp,"%d\n",i+1);
		}
	}
	fclose(fp);
}

void performKmean(int nrows, int ncols, double** data, int** mask){
	int i, j, itr;
	const int nclusters = 2;
	const char dist = 'e';
	cluster_array = (cluster *)malloc(MAX_CLUSTER_COUNT * sizeof(cluster));
	gaussian_centers = (double**) malloc(MAX_CLUSTER_COUNT * sizeof(double*));

	cluster_array[0].data = data;
	cluster_array[0].nrows = nrows;
	clusterCounter++;

	do{
		data = cluster_array[current_cluster].data;
		nrows = cluster_array[current_cluster].nrows;

		// initialize variables
		double weights[ncols];
		int* clusterid = malloc(nrows*sizeof(int));
		//define initial centroid
		double  **centers;
		centers    = (double**) malloc(nclusters * sizeof(double*));
		centers[0] = (double*)  malloc(nclusters * ncols * sizeof(double));
		for (i=1; i<nclusters; i++)
			centers[i] = centers[i-1] + ncols;

		// to find the initial centers that are farthest
		double ** distmatrix = distancematrix(nrows, ncols, data, mask, weights, dist, 0);
		int ip=1, jp=0;
		find_farthest_pair(nrows, distmatrix, &ip, &jp);
		for (j=0; j<ncols; j++)
			centers[0][j] = data[ip][j];
		for (j=0; j<ncols; j++)
			centers[1][j] = data[jp][j];

		// perform kmeans
		double** cdata = seq_kmeans(data, ncols,nrows, nclusters, centers, 0.001, clusterid, &itr);

		// to count the cluster assignemnt
		int cluster1rows=0, cluster2rows=0;

		for (i = 0; i < nrows; i++){
			//printf ("Gene %d: cluster %d\n", i, clusterid[i]);
			switch(clusterid[i]){
			case 0:
				cluster1rows++;
				break;
			case 1:
				cluster2rows++;
				break;
			}
		}
		//printf("%d",sizeof(clusterid));
		printf ("Cluster 1: %d & Cluster 2: %d", cluster1rows, cluster2rows);
		// to store both clusters
		//double cluster1[cluster1rows][ncols];
		double** cluster1 = malloc(cluster1rows*sizeof(double*));
		for(i = 0; i < cluster1rows; i++) {
			cluster1[i] = (double *)malloc(sizeof(double) * ncols);
		}
		//double cluster2[cluster2rows][ncols];
		double** cluster2 = malloc(cluster2rows*sizeof(double*));
		for(i = 0; i < cluster2rows; i++) {
			cluster2[i] = (double *)malloc(sizeof(double) * ncols);
		}

		int c1=0, c2=0;
		for (int x = 0; x < nrows; x++){
			if(clusterid[x] == 0){
				for(j=0; j < ncols; j++){
						cluster1[c1][j] = data[x][j];
				}
				c1++;
			}
			else{
				for(j=0; j < ncols; j++){
						cluster2[c2][j] = data[x][j];
				}
				c2++;
			}
		}

		printf("\n");
		printf("Microarray:");
		for(i=0; i<ncols; i++) printf("\t%7d", i+1);
		printf("\n");
		// cdata contains centers info
		for (i = 0; i < nclusters; i++)
		{
			printf("Cluster %2d:", i);
			for (j = 0; j < ncols; j++)
				printf("\t%7.3f", cdata[i][j]);
			printf("\n");
		}

		// difference between centers
		double v[ncols];
		findDiffBetweenCenters(v, cdata);
		printf("\n");
		//for (j = 0; j < ncols; j++)
		//	printf("\t%7.3f", v[j]);

		// dot product of vector v
		double vectorProduct = dot_product(v,v,ncols);
		printf("\n");
		//printf("Vector Product: %2f", vectorProduct);

		// dot product of each row from cluster 1 with vector v
		/*double x1[cluster1rows];
		double x2[cluster2rows];
		for(int i=0;i<cluster1rows;i++)
		{
			x1[i] = dot_product(cluster1[i],v,ncols)/vectorProduct;
		}
		for(int i=0;i<cluster2rows;i++)
		{
			x2[i] = dot_product(cluster2[i],v,ncols)/vectorProduct;
		}*/

		double x1[nrows];
		for(int i=0;i<nrows;i++)
		{
			x1[i] = dot_product(data[i],v,ncols)/vectorProduct;
		}

		// pre-processing -->
		// normalize the 1-dimensional transformed vector x
		// to a vector with mean 0 and variance 1
		double z1[nrows];
		//double z2[cluster2rows];
		preprocess(x1, nrows, z1);
		andersondarling(z1, nrows);

		if(cluster_array[current_cluster].testScore <= 1.869 || cluster_array[current_cluster].nrows <=7)
		{
			current_cluster++;
		}
		else
		{
			// to remove cluster that needs to be processed
			deque_cluster(current_cluster);

			//add two sub-clusters
			cluster_array[clusterCounter].data = cluster1;
			cluster_array[clusterCounter].nrows = cluster1rows;
			cluster_array[clusterCounter+1].data = cluster2;
			cluster_array[clusterCounter+1].nrows = cluster2rows;
			//cluster_array[clusterCounter].center = cdata[0];
			//cluster_array[clusterCounter+1].center = cdata[1];
			// copy individual cluster center
			cluster_array[clusterCounter].center = (double *)malloc(sizeof(double) * ncols);
			cluster_array[clusterCounter+1].center = (double *)malloc(sizeof(double) * ncols);
			for(int j=0; j<ncols; j++){
				cluster_array[clusterCounter].center[j] = cdata[0][j];
				cluster_array[clusterCounter+1].center[j] = cdata[1][j];
			}

			//increase cluster count by two
			clusterCounter = clusterCounter + 2;
		}
		//preprocess(x2, cluster2rows, z2);
		//andersondarling(z2, cluster2rows);
		//clusterCounter++;

	}while(current_cluster<clusterCounter);

	printf("\n=====================cluster asmts===================\n");
	printf("Number of clusters: %d\n", clusterCounter);
	for(int k=0; k<clusterCounter; k++){
		printf(" Score of cluster %d: %d", k, cluster_array[k].nrows);
	}

	printf("\n=====================cluster centers===================\n");
	for(int k=0; k<clusterCounter; k++){
		printf("Center %d: ", k);
		for(int p=0; p<ncols; p++){
			printf("\t%7.3f", cluster_array[k].center[p]);
		}
		printf("\n");
	}
}

int* get_clusters_rows_index(int k, int nrows, int* clusterid) {
	int* clusterrows = malloc(k * sizeof(int));
	for (int i = 0; i < k; i++) {
		clusterrows[i] = 0;
	}
	for (int i = 0; i < nrows; i++) {
		//printf ("Gene %d: cluster %d\n", i, clusterid[i]);
		clusterrows[clusterid[i]]++;
	}
	for (int i = 0; i < k; i++) {
		printf("Cluster %d : %d\n", i, clusterrows[i]);
	}
	return clusterrows;
}

int main(int argc, char** argv) {
	FILE *fp;
    double **data;
    int **mask;
    int row = 0, col = 0;
    int firstLine = 1;
    int i, j;
    char c, *line;

    char *filename = argv[1];

    //open file
	fp = fopen(filename, "r");
    if(fp == NULL) {
            perror("can't open file\n");
            exit(errno);
    }

    //count total rows and columns
    while((c = fgetc(fp)) != EOF) {
    	if(firstLine && c == ',')
    		col++;
    	else if(c == '\n') {
    		if(firstLine)
    			col++;
    		firstLine = 0;
    		row++;
    	}
    }

    //allocate 2d matrix of size row x col
    data = (double **)malloc(sizeof(double *) * row);
    mask = (int **)malloc(sizeof(int *) * row);
    if(data == NULL || mask == NULL) {
    	perror("malloc failed\n");
    	exit(errno);
    }
    for(i = 0; i < row; i++) {
    	data[i] = (double *)malloc(sizeof(double) * col);
    	mask[i] = (int *)malloc(sizeof(int) * col);
    }

    //read each line from file, split delimited strings, and convert to double
    line = (char *)malloc(sizeof(char) * SIZE);
    fseek(fp, 0, SEEK_SET);
    i = 0;
    while(fgets(line, SIZE, fp)) {
    	tokenize(line, data[i]);
    	i++;
    }

    for(i = 0; i < row; i++) {
    	for(j = 0; j < col; j++) {
    		mask[i][j] = 1;
    	}
    }

    num_cols = col;
    performKmean(row,col,data,mask);
    create_csv("output.csv", row,col);
    //perform_mykmeans(row,col,data, mask);
    fclose(fp);
    return 0;
}
