/*
 * try.c
 *
 *  Created on: Mar 30, 2018
 *      Author: Rasika
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include "cluster.h"

#define SIZE 4096
#define MAX_CLUSTER_SIZE 20
#define MAX_CLUSTER_COUNT 400
int clusterCounter = 0;
int scoreCounter = 0;
double* testScore;
int current_cluster=0;

//define cluster as a matrix
typedef struct cluster {
	double **data;
	int nrows;
	double testScore;
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
	for(int j=0;j<16;j++){
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
		x[i] = dot_product(cluster[i],v,16)/vectorProduct;
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
	cluster_array[clusterCounter].testScore = A;
	//printf(" Score : %f", A);
}

void printClusters(double** cluster, int n){
	for (int i = 0; i < 100; i++)
	  {
		printf("Row %2d:", i);
		for (int j = 0; j < 16; j++)
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

void performKmean(int nrows, int ncols, double** data, int** mask){
	int i, j;
	const int nclusters = 2;
	const int transpose = 0;
	const char dist = 'e';
	const char method = 'a';
	int npass = 1;
	int ifound = 0;
	double error;
	cluster_array = (cluster *)malloc(MAX_CLUSTER_COUNT * sizeof(cluster));

	cluster_array[0].data = data;
	cluster_array[0].nrows = nrows;
	cluster_array[0].testScore = 100;
	clusterCounter++;

	do{
		if(cluster_array[current_cluster].testScore <= 1.8692 || cluster_array[current_cluster].nrows <=50)
			current_cluster++;
		else{
			data = cluster_array[current_cluster].data;
			nrows = cluster_array[current_cluster].nrows;

			// to remove cluster that needs to be processed
			deque_cluster(current_cluster);

			double* weight = malloc(ncols*sizeof(double));
			int* clusterid = malloc(nrows*sizeof(int));
			double** cdata = malloc(nclusters*sizeof(double*));

			testScore = malloc(MAX_CLUSTER_SIZE*sizeof(double));
			//double ***cluster_matrix = (double ***)malloc(sizeof(double **) * 15);
			int** cmask = malloc(nclusters*sizeof(int*));

			for (i = 0; i < nclusters; i++)
			{
				cdata[i] = malloc(ncols*sizeof(double));
				cmask[i] = malloc(ncols*sizeof(int));
			}

			for (i = 0; i < ncols; i++)
				weight[i] = 1.0;

			kcluster(nclusters,nrows,ncols,data,mask,weight,transpose,npass,method,dist,
			clusterid, &error, &ifound);
			printf ("Solution found %d times; within-cluster sum of distances is %f\n",
			ifound, error);
			printf ("Cluster assignments:\n");

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

			cluster_array[clusterCounter].data = cluster1;
			cluster_array[clusterCounter].nrows = cluster1rows;
			cluster_array[clusterCounter+1].data = cluster2;
			cluster_array[clusterCounter+1].nrows = cluster2rows;

			printf ("------- Cluster centroids:\n");
			getclustercentroids(nclusters, nrows, ncols, data, mask, clusterid,
								  cdata, cmask, 0, 'a');

			printf("Microarray:");
			for(i=0; i<ncols; i++) printf("\t%7d", i);
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
			double v[16];
			findDiffBetweenCenters(v, cdata);
			printf("\n");
			//for (j = 0; j < ncols; j++)
			//	printf("\t%7.3f", v[j]);

			// dot product of vector v
			double vectorProduct = dot_product(v,v,ncols);
			printf("\n");
			//printf("Vector Product: %2f", vectorProduct);

			// dot product of each row from cluster 1 with vector v
			double x1[cluster1rows];
			double x2[cluster2rows];
			for(int i=0;i<cluster1rows;i++)
			{
				x1[i] = dot_product(cluster1[i],v,ncols)/vectorProduct;
			}
			for(int i=0;i<cluster2rows;i++)
			{
				x2[i] = dot_product(cluster2[i],v,ncols)/vectorProduct;
			}

			// pre-processing -->
			// normalize the 1-dimensional transformed vector x
			// to a vector with mean 0 and variance 1
			double z1[cluster1rows];
			double z2[cluster2rows];
			preprocess(x1, cluster1rows, z1);
			andersondarling(z1, cluster1rows);
			clusterCounter++;
			preprocess(x2, cluster2rows, z2);
			andersondarling(z2, cluster2rows);
			clusterCounter++;

			printf("\n=====================cluster asmts===================\n");
			for(int k=0; k<clusterCounter; k++){
				printf(" Score of cluster %d: %f", k, cluster_array[k].testScore);
				//printClusters(cluster_array[k].data, cluster_array[k].nrows);
			}
			printf("\n");
		}
	}while(current_cluster<clusterCounter);
}

int main() {
	FILE *fp;
    double **data;
    int **mask;
    int row = 0, col = 0;
    int firstLine = 1;
    int i, j;
    char c, *line;

    //open file
	fp = fopen("gmeans2.csv", "r");
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

    performKmean(row,col,data,mask);
    fclose(fp);
    return 0;
}
