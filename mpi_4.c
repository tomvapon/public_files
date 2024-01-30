#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"


void generateRandomMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (double)rand() / RAND_MAX * 10; // Valores aleatorios entre 0 y 10
    }
}

void printMatrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.2f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
 

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Crear la matriz en el proceso maestro (0)
    double *matrix = NULL;
    if (rank == 0) {
        matrix = (double *)malloc(size * size * sizeof(double));
        generateRandomMatrix(matrix, size, size);
        printf("Matriz original:\n");
        printMatrix(matrix, size, size);
    }

    // Distribuir filas de la matriz a cada proceso
    double *localRow = (double *)malloc(size * sizeof(double));
    MPI_Scatter(matrix, size, MPI_DOUBLE, localRow, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calcular el máximo, mínimo y promedio de la fila local
    double localMax = localRow[0];
    double localMin = localRow[0];
    double localSum = localRow[0];
    for (int i = 1; i < size; ++i) {
        if (localRow[i] > localMax) {
            localMax = localRow[i];
        }
        if (localRow[i] < localMin) {
            localMin = localRow[i];
        }
        localSum += localRow[i];
    }
    double localAvg = localSum / size;

    printf("Process %d with row %d – min: %f; max: %f; avg: %f \n", rank, rank, localMin, localMax, localAvg);

    // Recopilar resultados en el proceso maestro
    double *globalMax = (double *)malloc(size * sizeof(double));
    double *globalMin = (double *)malloc(size * sizeof(double));
    double *globalAvg = (double *)malloc(size * sizeof(double));
    MPI_Gather(&localMax, 1, MPI_DOUBLE, globalMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&localMin, 1, MPI_DOUBLE, globalMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&localAvg, 1, MPI_DOUBLE, globalAvg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Proceso maestro calcula el máximo, mínimo y promedio global
    if (rank == 0) {
        double overallMax = globalMax[0];
        double overallMin = globalMin[0];
        double overallSum = globalAvg[0];
        for (int i = 1; i < size; ++i) {
            if (globalMax[i] > overallMax) {
                overallMax = globalMax[i];
            }
            if (globalMin[i] < overallMin) {
                overallMin = globalMin[i];
            }
            overallSum += globalAvg[i];
        }
        double overallAvg = overallSum / size;

        printf("\nResultados globales:\n");
        printf("Máximo global: %.2f\n", overallMax);
        printf("Mínimo global: %.2f\n", overallMin);
        printf("Promedio global: %.2f\n", overallAvg);
    }

    free(localRow);
    free(globalMax);
    free(globalMin);
    free(globalAvg);

    if (rank == 0) {
        free(matrix);
    }

    MPI_Finalize();

    return 0;
}
