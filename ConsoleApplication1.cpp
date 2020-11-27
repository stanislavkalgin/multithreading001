
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


void printfArray(double *d, int n, double t) {
    printf("%.8f\n", t);
    for (int i = 0; i < n; i++) {
        printf("%.7f ", d[i]);
    }
    printf("\n");
}

void f(double t, double *x, double *dxdt, int n) {
    int N = 10;
    int NNN = 1000;
    int i = 0;
    dxdt[i] = - 6 * x[i] * (x[i+1] - x[i-1+n]) * 0.5 * N
              - (x[i+2] - 2 * x[i+1] + 2 * x[i-1+n] - x[i-2+n]) * 0.5 * NNN;
    i = 1;
    dxdt[i] = - 6 * x[i] * (x[i+1] - x[i-1]) * 0.5 * N
              - (x[i+2] - 2 * x[i+1] + 2 * x[i-1] - x[i-2+n]) * 0.5 * NNN;
    #pragma omp parallel for
    for (i = 2; i < n-2; i++)
    {
        dxdt[i] = - 6 * x[i] * (x[i+1] - x[i-1]) * 0.5 * N
                  - (x[i+2] - 2 * x[i+1] + 2 * x[i-1] - x[i-2]) * 0.5 * NNN;
    }
    i = n-2;
    dxdt[i] = - 6 * x[i] * (x[i+1] - x[i-1]) * 0.5 * N
              - (x[i+2-n] - 2 * x[i+1] + 2 * x[i-1] - x[i-2]) * 0.5 * NNN;
    i = n-1;
    dxdt[i] = - 6 * x[i] * (x[i+1-n] - x[i-1]) * 0.5 * N
              - (x[i+2-n] - 2 * x[i+1-n] + 2 * x[i-1] - x[i-2]) * 0.5 * NNN;
}

int RKCalculation(int n, double t, double *x, double h, double finish) {

    if (h <= 0 || (finish - t) <= 0) {
        return -1;
    }

    double k1[500];
    double k2[500];
    double k3[500];
    double k4[500];
    double temp[500];

    while (true) {
        if (t > finish) {
            break;
        }               
        
        printfArray(x, n, t);

        //k1
        f(t, x, k1, n);
        //k2
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            temp[i] = x[i] + 0.5 * h * k1[i];
        }
        f(t + 0.5 * h, temp, k2, n);
        //k3
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            temp[i] = x[i] + 0.5 * h * k2[i];
        }
        f(t + 0.5 * h, temp, k3, n);
        //k4
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            temp[i] = x[i] + h * k3[i];
        }
        f(t + h, temp, k4, n);
        //res
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            x[i] += h * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0;
        }

        t += h; 
    }

    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(temp);
    return 0;
}

int main(int argc, char * argv[]) {
    int n = 500;
    double h = 0.001; // Шаг времени
    double x[500];
    double from = 0.0, to = 10.0; // t (time)

    double start_time = 0., end_time = 0.;
    omp_set_num_threads(4);
    start_time = omp_get_wtime();

    double k = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        double xx = i*0.1;
        x[i] = 2.0 * k * k / (cosh(k * (xx - 25)) * cosh(k * (xx - 25)));
    }
    RKCalculation(n, from, x, h, to);

    end_time = omp_get_wtime();

    printf("\nЗатраченное время: %.16g\n", end_time - start_time);
    
    free(x);
    return 0;
}
