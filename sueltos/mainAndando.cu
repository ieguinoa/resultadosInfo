/* Filename:  main.cu **************************************************************************** /
 *
 * INPUT:
 *   -Particulas.in:
 *     cantParticles
 *     type   x   y   z   Vx   Vy   Vz   q	; where
 *     dt					; (x,y,z)	= posición respecto de algún (0,0,0)
 *     temp0					; (Vx,Vy,Vz)	= Velocidades iniciales
 *     tautp					; dt		= delta_tiempo
 *     tempi					; q		= carga
 *     						; temp0		= temperatura target
 *     						; tempi		= temperatura inicial (No se usa aún)
 *     						; tautp		= factor de corrección de velocidades
 *     
 *     
 *     
 *   -TablaCoeficientesLennard
 *     type   sigma   epsilon   mass	min   max	; donde min y max indican de qué valor
 *     							; a qué valor hay que densificar las muestras
 *     							; (NO ESTA IMPLEMENTADO AUN)
 *
 * ALGORITMO:
 *   1-Levantar Coeficientes
 *   2-Armar matriz de lennard para cant_samples_r muestras
 *	Para cada tipo de partícula:
 *	    Calcular en funcion de los coeficientes el potencial para cant_samples_r valores r
 *   3-Levantar partículas
 *	Ordenar y armar índices
 *   Para cada iteración de MD:
 *	4-Calcular distancias:
 *	    Cada partícula contra todas las otras
 *	    Armar matriz de distancias
 *      5-Calcular las derivadas respecto de r para cada par de partículas
 *	6-Calcular fuerza para cada particula:
 *	    Cada partícula contra todas las otras: matriz 3D
 *	    Obtener fuerza resultante para cada partícula: vector 3D
 *	7-Calcular nuevas posiciones: vector 3D
 *
 ***************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include <iomanip>
#include <sys/time.h>


/** **************************************************************** **/
/** ************* DEFAULT GLOBAL VARIABLES VALUES ****************** **/
#define BLOCK_SIZE_X		32
#define BLOCK_SIZE_Y		16
#define BLOCK_SIZE		(BLOCK_SIZE_X*BLOCK_SIZE_Y)

#define TEXTURE_MEM_SIZE	262000
#define DIF_FINITAS_DELTA	4

/** Variables físicas **/
#define CANT_TYPES		37
#define MAx			45		
#define MIn			0.001
#define DIST			(MAx - MIn)

#define DELTA_TIEMPO		0.0001
#define TEMP			100
#define TAO			0.1

#define BOX_MAX			999	// distancia máxima del 0 para cada coordenada
					// Determinamos un cubo de volumen = (2*BOX_MAX) ^3

/** Filenames **/
char* lennardTableFileName = "Input_Mache/TablaCoeficientesLennard";
char* particlesFileName = "Input_Mache/particles.in";
char* debugOutputFilename = "Output_Mache/debug.out";
char* outputFilename = "Output_Mache/results.out";
char* crdFilename = "Output_Mache/mdcrd";
char* timeFilename = "Output_Mache/times.out";

using namespace std;
// streamsize ss = cout.precision();

/** **************************************************************** **/
/** ******************** GLOBAL VARIABLES ************************** **/
texture <float, cudaTextureType2D,cudaReadModeElementType> texRef;
double delta_tiempo = DELTA_TIEMPO;
double temp0 = TEMP;
double tempi;
double tautp = TAO;

double Boltzmann_cte = 0.0019872041;
double box_max_x = BOX_MAX;
double box_max_y = BOX_MAX;
double box_max_z = BOX_MAX;
bool box = true;
double cut = 12;

int cant_steps = 1;
int cant_types = CANT_TYPES;

bool CPU=false;
bool derivative = false;
bool analytic = false;
bool results = false;
bool amberResults = false;
bool coordinates = false;
bool periodicity = false;






/** **************************************************************** **/
/** ************************* DEVICE ******************************* **/



/**
 *    RECIBE UN VALOR DE EPSILON Y SIGMA (e,s) Y EL ARREGLO CON TODOS LOS DEMAS VALORES (* EPS,* SIG)
 *    GUARDA EL POTENCIAL(EN LJ_POT) DE e,s  VS TODOS LOS VALORES DE EPS Y SIG
 */

__global__ 
void lennard_Kernel(float* LJ_POT, double* EPS, double* SIG,
		    double e, double s, double var, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      
      /* Variables */
      double sig12 = (double) (s + SIG[y])/2;
      double eps12 = (double) sqrt(e * EPS[y]);
      double r = (double) MIn+x*var;
      
      /* Resultado */
      LJ_POT[y*width +x] = (float) 4.0*eps12*( pow((sig12/r),12) - pow((sig12/r),6));
}



/** **************************************************************** **/

/**
 *    RECIBE UN VALOR DE EPSILON Y SIGMA (e,s) Y EL ARREGLO CON TODOS LOS DEMAS VALORES (* EPS,* SIG)
 *    GUARDA LA DERIVADA DEL POTENCIAL(EN dLJ_POT) DE e,s  VS TODOS LOS VALORES DE EPS Y SIG
 */

__global__ 
void derivatives_lennard_Kernel(float* dLJ_POT, double* EPS, double* SIG,
		    double e, double s, double var, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      
      /* Variables */
      double sig12 = (double) (s + SIG[y])/2;
      double eps12 = (double) sqrt(e * EPS[y]);
      double r = (double) MIn+x*var;
      
      /* Resultado */
      dLJ_POT[y*width +x] = (float) 24.0*eps12*( pow(sig12,6)/ pow(r,7) - 2 * pow(sig12,12)/ pow(r,13));
}
      
/** **************************************************************** **/
__global__ 
void close_distances_kernel(double* X, double* Y, double* Z, double* R,
			    double* position_x, double* position_y, double* position_z, 
			    double box_x, double box_y, double box_z, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(i >= width || j >= height) {return;}
      unsigned int pos = j*width+i;
      
      double _X = position_x[i] - position_x[j];
      double _Y = position_y[i] - position_y[j];
      double _Z = position_z[i] - position_z[j];
      
      _X = _X - box_x * round((double) _X/box_x);
      _Y = _Y - box_y * round((double) _Y/box_y);
      _Z = _Z - box_z * round((double) _Z/box_z);
      X[pos] = _X;
      Y[pos] = _Y;
      Z[pos] = _Z;
      R[pos] = (double) sqrt( _X*_X + _Y*_Y + _Z*_Z );
}


/** **************************************************************** **/

__global__ 
void distances_kernel(double* R, double* X, double* Y, double* Z, 
		      double* x1, double* y1, double* z1, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      
      double x_ = x1[x] - x1[y];
      double y_ = y1[x] - y1[y];
      double z_ = z1[x] - z1[y];
      X[y*width+x] = x_;
      Y[y*width+x] = y_;
      Z[y*width+x] = z_;
      R[y*width+x] = (double) sqrt( x_*x_ + y_*y_ + z_*z_ );
}

/** **************************************************************** **/

__global__ 
void derivative_E_r(double* dEr, double* r, double cut, int* item_to_type,
		    int cant_samples_r, int cant_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}
      
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      float t_o_p_1 = (float) item_to_type[y] * cant_types;	//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;	//this one decides which row on these
      
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      float index_x = (float)((double) (r[y*width+x] - MIn) * (double) cant_samples_r / DIST + 0.5);	// convert  r  to   x
      /*
	double rposta=r[y*width+x];
      if(rposta> MAx)
        rposta=MAx;
        else
                if(rposta<MIn)
                        rposta=MIn;

   float index_x = (float)((double) (rposta - MIn) * (double) cant_samples_r / DIST + 0.5);  // convert  r  to   x      
*/

      double E_r_up = (double) tex2D( texRef, index_x + DIF_FINITAS_DELTA, t_o_p_2 );
      double E_r_dwn = (double) tex2D( texRef, index_x - DIF_FINITAS_DELTA, t_o_p_2 );
      
      
      double r_dif = DIST * 2 * (DIF_FINITAS_DELTA) / cant_samples_r;
      
      
      dEr[y*width+x] = (E_r_up - E_r_dwn) / (r_dif);
}

/** **************************************************************** **/




/*********************************************
 * ESTA FUNCION RECIBE CALCULA dER A PARTIR DE LOS DATOS TABULADOS DE LA MATRIZ EN MEMORIA (dHLJPot)
 * *****************************************/
 
void direct_derivative_E_r_MEMORY(float* dLJPot, double* dEr, double* r, int* item_to_type,
		    int cant_samples_r, int cant_types, int width, int height, int x, int y)
{
      /* Elemento de la matriz a calcular */
      //unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      //unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      
      /* Dentro del bloque correspondiente */
      //if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      //float t_o_p_1 = (float) item_to_type[y] * cant_types;	//this one decides which subMatrix to use
      //float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;	//this one decides which row on these
      
      
      float t_o_p_1 = (float) item_to_type[y] * cant_types;	//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x] + t_o_p_1;	//this one decides which row on these
      float posInicial=t_o_p_2 * cant_samples_r;   //comienzo de la fila??
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      //float index_x = (float)((double) (r[y*width+x] - MIn) * (double) cant_samples_r / DIST + 0.5);	// convert  r  to   x
      int index=0;
      double r=r[y*width+x];
      if(r> MAx)
	dEr[y*width+x]=dLJPot[posInicial+cant_samples_r -1];
      else 
	if(r<MIn)
	  dEr[y*width+x]=dLJPot[posInicial];
	else
	  dEr[y*width+x]=dLJPot[posInicial+round((r-min)*(cant_samples_r/DIST))];
	  //dEr[y*width+x] = (double) tex2D( texRef, index_x, t_o_p_2 );
}


//******************************************************


__global__ 
void direct_derivative_E_r(double* dEr, double* r, double cut, int* item_to_type,
		    int cant_samples_r, int cant_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      float t_o_p_1 = (float) item_to_type[y] * cant_types;	//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;	//this one decides which row on these
      
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
     float index_x = (float)((double) (r[y*width+x] - MIn) * (double) cant_samples_r / DIST + 0.5);	// convert  r  to   x
     /* double rposta=r[y*width+x];
      if(rposta> MAx)
	rposta=MAx;
	else
		if(rposta<MIn)
			rposta=MIn;
   
   float index_x = (float)((double) (rposta - MIn) * (double) cant_samples_r / DIST + 0.5);  // convert  r  to   x	
*/	

      dEr[y*width+x] = (double) tex2D( texRef, index_x, t_o_p_2 );
}

/** **************************************************************** **/


__global__ 
void E_r(double* Er, double* r, double cut, int* item_to_type,
	 int cant_samples_r, int cant_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {Er[y*width+x] = 0; return;}
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      float t_o_p_1 = (float) item_to_type[y];		//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x];		//this one decides which row on these
      float row =  t_o_p_2 + 0.5 + (t_o_p_1* cant_types); 
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      float index_x = (float)((double) (r[y*width+x] - MIn) * (double) cant_samples_r / DIST + 0.5);	// convert  r  to   x

/*
	 double rposta=r[y*width+x];
      if(rposta> MAx)
        rposta=MAx;
        else
                if(rposta<MIn)
                        rposta=MIn;

   float index_x = (float)((double) (rposta - MIn) * (double) cant_samples_r / DIST + 0.5);  // convert  r  to   x      
    */  


      Er[y*width+x] = (double) tex2D( texRef, index_x, row );
}

/* ***************************************************************** **/
/** +ANALYTIC */
/** **************************************************************** **/

__global__ 
void derivative_E_r_analytic(double* dEr, double* r, double cut, int* item_to_type, int cant_samples_r,
		    double* EPS, double* SIG, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particle 2 **/
      int type_i = item_to_type[x];
      int type_j = item_to_type[y];
      
      double sig12 = (double) (SIG[type_i] + SIG[type_j])/2;
      double eps12 = (double) sqrt(EPS[type_i] * EPS[type_j]);
      
      dEr[y*width+x] = (double) 24.0*eps12*( pow(sig12,6)/ pow(r[y*width+x],7) - 2 * pow(sig12,12)/ pow(r[y*width+x],13));
}

__global__ 
void E_r_analytic(double* Er, double* r, double cut, int* item_to_type, int cant_samples_r,
	       double* EPS, double* SIG, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {Er[y*width+x] = 0; return;}
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particle 2 **/
      int type_i = item_to_type[x];
      int type_j = item_to_type[y];
      
      double sig12 = (double) (SIG[type_i] + SIG[type_j])/2;
      double eps12 = (double) sqrt(EPS[type_i] * EPS[type_j]);
      
      Er[y*width+x] = (double) 4.0*eps12*( pow((sig12/r[y*width+x]),12) - pow((sig12/r[y*width+x]),6));
}



/** **************************************************************** **/
/** -ANALYTIC */
/* ***************************************************************** **/


/** **************************************************************** **/


  /*   Fx =  dE(r) / dr  *  (x1-x2) / r               */
__global__ 
void Parcial_Forces_Kernel(double* force, double* dEr, double* dif, double* r, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      if(x == y) {force[y*width+x] = 0; return;}
      
	//force[y*width+x] = dEr[y*width+x] *  dif[y*width+x] ;
 
      force[y*width+x] = dEr[y*width+x] * dif[y*width+x] / r[y*width+x];
}

/** **************************************************************** **/


__global__ 
void Resultant_Forces_Kernel(double* result, double* forces, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(x >= cant) {return;}
      
      int i = 0;
      double tmp = 0;
      int row = x*cant;
      for(; i < cant; i++){
	 tmp += forces[row + i];
      }
      result[x] = tmp;
}

/** **************************************************************** **/


/*  V(t + Dt/2) = V(t - Dt/2) +  [ F(t) * Dt ] / m  */  
__global__
void Resultant_Velocities_Kernel(double* velocity, double* old_velocity, double* force, double* m,
				 int* item_to_type, double delta_tiempo, int cant_particles)
{      
      /* Elemento de la matriz a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i >= cant_particles) {return;}
      
      double Vt = old_velocity[i];
      int type = item_to_type[i];
      double dtx = delta_tiempo*20.454999999999;
      //double dtx=delta_tiempo;

	/* Result */
      velocity[i] = Vt + ( (force[i]*dtx) / m[type] );
}

/** **************************************************************** **/


/*       P(t + Dt) = P(t) +  V(t + Dt/2) * Dt       */
__global__ 
void Resultant_Positions_Kernel(double* positions, double* velocity, double delta_tiempo, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i >= cant) {return;}
      double dtx = delta_tiempo*20.454999999999;
      //double dtx=delta_tiempo;
	positions[i] = positions[i] + (velocity[i] * dtx);
}

/** **************************************************************** **/



/*  -BOX_MAX              0              BOX_MAX   */
/*      |-----------------|-----------------|      */

__global__ 
void Adjustin_Positions_Kernel(double* position, double box_max, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i >= cant) {return;}
      double pos = position[i] - box_max;
      if(pos > 0){
	position[i] = -box_max + fmod(pos, (double) (2*box_max));
      }
      if(pos < -2*box_max){
	position[i] = box_max + fmod(pos, (double) (2*box_max));
      }
      
}


/** **************************************************************** **/


/*            Ek = |v|^2  *  m / 2                  */
/*            Ek_x = (v_x)^2  *  m / 2              */
__global__
void Kinetic_Energy_Kernel(double* kE, double* vold, double* v, double* m, int* item_to_type, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i>= cant) {return;}
      
    double vi = vold[i] + v[i];
      //   double vi=v[i];
	 int type = item_to_type[i];
      
     // kE[i] = vi * vi * m[type] / 2;

     kE[i] = vi * vi * m[type] / 8;
}

/** **************************************************************** **/


__global__
void Total_Kinetic_Energy_Kernel(double* kE, double* Ke_x, double*  Ke_y, double* Ke_z, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i>= cant) {return;}
      
      kE[i] = Ke_x[i] + Ke_y[i] + Ke_z[i];
}

/** **************************************************************** **/


__global__
void Corrected_Velocities_Kernel(double* vold, double* v, double lambda, int cant){
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i>= cant) {return;}
      vold[i] = v[i];
      //vold[i] = v[i] * lambda;
}











/** **************************************************************** **/
/** *************************** HOST ******************************* **/

int main( int argc, char* argv[] )
{
  
 //cudaSetDevice(1); 
  
  //PROCESO LOS PARAMETROS DE ENTRADA
    for(uint i = 0; i < argc; i++){
      if(strcmp(argv[i], "-t") == 0){
	/* outputTimeFilename  */
	timeFilename = argv[i+1];
      }
      if(strcmp(argv[i], "-a") == 0){
	/* ANALYTIC mode */
	analytic = true;
      }
      if(strcmp(argv[i], "-d") == 0){
	/* DERIVATIVE mode */
	derivative = true;
      }
      if(strcmp(argv[i], "-r") == 0){
	/* RESULTS or TIMER mode */
	results = true;
	amberResults = true;
      }
      if(strcmp(argv[i], "-ar") == 0){
	/* RESULTS */
	amberResults = true;
      }
      
      if(strcmp(argv[i], "-c") == 0){
	/* PRINT mdcrd file */
	coordinates = true;
      }
      
      if(strcmp(argv[i], "-p") == 0){
	/* Periodicity */
	periodicity = true;
      }
      if(strcmp(argv[i], "-cpu") == 0){
	/* Periodicity */
	CPU = true;
      }
    }
    
    
    
   //IMPRIMO QUE CARAJO ESTOY EJECUTANDO 
    if (derivative)
      cout << "Derivative" << endl;
    if (analytic)
      cout << "Analytic" << endl;
    if(results){
      cout << "DEBUG mode ON" << endl;
    }
    if(amberResults){
      cout << "AMBER results ON" << endl;
    }
    
    
    
    
    //CONFIGURAR OUTPUT
    
    fstream out;
    fstream crd;
    //if(results or amberResults){
	/* Output file */
	out.open(outputFilename,fstream::out);
	streamsize ss = out.precision();
	out << setprecision(20);
    //}
    if(coordinates){
	/* CRD output file */
	crd.open(crdFilename,fstream::out);
	crd << setprecision(3);
	crd.setf( std::ios::fixed, std:: ios::floatfield );
	crd << "  POS(x)  POS(y)  POS(z)" << endl;
    }

    struct timeval  tv1, tv2;
    fstream taim;
    if(!results){    //timer mode ON
	/* Time output file */
	taim.open(timeFilename, fstream::app | fstream::out);
	taim << setprecision(20);
    }
    
    
    
    
       
    
    
    
  /* Levantamos Coeficientes de Lennard */
  ifstream table (lennardTableFileName);
    table >> cant_types;
    /**Variables y memoria*/
    size_t cant_types_size = cant_types * sizeof(double);
    
    vector<string> h_type;
    h_type.resize(cant_types);
    double* h_sigma = (double*) ( malloc(cant_types_size));
    double* h_epsilon = (double*) ( malloc(cant_types_size));
    double* h_mass = (double*) ( malloc(cant_types_size));
    
     
    /**Levantamos datos*/
    for(int j = 0; j<cant_types ; j++){
      table >> h_type[j];
      table >> h_sigma[j];
      table >> h_epsilon[j];
      table >> h_mass[j];
    }
  table.close();
    
  
  
// ***************** 
//VARIABLES PARA GUARDAR ENERGIA TOTAL

double diferencia,  etotalX , etotinicial;

//****************************** 




 
  
   /*******************************/  
   /*Armamos matrices de lennard */
  /******************************/
  /**Variables y memoria**/
    int cant_samples_r = TEXTURE_MEM_SIZE/(sizeof(float));	// cant of original sample values (máximo permitido por mem de textura)
    double var = DIST / ((double) cant_samples_r);		// variation of r
    size_t cant_samples_r_size = cant_samples_r * sizeof(float);
    
    float* h_dLJPot;
    float* h_LJPot;
    
    if(derivative)
      h_dLJPot = (float*) malloc(cant_samples_r_size*cant_types*cant_types);	// #samples * #particles * #particles (*float)
    else
      h_LJPot = (float*) malloc(cant_samples_r_size*cant_types*cant_types);	// #samples * #particles * #particles (*float)
    
    int width = cant_samples_r;
    int height = cant_types;
    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid( (int) ceil((double)width / (double)dimBlock.x), (int) ceil((double)height / (double)dimBlock.y) );
    
    
    double* d_EPS;     //ARRAY PARA TODOS LOS VALORES DE EPSILON
    double* d_SIG;      //ARRAY PARA TODOS LOS VALORES DE SIGMA
    float* d_LJPot;
    float* d_dLJPot;
    cudaMalloc(&d_EPS, cant_types_size);    
    cudaMalloc(&d_SIG, cant_types_size);   
    cudaMemcpy(d_EPS, h_epsilon, cant_types_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SIG, h_sigma, cant_types_size, cudaMemcpyHostToDevice);
    if(derivative)
      cudaMalloc(&d_dLJPot, cant_samples_r_size * cant_types);
    else
      cudaMalloc(&d_LJPot, cant_samples_r_size * cant_types);
    
    
    
    /** Rellenamos datos con CUDA **/
    //CANTIDAD TOTAL DE THREADS:  EN X=cant_samples_r     EN Y=cant_types 
    if(derivative) {   //LLENO LA TEXTURA CON LAS DERIVADAS PARA CADA PAR DE TIPOS
      for(int a = 0; a<cant_types; a++){
	derivatives_lennard_Kernel<<<dimGrid, dimBlock>>>(d_dLJPot, d_EPS, d_SIG, h_epsilon[a], h_sigma[a], var, width, height);
	cudaMemcpy( (float*) &(h_dLJPot[(a*cant_samples_r*cant_types)]), d_dLJPot, cant_types * cant_samples_r_size, cudaMemcpyDeviceToHost);
      }
    } else {
      //LLENO LA TEXTURA CON LAS DERIVADAS DEL POTENCIAL PARA CADA PAR DE TIPOS
      for(int a = 0; a<cant_types; a++){
	lennard_Kernel<<<dimGrid, dimBlock>>>(d_LJPot, d_EPS, d_SIG, h_epsilon[a], h_sigma[a], var, width, height);
	cudaMemcpy( (float*) &(h_LJPot[(a*cant_samples_r*cant_types)]), d_LJPot, cant_types * cant_samples_r_size, cudaMemcpyDeviceToHost);
      }
    }
    
    /** Liberamos memoria de CUDA **/
    cudaFree(&d_EPS);
    cudaFree(&d_SIG);
    cudaFree(&d_LJPot);   
  
    
    
    
    
    /** DEBUG **/
    if(results){
	    if(derivative)
	      out << " derivative LENNARD " << endl;
	    else
	      out << " LENNARD " << endl;
	    for(int a = 0; a<cant_types; a++){
	      out << " Type = " << h_type[a] << endl << "  ";
	      for(int i = 0; i<cant_types; i++){
		for(int j = 0; j<cant_samples_r; j+= cant_samples_r/8){
		  if(derivative)
		    out << h_dLJPot[(a*cant_types*cant_samples_r)+(i*cant_samples_r)+j] << ", ";
		  else
		    out << h_LJPot[(a*cant_types*cant_samples_r)+(i*cant_samples_r)+j] << ", ";
		}
		out << endl << "  ";
	      }
	      out << "***********************************************************************************"  << endl;
	    }
    }

    
    
    
    
    
    
    
  /*Levantamos partículas*/
  fstream particles;
  particles.open(particlesFileName);
    
    /** Variables y memoria **/
    uint cant_particles;
    double* h_position_x;
    double* h_position_y;
    double* h_position_z;
    double* h_velocity_x;
    double* h_velocity_y;
    double* h_velocity_z;
    double* h_velocity_old_x;
    double* h_velocity_old_y;
    double* h_velocity_old_z;
    double* h_chargue;
    double h_box_x;
    double h_box_y;
    double h_box_z;
    double h_box_alpha;
    double h_box_beta;
    double h_box_gamma;
    vector<string> h_particle_type;
    particles >> cant_particles;    //PRIMER LINEA DE particles.in ES EL NUMERO DE PARTICULAS QUE HAY
    size_t cant_particles_size = cant_particles * sizeof(double);
    h_position_x = (double*)malloc(cant_particles_size);
    h_position_y = (double*)malloc(cant_particles_size);
    h_position_z = (double*)malloc(cant_particles_size);
    h_velocity_x = (double*)malloc(cant_particles_size);
    h_velocity_y = (double*)malloc(cant_particles_size);
    h_velocity_z = (double*)malloc(cant_particles_size);
    h_velocity_old_x = (double*)malloc(cant_particles_size);
    h_velocity_old_y = (double*)malloc(cant_particles_size);
    h_velocity_old_z = (double*)malloc(cant_particles_size);
    h_chargue = (double*)malloc(cant_particles_size);
    h_particle_type.resize(cant_particles);
    
    
    /** Guardamos datos en memoria : coordenadas, velocidades, tipos, cargas **/
    for(uint i = 0; i < cant_particles ; i++) {
      particles >> h_particle_type[i];
      
      particles >> h_position_x[i];
      particles >> h_position_y[i];
      particles >> h_position_z[i];
      
      particles >> h_velocity_old_x[i];
      particles >> h_velocity_old_y[i];
      particles >> h_velocity_old_z[i];
      
      particles >> h_chargue[i];
    }
    
    
    
    
    
    
    /** Perioricidad **/ 
    //TODO: por ahora usamos cubo,
    //situamos el cero en el centro del mismo
    //Recibimos en orden x, y, z
    particles >> box;
    if(box){
      cout << " Levantamos caja" << endl;
      particles >> h_box_x;
      particles >> h_box_y;
      particles >> h_box_z;
      particles >> h_box_alpha;
      particles >> h_box_beta;
      particles >> h_box_gamma;
      if( h_box_alpha != 90 or h_box_beta != 90 or h_box_gamma != 90){
	    cout << " Se forzaron los angulos para que sea un CUBO: " << endl;
      }
      box_max_x = h_box_x/2;
      box_max_y = h_box_y/2;
      box_max_z = h_box_z/2;
    }
    /** Parametros **/
    particles >> cant_steps;
    particles >> delta_tiempo;
    particles >> temp0;
    particles >> tempi;
    particles >> tautp;
    particles >> cut;
  particles.close();
  
  
  
  
  
//     if(results){
//       /** DEBUG **/
// 	    out << " INITIAL VALUES" << endl;
// 	    for(int i = 0; i<cant_particles; i++){
// 	      out << "  Type: " << h_particle_type[i] << " | Pos: (" << h_position_x[i] << " , " << h_position_y[i] << " , " << h_position_z[i] << ")";
// 	      out << " | Vel: (" << h_velocity_old_x[i] << " , " << h_velocity_old_y[i] << " , " << h_velocity_old_z[i] << ")" << endl;
// 	    }
// 	    out << endl;
// 
//       /** DEBUG **/
//     }
//     if(results){
    //   /** DEBUG **/
    // 	out << " CANT of TYPES" << endl;
    // 	for(int i = 0; i < h_type.size(); i++){
    // 	  out << "  " << h_type[i] << " " << cant_of_typ[i] << endl;
    // 	}
    // 	out << endl;
      /** DEBUG **/
//     }  
  
    /* Armamos estructura de items para saber de qué tipo
    /* es la partícula en la que estamos en CUDA */
    /** h_particle_type =  H H H H H K K K K K O O O O O O O O O ... **/
    /** h_item_particle =  1 1 1 1 1 3 3 3 3 3 9 9 9 9 9 9 9 9 9 ... **/
    
    
    
    //ARMO UN ARRAY CON EL TIPO DE CADA PARTICULA
    int * h_item_particle = (int*)malloc(cant_particles * sizeof(int));
    int * d_item_particle;
    cudaMalloc(&d_item_particle, cant_particles * sizeof(int));
    
    /** Convertimos anotamos type de la partícula como un int que sería el index dentro de h_type **/
    for(int i = 0; i< cant_particles; i++){
      for(int j = 0; j< h_type.size(); j++){
	if(h_type[j] == h_particle_type[i]){
	    h_item_particle[i] = j;
	    break;
	}
      }
    }
    
    cudaMemcpy(d_item_particle, h_item_particle, cant_particles * sizeof(int), cudaMemcpyHostToDevice);
  

//     if(results){
//       /** DEBUG **/
// 	    out << " ITEM to TYPE" << endl;
// 	    for(int i = 0; i < cant_particles; i++){
// 	      out << "  Particle[" << i << "]  | Type: " << h_type[h_item_particle[i]] << " (index :" << h_item_particle[i] << ") " << endl;
// 	    }
// 	    out << endl;
//       /** DEBUG **/
//     }  
  
  
    
    
    
    
    
    
    
  /* ************************************************ */
  /*     MANEJO DE MEMORIA EN EL DISPOSITIVO GPU      */
  /* ************************************************ */
    /** Variables **/
    size_t s_size = cant_particles_size * cant_particles;
    
    /** Positions **/
    double* d_position_x;
    double* d_position_y;
    double* d_position_z;
    cudaMalloc(&d_position_x, cant_particles_size);
    cudaMalloc(&d_position_y, cant_particles_size);
    cudaMalloc(&d_position_z, cant_particles_size);
    cudaMemcpy(d_position_x, h_position_x, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_position_y, h_position_y, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_position_z, h_position_z, cant_particles_size, cudaMemcpyHostToDevice);
    
    /** Positions **/
    double* d_pos_close_x;
    double* d_pos_close_y;
    double* d_pos_close_z;
    cudaMalloc(&d_pos_close_x, cant_particles_size);
    cudaMalloc(&d_pos_close_y, cant_particles_size);
    cudaMalloc(&d_pos_close_z, cant_particles_size);
    
    /** Particle's mass **/
    double* d_mass;
    cudaMalloc(&d_mass, cant_types_size);
    cudaMemcpy(d_mass, h_mass, cant_types_size, cudaMemcpyHostToDevice);
    
    /** Velocities **/
    double* d_velocity_x;
    double* d_velocity_y;
    double* d_velocity_z;
    double* d_velocity_old_x;
    double* d_velocity_old_y;
    double* d_velocity_old_z;
    cudaMalloc(&d_velocity_x, cant_particles_size);
    cudaMalloc(&d_velocity_y, cant_particles_size);
    cudaMalloc(&d_velocity_z, cant_particles_size);
    cudaMalloc(&d_velocity_old_x, cant_particles_size);
    cudaMalloc(&d_velocity_old_y, cant_particles_size);
    cudaMalloc(&d_velocity_old_z, cant_particles_size);
    cudaMemcpy(d_velocity_old_x, h_velocity_old_x, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_old_y, h_velocity_old_y, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_old_z, h_velocity_old_z, cant_particles_size, cudaMemcpyHostToDevice);
    
    /** Distances **/
    double* d_distance_x;
    double* d_distance_y;
    double* d_distance_z;
    double* d_distance_r;
    cudaMalloc(&d_distance_x, s_size);
    cudaMalloc(&d_distance_y, s_size);
    cudaMalloc(&d_distance_z, s_size);
    cudaMalloc(&d_distance_r, s_size);
    
    /** Derivatives **/
    double* d_dEr;
    cudaMalloc(&d_dEr, s_size);
    
    /** VDWAALS **/
    double* d_Er;
    cudaMalloc(&d_Er, s_size);
    
    /** Forces **/
    double* d_Force_x;
    double* d_Force_y;
    double* d_Force_z;
    cudaMalloc(&d_Force_x, s_size);
    cudaMalloc(&d_Force_y, s_size);
    cudaMalloc(&d_Force_z, s_size);
    
    double* d_Force_x_resultant;
    double* d_Force_y_resultant;
    double* d_Force_z_resultant;
    cudaMalloc(&d_Force_x_resultant, cant_particles_size);
    cudaMalloc(&d_Force_y_resultant, cant_particles_size);
    cudaMalloc(&d_Force_z_resultant, cant_particles_size);
    
    /** Kinetic Energy **/
    double* d_kinetic_energy;
    double* d_kinetic_energy_x;
    double* d_kinetic_energy_y;
    double* d_kinetic_energy_z;
    cudaMalloc(&d_kinetic_energy, cant_particles_size);
    cudaMalloc(&d_kinetic_energy_x, cant_particles_size);
    cudaMalloc(&d_kinetic_energy_y, cant_particles_size);
    cudaMalloc(&d_kinetic_energy_z, cant_particles_size);
    
  
    
    
    
    
    
    
    
  /* ************************************************ */
  /*           MANEJO DE MEMORIA EN EL HOST           */
  /* ************************************************ */
    /** Distances **/
    double (*h_distance_x)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_distance_y)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_distance_z)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_distance_r)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    
    /** Forces **/
    double (*h_Force_x)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_Force_y)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_Force_z)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    
    double* h_Force_x_resultant = (double*)malloc(cant_particles_size);
    double* h_Force_y_resultant = (double*)malloc(cant_particles_size);
    double* h_Force_z_resultant = (double*)malloc(cant_particles_size);
    
    /** Kinetic Energy **/
    double* h_kinetic_energy = (double*)malloc(cant_particles_size);
    double* h_kinetic_energy_x = (double*)malloc(cant_particles_size);
    double* h_kinetic_energy_y = (double*)malloc(cant_particles_size);
    double* h_kinetic_energy_z = (double*)malloc(cant_particles_size);
  
    
    
    
    
    
    
    
  /* ************************************************ */
  /*       Calculamos ENERGIA CINETICA deseada        */
  /* ************************************************ */
  /*            Ek = Kb * T (3N - Nc) / 2             */
    double Nc = 5;
    double factor_conv_T_Ek = 2 / (Boltzmann_cte * (3 *cant_particles - Nc) );

    if(amberResults){
	double kinetic_Energy = Boltzmann_cte * temp0 * (3*cant_particles - Nc) / 2;
	
	/** DEBUG **/
	    out << " THEORETICAL VALUES:" << endl << endl;
	    out << "  * Kb = " << Boltzmann_cte << endl << endl;
	    out << "  * Temperature = " << temp0 << endl << endl;
	    out << "  * Kinetic Energy = " << kinetic_Energy << endl << endl;
	    out << "  * Factor_conv_T_Ek = " << factor_conv_T_Ek << endl << endl;
	/** DEBUG **/
    }    
    
    
    
    
    
  /* ************************************************ */
  /*         Seteamos la memoria de TEXTURA           */
  /* ************************************************ */
    cudaArray* cuLennard_i;
//     if(!analytic){
      /** Usamos texturas **/
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
      cudaMallocArray(&cuLennard_i, &channelDesc, cant_samples_r, cant_types*cant_types);		//width x height
      
      texRef.addressMode[0] = cudaAddressModeClamp;
    //texRef.addressMode[0] = cudaAddressModeBorder;  

     texRef.filterMode = cudaFilterModeLinear; //cudaFilterModePoint; //		//Tipo de interpolación
      
      if(derivative) {
	cudaMemcpyToArray(cuLennard_i, 0, 0, h_dLJPot, cant_types * cant_types * cant_samples_r_size, cudaMemcpyHostToDevice);
      } else {
	cudaMemcpyToArray(cuLennard_i, 0, 0, h_LJPot, cant_types * cant_types * cant_samples_r_size, cudaMemcpyHostToDevice);
      }
      /** Bindeamos la textura **/
      cudaBindTextureToArray(texRef, cuLennard_i, channelDesc); 
//     }   
    

    if(amberResults){
	out << endl << "   ESTARTIN DE PROGRAM" << endl;
	out << "    Amaunt of itereishons = " << cant_steps << endl << endl;
    }
	  
  // for(int i=0 ; i<1000000 ; i++){
    //  for(int j=0 ; j<1000 ; j++){
	
     //}
  // }
    /** Esperamos a que termine de bindear la textura **/
    cudaDeviceSynchronize();
    if(!results){    //timer mode ON
	/** Arrancamos medicion del tiempo **/
	gettimeofday(&tv1, NULL);
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for(int step = 0; step < cant_steps; step++){
	/* ********************************************************************************************************** */
	/* ****************************************** INICIO Iteracion DM ******************************************* */
	/* ********************************************************************************************************** */
	
	    
	if(amberResults){
	    out << "/* ************************************************************************************************ */" << endl;
	    out << "/* ************************************* INICIO Iteracion " << step << " ************************************ */" << endl;
	    out << "/* ************************************************************************************************ */" << endl;
	}
	
	    dimBlock.x = BLOCK_SIZE_X;
	    dimBlock.y = BLOCK_SIZE_Y;
	    
	  /* ************************************************ */
	  /* Calculamos Matriz de Distancias entre partículas */
	  /* ************************************************ */
	    /**Variables y memoria*/
	    width = cant_particles;
	    height = cant_particles;
	    dimGrid.x = ceil((double)width / (double)dimBlock.x);
	    dimGrid.y = ceil((double)height / (double)dimBlock.y);
	    
	    
 	    if(!periodicity){
		distances_kernel<<<dimGrid, dimBlock>>>(d_distance_r, d_distance_x, d_distance_y, d_distance_z,
							d_position_x, d_position_y, d_position_z, width, height);
 	      
	    } else {
	    /**Rellenamos datos**/
 		close_distances_kernel<<<dimGrid, dimBlock>>>(d_distance_x, d_distance_y, d_distance_z, d_distance_r,
							      d_position_x, d_position_y, d_position_z, 
							      h_box_x, h_box_y, h_box_z, width, height);
	      
 	    }
	    
	
	
	//TRAIGO AL HOST LAS DISTANCIAS PORQUE LAS VOY A NECESITAR PARA HACER EL CALCULO DE dEr  EN CPU
	if (CPU)
        cudaMemcpy(h_distance_r, d_distance_r, s_size, cudaMemcpyDeviceToHost);
	
	
	
	
	//if(results){
	    /** DEBUG **/
		/*cudaMemcpy(h_distance_r, d_distance_r, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_distance_x, d_distance_x, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_distance_y, d_distance_y, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_distance_z, d_distance_z, s_size, cudaMemcpyDeviceToHost);

		if (step %10000 == 0){
	
		out << " DISTANCES -  R" << endl << "  ";
        	 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_r[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;

            }*/
/*
	   out << " DISTANCES -  X" << endl << "  ";
                 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_x[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;

	   out << " DISTANCES -  Y" << endl << "  ";
                 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_y[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;
	

	   out << " DISTANCES -  Z" << endl << "  ";
                 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_z[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;
*/
	/*	double (*matriz)[cant_particles] = (double (*)[cant_particles]) h_distance_r;
		for(int i = 0; i<cant_particles; i+= cant_particles){
		  out << " " << i << "  | ";
		  for(int j = 0; j<cant_particles; j+= cant_particles){
		    out << matriz[i][j] << "\t";
		  }
		  out << endl << "  ";
		}
		out << endl;
	  */ 
	 /** DEBUG **/ 
	//}    
	    
  
  if(CPU)
	double (*h_dEr)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
  
  
	  /* ************************************************ */
	  /*              Calculamos Derivadas                */
	  /* ************************************************ */
	    /** Variables y memoria **/
	    width = cant_particles;
	    height = cant_particles;
	    dimGrid.x = ceil((double)width / (double)dimBlock.x);
	    dimGrid.y = ceil((double)height / (double)dimBlock.y);
	    
//	    derivative_E_r_analytic<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, cant_samples_r, d_EPS, d_SIG, width, height);
 	    if(analytic){
 		derivative_E_r_analytic<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, cant_samples_r, d_EPS, d_SIG, width, height);
		if(CPU)   //VERSION ANALITICA SOBRE CPU
		 derivative_E_r_analytic_MEMORY(h_dEr, h_distance_r, cut, h_item_particle, cant_samples_r, h_EPS, h_SIG, width, height); 
 //		if(amberResults){
// 		   /** Calculo la energia     E(r) para debug **/
		  E_r_analytic<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, cant_samples_r, d_EPS, d_SIG, width, height);
 		
 	   } else {
// 		/** Calculo de la derivada   dE(r)/dr  usando diferencias finitas **/
		if(derivative){
 		  //derivative_E_r_analytic<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, cant_samples_r, d_EPS, d_SIG, width, height);
		  if (CPU){
		    int x,y;
		   for (x=0;x<cant_particles;x++)
		     for(y=0;y<cant_particles;y++)
		      direct_derivative_E_r_MEMORY(h_dLJPot,h_dEr, h_distance_r,cut,h_item_particle, cant_samples_r,cant_types,width,height, x, y );
		 
		  //mando los resultados a gpu   
		  cudaMemcpy( d_dEr,h_dEr, s_size, cudaMemcpyHostToDevice);
		  } 
		    else
		     direct_derivative_E_r<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, cant_samples_r, cant_types, width, height);
		  E_r_analytic<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, cant_samples_r, d_EPS, d_SIG, width, height); 

		} else {
 	//	  derivative_E_r_analytic<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, cant_samples_r, d_EPS, d_SIG, width, height);
 		  derivative_E_r<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, cant_samples_r, cant_types, width, height);
		
// 		  if(amberResults){
// 		    /** Calculo la energia     E(r) para debug **/
		    E_r<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, cant_samples_r, cant_types, width, height);
 	//	  }
		}
// 		  
 	    }
	    
	   // if(amberResults){
		//if(!derivative){
		  /** DEBUG **/
		      //out << " Lennard-Jones" << endl << "  ";
		      double vdwaals = 0;
		      double (*h_Er)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
		      cudaMemcpy(h_Er, d_Er, s_size, cudaMemcpyDeviceToHost);
		      for(int i = 0; i<cant_particles; i++){
 		//	out << " " << i << "  | ";
			for(int j = 0; j<cant_particles; j++){
 		//	  out << h_Er[i][j] << "\t";
			  if(i<=j)
			      vdwaals += h_Er[i][j];
			}
 		//	out << endl << "  ";
		      }  
// 		      out << endl;
		if(step == 0)
			etotinicial= vdwaals;

		if(step % 10000 == 0){
			etotalX=vdwaals;
		//	out << " STEP = " << step  << endl;
		//	out << " VDWAALS = " << vdwaals << endl << endl;
			}
		      free(h_Er);
		  /** DEBUG **/
	//	}
	    //}
	    
	    
	    if(results){
		  /** DEBUG **/
		      out << " DERIVATIVES" << endl << "  ";
		      double (*h_dEr)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
		      cudaMemcpy(h_dEr, d_dEr, s_size, cudaMemcpyDeviceToHost);
		      for(int i = 0; i<cant_particles; i+= cant_particles/8){
			out << " " << i << "  | ";
			for(int j = 0; j<cant_particles; j+= cant_particles/8){
			  out << h_dEr[i][j] << "\t";
			}
			out << endl << "  ";
		      }
		      out << endl;
		      free(h_dEr);
		  /** DEBUG **/ 
	    }
	    if(results){
		  /** DEBUG **/
			cudaMemcpy(h_velocity_old_x, d_velocity_old_x, cant_particles_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_velocity_old_y, d_velocity_old_y, cant_particles_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_velocity_old_z, d_velocity_old_z, cant_particles_size, cudaMemcpyDeviceToHost);
			out << " OLD VELOCITIES" << endl;
			for(int i = 0; i<cant_particles; i++){
			  out << i+1 << ": (" << h_velocity_old_x[i] << " , " << h_velocity_old_y[i] << " , " << h_velocity_old_z[i] << ")" << endl;
			}
			out << endl;
		  /** DEBUG **/
	    }
	  





















	/* ************************************************ */
	  /*          Calculamos FUERZAS resultantes          */
	  /* ************************************************ */
	  /*   Fx =  dE(r) / dr  *  (x1-x2) / r               *
	  *   Fy =  dE(r) / dr  *  (y1-y2) / r               *
	  *   Fz =  dE(r) / dr  *  (z1-z2) / r               */
	    
	    /* Calculo de vectores parciales */
	    /**Variables y memoria*/
	    width = cant_particles;
	    height = cant_particles;
	    dimGrid.x = ceil((double)width / (double)dimBlock.x);
	    dimGrid.y = ceil((double)height / (double)dimBlock.y);
	    
	    /** Calculo del vector F **/
	    Parcial_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_x, d_dEr, d_distance_x, d_distance_r, width, height);
	    Parcial_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_y, d_dEr, d_distance_y, d_distance_r, width, height);
	    Parcial_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_z, d_dEr, d_distance_z, d_distance_r, width, height);

	    //if(results){
		/** DEBUG **/
		      /*double fuerzaTot=0;
			cudaMemcpy(h_Force_x, d_Force_x, s_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_y, d_Force_y, s_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_z, d_Force_z, s_size, cudaMemcpyDeviceToHost);
			  out << " FORCES" << endl << "  ";
			  for(int i = 0; i<cant_particles; i++){
			    for(int j = 0; j<cant_particles; j++){
				 if(i<=j)
					fuerzaTot+=h_Force_x[i][j] + h_Force_y[i][j] + h_Force_z[i][j];
			     
				out << h_Force_x[i][j] << "\n" << h_Force_y[i][j] << "\n" << h_Force_z[i][j] << "\n";
				// out << "(" << h_Force_x[i][j] << " , " << h_Force_y[i][j] << " , " << h_Force_z[i][j] << ")\t";
			    }
			    out << endl << "  ";
			  }
			  out << endl;
		*/
		/** DEBUG **/
	    //}
	    
//		 out << "LA SUMA TOTAL DE FUERZAS ES: " << fuerzaTot << endl; 
	    /* Calculo del vector F */
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    Resultant_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_x_resultant, d_Force_x, cant_particles);
	    Resultant_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_y_resultant, d_Force_y, cant_particles);
	    Resultant_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_z_resultant, d_Force_z, cant_particles);

//	   if(results){
		/** DEBUG **/

		      cudaMemcpy(h_Force_x_resultant, d_Force_x_resultant, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_y_resultant, d_Force_y_resultant, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_z_resultant, d_Force_z_resultant, cant_particles_size, cudaMemcpyDeviceToHost);
		      //out << " RESULTANT FORCES" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << h_Force_x_resultant[i] <<"\n" <<h_Force_y_resultant[i] << "\n"  << h_Force_z_resultant[i] <<  endl;

			//out << i+1 << ": (" << h_Force_x_resultant[i] << " , " << h_Force_y_resultant[i] << " , " << h_Force_z_resultant[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
//	    }
	    
	    
	    
	  /* ************************************************ */
	  /*       Calculamos VELOCIDADES Resultantes         */
	  /* ************************************************ */
	  /*  V(t + Dt/2) = V(t - Dt/2) +  [ F(t) * Dt ] / m  */  
	    
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    //out <<  "dtx= " << delta_tiempo*20.455 << endl;
	    /** Piso las velocidades acumuladas al tiempo t con las nuevas de t+Dt */
	    Resultant_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_x, d_velocity_old_x, d_Force_x_resultant, d_mass, d_item_particle, delta_tiempo, cant_particles);
	    Resultant_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_y, d_velocity_old_y, d_Force_y_resultant, d_mass, d_item_particle, delta_tiempo, cant_particles);
	    Resultant_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_z, d_velocity_old_z, d_Force_z_resultant, d_mass, d_item_particle, delta_tiempo, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_velocity_x, d_velocity_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_y, d_velocity_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_z, d_velocity_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " RESULTANT VELOCITIES" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << i+1 << ": (" << h_velocity_x[i] << " , " << h_velocity_y[i] << " , " << h_velocity_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	  /* ************************************************ */
	  /*        Calculamos POSICIONES Resultantes         */
	  /* ************************************************ */
	  /*       P(t + Dt) = P(t) +  V(t + Dt/2) * Dt       */
	  /* (TODO: ajustar condiciones de perioricidad       */
	    
	    /**Variables y memoria*/
	    
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	   


 
	    Resultant_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_x, d_velocity_x, delta_tiempo, cant_particles);
	    Resultant_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_y, d_velocity_y, delta_tiempo, cant_particles);
	    Resultant_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_z, d_velocity_z, delta_tiempo, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_position_x, d_position_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_y, d_position_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_z, d_position_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " RESULTANT POSITIONS" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << i+1 << ": (" << h_particle_type[i] << " (" << h_position_x[i] << " , " << h_position_y[i] << " , " << h_position_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }

	    
	    if(periodicity){
		    /* ************************************************ */
		    /*     Calculamos POSICIONES con PERIORICIDAD       */
		    /* ************************************************ */
		    /*       P(t + Dt) = P(t) +  V(t + Dt/2) * Dt       */
		      
		      /**Variables y memoria*/
		      dimBlock.x = 1024;
		      dimBlock.y = 1;
		      dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
		      dimGrid.y = 1;
		      
		      Adjustin_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_x, box_max_x, cant_particles);
		      Adjustin_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_y, box_max_y, cant_particles);
		      Adjustin_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_z, box_max_z, cant_particles);
	    }
	    if(coordinates){
		/** DEBUG **/
		      cudaMemcpy(h_position_x, d_position_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_y, d_position_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_z, d_position_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      if(results){
			      out << " RESULTANT POSITIONS in the CUBE" << endl;
			      for(int i = 0; i<cant_particles; i++){
				out << i+1 << ": (" << h_particle_type[i] << " (" << h_position_x[i] << " , " << h_position_y[i] << " , " << h_position_z[i] << ")" << endl;
			      }
			      out << endl;
		      }
		      for(int i = 0; i<cant_particles; i+=2){
			crd << "  " << h_position_x[i] << "  " << h_position_y[i] << "  " << h_position_z[i];
			if(i+1 < cant_particles){
			  crd << "  " << h_position_x[i+1] << "  " << h_position_y[i+1] << "  " << h_position_z[i+1] << endl;
			} else
			  crd << endl;
		      } 
		      
		/** DEBUG **/
	    }
	  

	  /* ************************************************ */
	  /*        Calculamos Ek de cada partícula           */
	  /* ************************************************ */
	  /* Ek = |vp|^2  *  m / 2        con vp = (vold+v)/2 */
	  /*            Ek_x = (v_x)^2  *  m / 2              */
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    /** Calculamos la energía cinética para las tres coordenadas de cada partícula      **/
	    /** Puede hacerse directamente así, sin calcular módulo por propiedades algebraicas **/
	    Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy_x, d_velocity_old_x, d_velocity_x, d_mass, d_item_particle, cant_particles);
	    Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy_y, d_velocity_old_y, d_velocity_y, d_mass, d_item_particle, cant_particles);
	    Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy_z, d_velocity_old_z, d_velocity_z, d_mass, d_item_particle, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_kinetic_energy_x, d_kinetic_energy_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_kinetic_energy_y, d_kinetic_energy_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_kinetic_energy_z, d_kinetic_energy_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " KINETIC ENERGY" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << " " << i << "  | ";
			out << i+1 << ": (" << h_kinetic_energy_x[i] << " , " << h_kinetic_energy_y[i] << " , " << h_kinetic_energy_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	  /* ************************************************ */
	  /*            Calculamos Ek Resultante              */
	  /* ************************************************ */
	  /*               Ek_TOT = sum (Ek_i)                */  
	    
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    /** Calculamos la Energía cinética total de cada partícula **/
	    Total_Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy, d_kinetic_energy_x, d_kinetic_energy_y, d_kinetic_energy_z, cant_particles);
	    
	    
	    /*  */
	    /** Calculamos la Energía cinética total del sistema **/
	    cudaMemcpy(h_kinetic_energy, d_kinetic_energy, cant_particles_size, cudaMemcpyDeviceToHost);
	    double Ek_TOT = 0;
	    for(int i = 0; i<cant_particles; i++){
		Ek_TOT += h_kinetic_energy[i];
	    }
	    
	    if(results){
		/** DEBUG **/
		      out << " KINETIC ENERGY" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << " " << i << "  | ";
			out << "  " << h_kinetic_energy[i] << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	    //if(amberResults){
	if(step==0)
		etotinicial=etotinicial + Ek_TOT;

	if (step %10000 == 0){
		etotalX=etotalX + Ek_TOT;
	      diferencia= etotalX - etotinicial;	
	      //out << " Total Kinetic Energy(t) = " << Ek_TOT << endl << endl;
	      //out << " Diferencia energia total= " << diferencia << endl; 	        	
		}


	//   }

	    
	  /* ************************************************ */
	  /*        Calculamos Temperatura Resultante         */
	  /* ************************************************ */
	  /*          T(t) = 2*Ek_TOT / (Kb*(3N-Nc))          */
	    
	    double Temp_TOT = Ek_TOT *  factor_conv_T_Ek;

	    //if(amberResults){
		/** DEBUG **/
		    if(step % 10000 == 0)
			out << " Temp(t) = " << Temp_TOT << endl << endl;
		/** DEBUG **/
	   // }
	    
	    
	  /* *********************************************** */
	  /*       Calculamos Factor de Correccion           */
	  /* *********************************************** */
	  /*   lambda = sqrt( 1 + 2 * dt / tautp * (T/T(t) -1) )   */
	    
	    double lambda = sqrt( 1 + delta_tiempo / tautp * (temp0/Temp_TOT -1)  );


	    if(amberResults){
		/** DEBUG **/
		    out << " lambda(t) = " << lambda << endl << endl;
		/** DEBUG **/
	    }
	    
	  /* ************************************************ */
	  /*        Calculamos Velocidades Corregidas         */
	  /* ************************************************ */
	  /*                vi = lambda * vi                  */
	    
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    /** Piso las velocidades acumuladas al tiempo t+Dt con las nuevas de t+Dt corregidas */
	    Corrected_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_old_x, d_velocity_x, lambda, cant_particles);
	    Corrected_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_old_y, d_velocity_y, lambda, cant_particles);
	    Corrected_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_old_z, d_velocity_z, lambda, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_velocity_x, d_velocity_old_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_y, d_velocity_old_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_z, d_velocity_old_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " CORRECTED RESULTANT VELOCITIES" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << i << ": (" << h_velocity_x[i] << " , " << h_velocity_y[i] << " , " << h_velocity_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	    dimBlock.x = BLOCK_SIZE_X;
	    dimBlock.y = BLOCK_SIZE_Y;  
	    
	    
	/* ********************************************************************************************************** */
	/* ******************************************* FIN Iteracion DM ********************************************* */
	/* ********************************************************************************************************** */
    }
    
    
    if(!results){    //timer mode ON
	gettimeofday(&tv2, NULL);
	taim << cut << " " << (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec) << endl;
    }
    
//     if(!analytic){
	/** Unbindeamos Textura y liberamos memoria **/
	    cudaUnbindTexture(texRef);
	    cudaFreeArray(cuLennard_i);
//     }
    if(results or amberResults){
      out.close();
    }
    if(coordinates){
      crd.close();
    }
  
  /* ************************************************ */
  /*         Liberamos memoria en Dispositivo         */
  /* ************************************************ */
    cudaFree(&d_item_particle);
    
    /** Positions **/
    cudaFree(&d_position_x);
    cudaFree(&d_position_y);
    cudaFree(&d_position_z);
    
    /** Distances **/
    cudaFree(&d_distance_x);
    cudaFree(&d_distance_y);
    cudaFree(&d_distance_z);
    cudaFree(&d_distance_r);
    
    /** Particle's mass **/
    cudaFree(d_mass);
    
    /** Velocities **/
    cudaFree(d_velocity_x);
    cudaFree(d_velocity_y);
    cudaFree(d_velocity_z);
    
    /** Derivatives **/
    cudaFree(&d_dEr);
   cudaFree(&d_Er); 
    /** Forces **/
    cudaFree(&d_Force_x);
    cudaFree(&d_Force_y);
    cudaFree(&d_Force_z);
    
    cudaFree(d_Force_x_resultant);
    cudaFree(d_Force_y_resultant);
    cudaFree(d_Force_z_resultant);
    
    /** Kinetic Energy **/
    cudaFree(d_kinetic_energy);
    cudaFree(d_kinetic_energy_x);
    cudaFree(d_kinetic_energy_y);
    cudaFree(d_kinetic_energy_z);
    
  /* ************************************************ */
  /*             Liberamos memoria en Host            */
  /* ************************************************ */  
    free(h_sigma);
    free(h_epsilon);
    free(h_mass);
    
    /** Matriz de Lennard Jones **/
    if(derivative)
      free(h_dLJPot);
    else
      free(h_LJPot);
    
    free(h_item_particle);
    
    /** Positions **/
    free(h_position_x);
    free(h_position_y);
    free(h_position_z);
    
    /** Distances **/
    free(h_distance_x);
    free(h_distance_y);
    free(h_distance_z);
    free(h_distance_r);
    
    /** Velocities **/
    free(h_velocity_x);
    free(h_velocity_y);
    free(h_velocity_z);
    
    /** Chargue **/
    free(h_chargue);
    
    /** Forces **/
    free(h_Force_x);
    free(h_Force_y);
    free(h_Force_z);
    
    free(h_Force_x_resultant);
    free(h_Force_y_resultant);
    free(h_Force_z_resultant);
    
    /** Kinetic Energy **/
    free(h_kinetic_energy);
    free(h_kinetic_energy_x);
    free(h_kinetic_energy_y);
    free(h_kinetic_energy_z);
    
  return 0;
}
