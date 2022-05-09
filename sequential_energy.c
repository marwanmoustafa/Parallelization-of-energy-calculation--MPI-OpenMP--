// waters.c
#include <stdio.h>
#include <math.h>
#include <time.h> // for CPU time
#include <sys/time.h> //for gettimeofday
#include "mpi.h"
#define LENGTH 100
// global variables
const int maxnum=100000;
double r[maxnum][3][3],rcutsq=1.44,L;
// r(number of molecule, atom 0=O,1=H,2=H, coordinate 0=x,1=y,2=z)

double sqr(double a){return a*a;}

double energy12(int i1,int i2){
// ============================
  int m,n,xyz;
  double shift[3],dr[3],mn[3],r6,distsq,dist,ene=0;
  const double sig=0.3166,eps=0.65,eps0=8.85e-12,e=1.602e-19,Na=6.022e23,q[3]={-0.8476,0.4238,0.4238};
  double elst,sig6;
  elst=e*e/(4*3.141593*eps0*1e-9)*Na/1e3,sig6=pow(sig,6);

  // periodic boundary conditions
  for(xyz=0;xyz<=2;xyz++){
   dr[xyz]=r[i1][0][xyz]-r[i2][0][xyz];shift[xyz]=-L*floor(dr[xyz]/L+.5); //round dr[xyz]/L to nearest integer
   dr[xyz]=dr[xyz]+shift[xyz];
  }
  distsq=sqr(dr[0])+sqr(dr[1])+sqr(dr[2]);
  if(distsq<rcutsq){ // calculate energy if within cutoff
    r6=sig6/pow(distsq,3);
    ene=4*eps*r6*(r6-1.); // LJ energy
    for(m=0;m<=2;m++){
      for(n=0;n<=2;n++){
        for(xyz=0;xyz<=2;xyz++) mn[xyz]=r[i1][m][xyz]-r[i2][n][xyz]+shift[xyz];
        dist=sqrt(sqr(mn[0])+sqr(mn[1])+sqr(mn[2]));
        ene=ene+elst*q[m]*q[n]/dist;
    } }
  }
  return ene;
}

main(int argc, char *argv[]) {
double counter=0;
double totalenergy;
int me,nproc;
int x=1;
MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD,&nproc);
MPI_Comm_rank(MPI_COMM_WORLD,&me);
  int i,j,natoms,nmol;
  double energy=0,dtime1,dtime2,dtime3,mint1,maxt1,mint2,maxt2,mint3,maxt3;
int total_matrix;
total_matrix=maxnum*3*3;
double  maxcounter=0;
double mincounter=0;
double loadbalance;
  FILE *fp;
  char line[LENGTH],nothing[LENGTH],name[20];
clock_t cputime1; /* clock_t defined in <time.h> and <sys/types.h> as int */
clock_t cputime2;
clock_t cputime3;
 struct timeval start,middle,end;

 cputime1 = clock();  
gettimeofday(&start, NULL);
for(int i=0;i<nproc;i++)
{
if(me==i){
 // printf("Program to calculate energy of water\n");
 // printf("Input NAME of configuration file\n");
 // scanf("%s",name); // reading of filename from keyboard 
  fp=fopen("100k.gro", "r"); //opening of file and beginning of reading from HDD
  fgets(line, LENGTH,fp); //skip first line
  fgets(line, LENGTH,fp); sscanf(line,"%i",&natoms);
  nmol=natoms/3;// printf("Number of molecules %i\n",nmol);
  for (i=0;i<nmol;i++){
    for(j=0;j<=2;j++){
      fgets(line, LENGTH,fp);
      sscanf(line, "%s %s %s %lf %lf %lf",nothing,nothing,nothing, &r[i][j][0],&r[i][j][1],&r[i][j][2]);

  } }
 // printf("first line %lf %lf %lf\n",r[0][0][0],r[0][0][1],r[0][0][2]);
  fscanf(fp, "%lf",&L); // read box size
 // printf("Box size %lf\n",L);
fclose(fp);
}
}
MPI_Barrier(MPI_COMM_WORLD);

//MPI_Bcast(&nmol,1,MPI_INT,0,MPI_COMM_WORLD);
//MPI_Bcast(&r,nmol*9,MPI_DOUBLE,0,MPI_COMM_WORLD);
//MPI_Bcast(&L,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
//MPI_Bcast(&mincounter,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
//MPI_Bcast(&maxcounter,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

gettimeofday(&middle, NULL);
  cputime2 = clock();    // assign initial CPU time (IN CPU CLOCKS)
   // returns structure with time in s and us (microseconds)
 // for(i=(me*nmol)/nproc;i<(me+1)*nmol/nproc;i=i++){
  for(i=me;i<nmol-1;i=i+nproc) {
     for(j=i+1;j<nmol;j++){
 energy=energy+energy12(i,j);
counter++;
}
 }
 cputime3 = clock();  
MPI_Allreduce(&energy,&totalenergy,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); 
MPI_Allreduce(&counter,&mincounter,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
MPI_Allreduce(&counter,&maxcounter,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  cputime3= clock();      // calculate  cpu clock time as difference of times after-before
  gettimeofday(&end, NULL);
  dtime3 = ((end.tv_sec  - start.tv_sec)+(end.tv_usec - start.tv_usec)/1e6);
  dtime1 = ((middle.tv_sec  - start.tv_sec)+(middle.tv_usec - start.tv_usec)/1e6);
  dtime2 = ((end.tv_sec  - middle.tv_sec)+(end.tv_usec - middle.tv_usec)/1e6);

printf("thread number:%i its counter is %lf t1 %lf t2  %lf T3 is %lf \n",me,counter,dtime1,dtime2,dtime3);
MPI_Allreduce(&dtime1,&mint1,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
MPI_Allreduce(&dtime2,&maxt2,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
MPI_Allreduce(&dtime2,&mint2,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
MPI_Allreduce(&dtime1,&maxt1,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
MPI_Allreduce(&dtime3,&mint3,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
MPI_Allreduce(&dtime3,&maxt3,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

 if(me==0)
{
  printf("Total energy %lf \n",totalenergy);
  printf("Energy per molecule %lf \n",totalenergy/nmol);
printf("the min load blance is %lf the max is %lf\n",mincounter,maxcounter);
  loadbalance=2.0*(maxcounter-mincounter)/(0.+maxcounter+mincounter);
  printf("load imbalance =%lf\n",loadbalance);
printf("the min T1 is %lf the max T1 is %lf\n",mint1,maxt1);
printf("the min T2 is %lf the max T2 is %lf\n",mint2,maxt2);
printf("the min T3 is %lf the max T3 is %lf\n",mint3,maxt3);


}
MPI_Finalize();
}










