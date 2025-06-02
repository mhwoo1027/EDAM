/*
 Analytical Triangle Integral Core function as Python - Extension
     Main purpose of this code is to speed up the FEM integration,
     so, some of code are written explicitly to speed up
*/

#include<math.h>
#include<stdio.h>
#include"trig_func.h"

#define A(m,n,k) (A##m##n##k)

#define F(m,n)     ((F##m##n(A32,B32,X2t))-(F##m##n(A31,B31,X2t))-((F##m##n(A32,B32,X3t))-(F##m##n(A31,B31,X3t)))\
                   +(F##m##n(A21,B21,X1t))-(F##m##n(A31,B31,X1t))-((F##m##n(A21,B21,X2t))-(F##m##n(A31,B31,X2t))))

#define swap(x, y, T) do { T swap = x; x = y; y = swap;} while (0)

double trig_intg(int *,double *,double *,double *,double *);
double trig_intg(int * mele,double * mcod,double * mval,double * mprm,double *temp)
{
    int ii,jj,m,n,k;
    int Nele,Ncod;
    double amn[3][3][7];
    double Fmn[4][4];
    double Fv[7];
    double tmp1,tmp2;
    double sign;
    double res;

    double nx,nz;
    double X0,Z0,X1,Z1,X2,Z2,X3,Z3;
    double X1t,Z1t,X2t,Z2t,X3t,Z3t;
    double A32,A31,A21,B32,B31,B21;

    double eps=1.0e-8;

    /* Some neccessary initialization */
    nx=mprm[0];
    nz=mprm[1];
    X0=mprm[2];
    Z0=mprm[3];
    Nele=(int)mprm[4];
    Ncod=(int)mprm[5];

   /* For every iith mesh element - it changes with ii*6 */
    res=0;
    for(ii=0;ii<Nele/6;ii++)
    {
         /* Three shifted (X,Z) coordinates */
         X1=mcod[(mele[6*ii]-1)*2]-X0;
         Z1=mcod[(mele[6*ii]-1)*2+1]-Z0;
         X2=mcod[(mele[6*ii+1]-1)*2]-X0;
         Z2=mcod[(mele[6*ii+1]-1)*2+1]-Z0;
         X3=mcod[(mele[6*ii+2]-1)*2]-X0;
         Z3=mcod[(mele[6*ii+2]-1)*2+1]-Z0;

         /* Get F value for triangle*/
         Fv[0]=0.0;
         Fv[1]=mval[mele[6*ii]-1];
         Fv[2]=mval[mele[6*ii+1]-1];
         Fv[3]=mval[mele[6*ii+2]-1];
         Fv[4]=mval[mele[6*ii+3]-1];
         Fv[5]=mval[mele[6*ii+4]-1];
         Fv[6]=mval[mele[6*ii+5]-1];

         /* Get amn function */
         amn[0][0][0]=0.;
         amn[0][0][1]=A001(X1,Z1,X2,Z2,X3,Z3);
         amn[0][0][2]=A002(X1,Z1,X2,Z2,X3,Z3);
         amn[0][0][3]=A003(X1,Z1,X2,Z2,X3,Z3);
         amn[0][0][4]=A004(X1,Z1,X2,Z2,X3,Z3);
         amn[0][0][5]=A005(X1,Z1,X2,Z2,X3,Z3);
         amn[0][0][6]=A006(X1,Z1,X2,Z2,X3,Z3);

         amn[0][1][0]=0.;
         amn[0][1][1]=A011(X1,Z1,X2,Z2,X3,Z3);
         amn[0][1][2]=A012(X1,Z1,X2,Z2,X3,Z3);
         amn[0][1][3]=A013(X1,Z1,X2,Z2,X3,Z3);
         amn[0][1][4]=A014(X1,Z1,X2,Z2,X3,Z3);
         amn[0][1][5]=A015(X1,Z1,X2,Z2,X3,Z3);
         amn[0][1][6]=A016(X1,Z1,X2,Z2,X3,Z3);

         amn[0][2][0]=0.;
         amn[0][2][1]=A021(X1,Z1,X2,Z2,X3,Z3);
         amn[0][2][2]=A022(X1,Z1,X2,Z2,X3,Z3);
         amn[0][2][3]=A023(X1,Z1,X2,Z2,X3,Z3);
         amn[0][2][4]=A024(X1,Z1,X2,Z2,X3,Z3);
         amn[0][2][5]=A025(X1,Z1,X2,Z2,X3,Z3);
         amn[0][2][6]=A026(X1,Z1,X2,Z2,X3,Z3);

         amn[1][0][0]=0.;
         amn[1][0][1]=A101(X1,Z1,X2,Z2,X3,Z3);
         amn[1][0][2]=A102(X1,Z1,X2,Z2,X3,Z3);
         amn[1][0][3]=A103(X1,Z1,X2,Z2,X3,Z3);
         amn[1][0][4]=A104(X1,Z1,X2,Z2,X3,Z3);
         amn[1][0][5]=A105(X1,Z1,X2,Z2,X3,Z3);
         amn[1][0][6]=A106(X1,Z1,X2,Z2,X3,Z3);

         amn[1][1][0]=0.;
         amn[1][1][1]=A111(X1,Z1,X2,Z2,X3,Z3);
         amn[1][1][2]=A112(X1,Z1,X2,Z2,X3,Z3);
         amn[1][1][3]=A113(X1,Z1,X2,Z2,X3,Z3);
         amn[1][1][4]=A114(X1,Z1,X2,Z2,X3,Z3);
         amn[1][1][5]=A115(X1,Z1,X2,Z2,X3,Z3);
         amn[1][1][6]=A116(X1,Z1,X2,Z2,X3,Z3);

         amn[1][2][0]=0.;
         amn[1][2][1]=0.;
         amn[1][2][2]=0.;
         amn[1][2][3]=0.;
         amn[1][2][4]=0.;
         amn[1][2][5]=0.;
         amn[1][2][6]=0.;

         amn[2][0][0]=0.;
         amn[2][0][1]=A201(X1,Z1,X2,Z2,X3,Z3);
         amn[2][0][2]=A202(X1,Z1,X2,Z2,X3,Z3);
         amn[2][0][3]=A203(X1,Z1,X2,Z2,X3,Z3);
         amn[2][0][4]=A204(X1,Z1,X2,Z2,X3,Z3);
         amn[2][0][5]=A205(X1,Z1,X2,Z2,X3,Z3);
         amn[2][0][6]=A206(X1,Z1,X2,Z2,X3,Z3);

         amn[2][1][0]=0.;
         amn[2][1][1]=0.;
         amn[2][1][2]=0.;
         amn[2][1][3]=0.;
         amn[2][1][4]=0.;
         amn[2][1][5]=0.;
         amn[2][1][6]=0.;

         amn[2][2][0]=0.;
         amn[2][2][1]=0.;
         amn[2][2][2]=0.;
         amn[2][2][3]=0.;
         amn[2][2][4]=0.;
         amn[2][2][5]=0.;
         amn[2][2][6]=0.;


         /* Get Fmn function */

         /* Exchange Variable to Make it X1t > X2t > X3t */
         X1t=X1;
         X2t=X2;
         X3t=X3;
         Z1t=Z1;
         Z2t=Z2;
         Z3t=Z3;

         if(X3t>X2t)
             {swap(X3t,X2t,double);
              swap(Z3t,Z2t,double);}
         if(X2t>X1t)
             {swap(X1t,X2t,double);
              swap(Z1t,Z2t,double);}
         if(X3t>X2t)
             {swap(X3t,X2t,double);
              swap(Z3t,Z2t,double);}

         A32=(Z3t-Z2t)/(X3t-X2t);
         A31=(Z3t-Z1t)/(X3t-X1t);
         A21=(Z2t-Z1t)/(X2t-X1t);

         B32=Z2t-(X2t*(Z3t-Z2t)/(X3t-X2t));
         B31=Z1t-(X1t*(Z3t-Z1t)/(X3t-X1t));
         B21=Z1t-(X1t*(Z2t-Z1t)/(X2t-X1t));


         if(Z2t-(A31*X2t+B31)>0) 
             sign=1.0;
         else
             sign=-1.0;

         Fmn[0][0]=0.0;
         Fmn[0][1]=sign*F(0,1);
         Fmn[0][2]=sign*F(0,2);
         Fmn[0][3]=sign*F(0,3);
         Fmn[1][0]=sign*F(1,0);
         Fmn[1][1]=sign*F(1,1);
         Fmn[1][2]=sign*F(1,2);
         Fmn[1][3]=sign*F(1,3);
         Fmn[2][0]=sign*F(2,0);
         Fmn[2][1]=sign*F(2,1);
         Fmn[2][2]=sign*F(2,2);
         Fmn[2][3]=sign*F(2,3);
         Fmn[3][0]=sign*F(3,0);
         Fmn[3][1]=sign*F(3,1);
         Fmn[3][2]=sign*F(3,2);
         Fmn[3][3]=sign*F(3,3);

         tmp1=0;
         for(jj=0;jj<7;jj++)
            {tmp2=0;
             for(m=0;m<3;m++)
                 for(n=0;n<3;n++)
                     {
                      tmp2=amn[m][n][jj]*(nx*Fmn[m+1][n]+nz*Fmn[m][n+1])+tmp2;
                      }
              tmp1=tmp2*Fv[jj]+tmp1;
             }
         res=res+tmp1;
         temp[ii]=tmp1;

    }/* End of Every Mesh elements */

    return -res/(2*M_PI); /*Final Integral include -1/2pi */

}

