/*****************************************************************

Overall amn functions

******************************************************************/

/*      Overall Amn realated functions      */

/* Area Related Term */

#include<math.h>

#define ak(x1,z1,x2,z2,x3,z3) ((x3*(-z1+z2)+x2*(z1-z3)+x1*(-z2+z3))\
                                *(x3*(-z1+z2)+x2*(z1-z3)+x1*(-z2+z3)))

#define A001(x1,z1,x2,z2,x3,z3) ((x3*z2-x2*z3)*(x3*(z1+z2)+x1*(z2-z3)-x2*(z1+z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A002(x1,z1,x2,z2,x3,z3) ((x3*z1 - x1*z3)*(x3*(z1 + z2) + x2*(z1 - z3) - x1*(z2 + z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A003(x1,z1,x2,z2,x3,z3) ((x2*z1 - x1*z2)*(x3*(z1 - z2) + x2*(z1 + z3) - x1*(z2 + z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A004(x1,z1,x2,z2,x3,z3) -(4*(x3*z1 - x1*z3)*(x3*z2 - x2*z3))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A005(x1,z1,x2,z2,x3,z3) -(4*(x2*z1 - x1*z2)*(x3*z1 - x1*z3))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A006(x1,z1,x2,z2,x3,z3) -(4*(x2*z1 - x1*z2)*((-x3)*z2 + x2*z3))\
                                 /ak(x1,z1,x2,z2,x3,z3)

#define A011(x1,z1,x2,z2,x3,z3) -((x2-x3)*((-x3)*(z1+3*z2)+x1*(-z2+z3)+x2*(z1+3*z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)

#define A012(x1,z1,x2,z2,x3,z3) -((x1-x3)*((-x3)*(3*z1+z2)+x2*(-z1+z3)+x1*(z2+3*z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A013(x1,z1,x2,z2,x3,z3) -((x1-x2)*(x3*(-z1+z2)-x2*(3*z1+z3)+x1*(3*z2+z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A014(x1,z1,x2,z2,x3,z3) (-4*x2*(-2*x1*z3+x3*(z1+z3))+4*x3*(x3*(z1+z2)-x1*(z2+z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A015(x1,z1,x2,z2,x3,z3) (8*x2*x3*z1+4*x1*x1*(z2+z3)-4*x1*(x3*(z1+z2)+x2*(z1+z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A016(x1,z1,x2,z2,x3,z3) (8*x1*x3*z2+4*x2*x2*(z1+z3)-4*x2*(x3*(z1+z2)+x1*(z2+z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)

#define A021(x1,z1,x2,z2,x3,z3) (2*(x2-x3))*((x2-x3))/ak(x1,z1,x2,z2,x3,z3)

#define A022(x1,z1,x2,z2,x3,z3) (2*(x1-x3))*((x1-x3))/ak(x1,z1,x2,z2,x3,z3)

#define A023(x1,z1,x2,z2,x3,z3) (2*(x1-x2))*((x1-x2))/ak(x1,z1,x2,z2,x3,z3)

#define A024(x1,z1,x2,z2,x3,z3) (4*(x1-x3)*(-x2+x3))/ak(x1,z1,x2,z2,x3,z3)

#define A025(x1,z1,x2,z2,x3,z3) -(4*(x1-x2)*(x1-x3))/ak(x1,z1,x2,z2,x3,z3)

#define A026(x1,z1,x2,z2,x3,z3) (4*(x1-x2)*(x2-x3))/ak(x1,z1,x2,z2,x3,z3)



#define A101(x1,z1,x2,z2,x3,z3) ((z2-z3)*((-x3)*(z1+3*z2)+x1*(-z2+z3)+x2*(z1+3*z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A102(x1,z1,x2,z2,x3,z3) ((z1-z3)*((-x3)*(3*z1+z2)+x2*(-z1+z3)+x1*(z2+3*z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A103(x1,z1,x2,z2,x3,z3) -((z1-z2)*(x3*(z1-z2)+x2*(3*z1+z3)-x1*(3*z2+z3)))\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A104(x1,z1,x2,z2,x3,z3) (8*x3*z1*z2-4*((x2+x3)*z1+(x1+x3)*z2)*z3+4*(x1+x2)*z3*z3)\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A105(x1,z1,x2,z2,x3,z3) (4*z1*((x2+x3)*z1-(x1+x3)*z2)-4*((x1+x2)*z1-2*x1*z2)*z3)\
                                 /ak(x1,z1,x2,z2,x3,z3)
#define A106(x1,z1,x2,z2,x3,z3) (4*z2*(-((x2+x3)*z1)+(x1+x3)*z2)-4*(-2*x2*z1+(x1+x2)*z2)*z3)\
                                 /ak(x1,z1,x2,z2,x3,z3)


#define A111(x1,z1,x2,z2,x3,z3) -(4*(x2-x3)*(z2-z3))/ak(x1,z1,x2,z2,x3,z3)

#define A112(x1,z1,x2,z2,x3,z3) -(4*(-x1+x3)*(-z1+z3))/ak(x1,z1,x2,z2,x3,z3)

#define A113(x1,z1,x2,z2,x3,z3) (4*(-x1+x2)*(z1-z2))/ak(x1,z1,x2,z2,x3,z3)

#define A114(x1,z1,x2,z2,x3,z3) -(4*(x3*(z1+z2-2*z3)+x2*(-z1+z3)+x1*(-z2+z3)))\
                                  /ak(x1,z1,x2,z2,x3,z3)
#define A115(x1,z1,x2,z2,x3,z3) (4*((x1-x3)*(z1-z2)+(x1-x2)*(z1-z3)))\
                                  /ak(x1,z1,x2,z2,x3,z3)
#define A116(x1,z1,x2,z2,x3,z3) -(4*(x3*(-z1+z2)+x1*(z2-z3)+x2*(z1-2*z2+z3)))\
                                  /ak(x1,z1,x2,z2,x3,z3)



#define A121(x1,z1,x2,z2,x3,z3) 0.0

#define A122(x1,z1,x2,z2,x3,z3) 0.0

#define A123(x1,z1,x2,z2,x3,z3) 0.0

#define A124(x1,z1,x2,z2,x3,z3) 0.0

#define A125(x1,z1,x2,z2,x3,z3) 0.0

#define A126(x1,z1,x2,z2,x3,z3) 0.0



#define A201(x1,z1,x2,z2,x3,z3) (2*(z2-z3))*((z2-z3))/ak(x1,z1,x2,z2,x3,z3)

#define A202(x1,z1,x2,z2,x3,z3) (2*(z1-z3))*((z1-z3))/ak(x1,z1,x2,z2,x3,z3)

#define A203(x1,z1,x2,z2,x3,z3) (2*(z1-z2))*((z1-z2))/ak(x1,z1,x2,z2,x3,z3)

#define A204(x1,z1,x2,z2,x3,z3) (4*(z1-z3)*(-z2+z3))/ak(x1,z1,x2,z2,x3,z3)

#define A205(x1,z1,x2,z2,x3,z3) -(4*(z1-z2)*(z1-z3))/ak(x1,z1,x2,z2,x3,z3)

#define A206(x1,z1,x2,z2,x3,z3) (4*(z1-z2)*(z2-z3))/ak(x1,z1,x2,z2,x3,z3)


#define A211(x1,z1,x2,z2,x3,z3) 0.0

#define A212(x1,z1,x2,z2,x3,z3) 0.0

#define A213(x1,z1,x2,z2,x3,z3) 0.0

#define A214(x1,z1,x2,z2,x3,z3) 0.0

#define A215(x1,z1,x2,z2,x3,z3) 0.0

#define A216(x1,z1,x2,z2,x3,z3) 0.0

#define A221(x1,z1,x2,z2,x3,z3) 0.0

#define A222(x1,z1,x2,z2,x3,z3) 0.0

#define A223(x1,z1,x2,z2,x3,z3) 0.0

#define A224(x1,z1,x2,z2,x3,z3) 0.0

#define A225(x1,z1,x2,z2,x3,z3) 0.0

#define A226(x1,z1,x2,z2,x3,z3) 0.0



#define Ar(A,B,X) A*B+X+A*A*X
#define Lr(A,B,X)  ((X*X+(B+A*X)*(B+A*X))==0. ? 1. : (X*X+(B+A*X)*(B+A*X)))
#define Br(A,B,X) (fabs(A*X+B)<1.0e-10 ? 0. : A*X+B)


#define F01(A,B,X) -X+(B*(M_PI/2.-atan2(B,Ar(A,B,X))))/(1.+A*A)\
                            +(1./2.)*((A*B)/(1.+A*A)+X)*log(Lr(A,B,X))

#define F02(A,B,X) (X*(B+2.*A*A*B+A*(1.+A*A)*X))/(2.*(1.+A*A))+\
                   ((-1.+A*A)*B*B*atan2(B,Ar(A,B,X)))/(2.*(1+A*A)*(1+A*A))\
                    -(1./2.)*X*X*atan2(Br(A,B,X),X)+(A*B*B*log(Lr(A,B,X)))/(2.*(1.+A*A)*(1.+A*A))

#define F03(A,B,X) (X*(3.*(1.+8.*A*A+3.*A*A*A*A)*B*B+3.*A*(2.+5.*A*A+3.*A*A*A*A)*B*\
                   X+(1.+A*A)*(1.+A*A)*(2.+3.*A*A)*X*X))\
      /(18.*(1.+A*A)*(1.+A*A))+((1.-3.*A*A)*B*B*B*(M_PI/2.-atan2(B,Ar(A,B,X))))/(3.*(1.+A*A)*(1.+A*A)*(1.+A*A))+\
      (-((A*(-3.+ A*A)*B*B*B)/(6.*(1.+A*A)*(1.+A*A)*(1.+A*A)))-X*X*X/6.)*log(Lr(A,B,X))

#define F10(A,B,X) (A*B*atan2(B,Ar(A,B,X)))/(1.+A*A)+X*atan2(Br(A,B,X),X)\
        +(B*log(Lr(A,B,X)))/(2.+2.*A*A)

#define F11(A,B,X) (1./4.)*((2.*A*B)/(1.+A*A)-X)*X-(A*B*B*(M_PI/2.-atan2(B,Ar(A,B,X))))/((1.+A*A)*(1.+A*A))\
        +(1./4.)*(-(((-1.+A*A)*B*B)/((1.+A*A)*(1.+A*A)))+X*X)*log(Lr(A,B,X))

#define F12(A,B,X) (X*(4*A*B*B+(1+A*A)*(2+3*A*A)*B*X+2*A*(1+A*A)*(1+A*A)*X*X))\
        /(6*(1+A*A)*(1+A*A))-(A*(-3+A*A)*B*B*B*atan2(B,Ar(A,B,X)))/(3*(1+A*A)*(1+A*A)*(1+A*A))\
        -(1./3.)*X*X*X*atan2(Br(A,B,X),X)+((1-3*A*A)*B*B*B*log(Lr(A,B,X)))/(6*(1+A*A)*(1+A*A)*(1+A*A))

#define F13(A,B,X) 0.

#define F20(A,B,X) (B*X)/(2+2*A*A)-((-1+A*A)*B*B*atan2(B,Ar(A,B,X)))/(2*(1+A*A)*(1+A*A))\
        +(1./2.)*X*X*atan2(Br(A,B,X),X)-(A*B*B*log(Lr(A,B,X)))/(2*(1+A*A)*(1+A*A))

#define F21(A,B,X) (1./18.)*X*(-((6*(-1+A*A)*B*B)/((1+A*A)*(1+A*A)))+(3*A*B*X)/(1+A*A)-2*X*X)\
        +((-1+3*A*A)*B*B*B*(M_PI/2-atan2(B,Ar(A,B,X))))/(3*(1+A*A)*(1+A*A)*(1+A*A))+\
        (1./6.)*((A*(-3+A*A)*B*B*B)/((1+A*A)*(1+A*A)*(1+A*A))+X*X*X)*log(Lr(A,B,X))

#define F22(A,B,X) 0.

#define F23(A,B,X) 0.

#define F30(A,B,X) (B*X*(-4*A*B+X+A*A*X))/(6*(1+A*A)*(1+A*A)) \
        + (A*(-3+A*A)*B*B*B*atan2(B,Ar(A, B, X)))/(3*(1+A*A)*(1+A*A)*(1+A*A))\
        + (1./3.)*X*X*X*atan2(Br(A, B, X),X)\
        +((-1+3*A*A)*B*B*B*log(Lr(A,B,X)))/(6*(1+A*A)*(1+A*A)*(1+A*A))

#define F31(A,B,X) 0.

#define F32(A,B,X) 0.

#define F33(A,B,X) 0.


