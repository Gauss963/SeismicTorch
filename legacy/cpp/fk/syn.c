/******************************************************************
   F-K: @(#) syn.c             1.1 9/08/2021
 
   Copyright (c) by L. Zhu
   See README file for copying and redistribution conditions.

 Synthesize the Green's functions by adding source radiation pattern

 Written by Lupei Zhu, 1996, seismo lab, Caltech

 Revision History:
   05/05/1996	Lupei Zhu	Initial coding.
   04/29/2000	Lupei Zhu	documatation.
   07/12/2000	Lupei Zhu	add component orientations in SAC.
   07/18/2000	Lupei Zhu	add -I option for integration.
   07/22/2000	Lupei Zhu	add option for convolving a trapezoid.
   03/29/2002	Lupei Zhu	add -P option for computing static disp.
   04/16/2004	Lupei Zhu	add -Mm for double-couple moment-tensor source.
   03/13/2006   Lupei Zhu	add -Me for explosion source.
   11/07/2006	Lupei Zhu	modify to input a general moment tensor source.
   11/01/2007	Lupei Zhu	add band-pass filtering (-F) options.
   05/15/2008   Lupei Zhu	correct a bug introduced when using a general MT.
   02/13/2012	Lupei Zhu	correct a bug of writing sac head info.
   10/25/2014	Lupei Zhu	add using MT greens functions (-T)
   03/29/2017	Lupei Zhu	save p and s wave take-off angles in synthetics.
   12/21/2019	Lupei Zhu	correct a bug in setting mt for explosion source.
   09/08/2021	Lupei Zhu	delete the -T option and set -A to be optional.
				delete the -P option by make -G optional.
				add zeta, chi and VTI inputs.
   02/25/2022	Lupei Zhu	handle explosion source only (nn=1).
   07/20/2022	Lupei Zhu	use trapezoid trap() in cap/
******************************************************************/
#include <stdio.h> 
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "sac.h"
#include "Complex.h"
#include "radiats.h"

int main(int argc, char **argv) {
  int	i,j,k,nn,ns,npt,error,intg,diff,src_type, filter;
  char	nam[128],outnm[128],*ccc,com[3]={'z','r','t'};
  float	coef,rad[6][3],m0,az,*grn,*syn[3],*src,*pt,disp[3],mt[3][3],vti[5];
  float cmpinc[3]={0,90.,90.}, cmpaz[3];
  float dt, dura, rise, tp, ts, t0, tmp, dist, shift, ap, as;
  float	*trap(float, float, float, int *);
  SACHEAD	hd;
#ifdef SAC_LIB
  char type[2] = {'B','P'}, proto[2] = {'B','U'};
  float	sn[30], sd[30];
  double f1, f2;
  int order, nsects;
#endif
  void  fttq_(float *, float *, int *, int *, float *);
  int	mftm=2048, nftm;
  float	tstar=0., ftm[2048];

  /* input parameters */
  int dynamic = 0;
  ns = 0;
  dura = 0.;
  src_type=0;
  intg=0;
  diff=0;
  filter=0;
  shift=0.;
  az = -12345.;
  error = 0;
  vti[0] = 0.;
  for (i=1; !error && i < argc; i++) {
     if (argv[i][0] == '-') {
	switch(argv[i][1]) {

 	   case 'A':
	      sscanf(&argv[i][2], "%f",&az);
	      if (az<0. || az>360.) error++;
	      cmpaz[0] = 0.;
	      cmpaz[1] = az;
	      cmpaz[2] = az+90.;
	      if (cmpaz[2] > 360.) cmpaz[2] -= 360.;
	      break;

 	   case 'D':
	      j = sscanf(&argv[i][2], "%f/%f",&dura,&rise);
	      if (j<2) rise = 0.5;
	      break;

#ifdef SAC_LIB
 	   case 'F':
	      filter = 1;
	      j = sscanf(&argv[i][2], "%lf/%lf/%d",&f1,&f2,&order);
	      if (j<3) order = 4;
	      break;
#endif

 	   case 'G':
	      dynamic = 1;
	      strcpy(nam,&argv[i][2]);
	      break;

	   case 'I':
	      intg = 1;
	      break;

	   case 'J':
	      diff = 1;
	      break;

 	   case 'M':
	      src_type = sscanf(&argv[i][2], "%f/%f/%f/%f/%f/%f/%f",&m0,&mt[0][0],&mt[0][1],&mt[0][2],&mt[1][1],&mt[1][2],&mt[2][2]);
	      break;

 	   case 'O':
	      strcpy(outnm, &argv[i][2]);
	      break;

	   case 'Q':
	      sscanf(&argv[i][2], "%f",&tstar);
	      break;

 	   case 'S':
	      if ( (src=read_sac(&argv[i][2],&hd)) != NULL ) {
                 ns = hd.npts;
		 shift = -hd.b;
	      }
	      break;

 	   case 'V':
	      j = sscanf(&argv[i][2], "%f/%f/%f/%f/%f",vti,vti+1,vti+2,vti+3,vti+4);
	      if (j<5) vti[4]=0.;
	      if (j<4) vti[3]=0.;
	      if (j<3) vti[2]=0.;
	      if (j<2) vti[1]=1.732;
	      vti[1] = vti[1]*vti[1]*vti[0];
	      vti[2] = (1+2*vti[2])*vti[1];
	      vti[3] = (1+2*vti[3])*vti[0];
	      vti[4] = sqrt(((1+2*vti[4])*vti[1]-vti[0])*(vti[1]-vti[0]))-vti[0];
	      break;

	   default:
	      error++;
	      break;
	}
     }
        else error++;
  }

  if(argc < 3 || error ) {
     fprintf(stderr,"Usage: %s -Mmag([[/Strike/Dip]/Rake[/ISO[/CLVD]]]|/Mxx/Mxy/Mxz/Myy/Myz/Mzz) -OoutName.z [-Aazimuth] ([-SsrcFunctionName] | -Ddura[/rise]]) [-Ff1/f2[/n]] ([-I] | [-J]) [-GFirstCompOfGF]\n\
   Compute displacements in cm in the up, radial (outward), and transverse (clockwise) directions produced by different seismic sources.\n\
   -M Specify source magnitude and geometry or moment-tensor.\n\
      For double couple, mag=Mw, strike/dip/rake are in A&R convention. Need GF .[0-9].\n\
      For explosion, mag=M0 in dyne-cm, no strike, dip, and rake. needed Need GF .[a-c].\n\
      For single force. mag is in dyne, only strike and dip are needed. Need GF .[0-5].\n\
      For moment-tensor, mag=M0 in dyne-cm, Mij components, x=N,y=E,z=Down.\n\
          or use strike/dip/rake/ISO/CLVD. Need GF .[0-9,a-c]\n\
   -O Output SAC file name.\n\
 Optional:\n\
   -A Set station azimuth in degree measured from the North.\n\
      If not given, 18-component MT Green functions .[a-r] are needed.\n\
   -D Specify the source time function as a trapezoid,\n\
      give the total duration (dt) and rise-time (0-0.5, default 0.5=triangle).\n\
   -F apply n-th order Butterworth band-pass filter, SAC lib required (off, n=4, must be < 10).\n\
   -G Give the name of the first component of the FK or MT Green functions.\n\
      If not given, input static displacement Green functions from stdin in the form:\n\
	distance Z45 R45 T45 ZDD RDD TDD ZSS RSS TSS [distance ZEX REX TEX].\n\
      The displacements will be output to stdout in the form:\n\
	distance azimuth z r t\n\
   -I Integration once.\n\
   -J Differentiate the synthetics.\n\
   -Q Convolve a Futterman Q operator of tstar (no).\n\
   -S Specify the SAC file name of the source time function (its sum. must be 1).\n\
   -V Compute potency tensor based on strike/dip/rake/ISO/CLVD and transform it to\n\
      moment tensor in VTI medium of mu/vpvs/epsilon/gamma/delta (0/1.732/0/0/0).\n\
   Examples:\n\
   * To compute three-component velocity at N33.5E azimuth from a Mw 4.5\n\
earthquake (strike 355, dip 80, rake -70), use:\n\
	syn -M4.5/355/80/-70 -D1 -A33.5 -OPAS.z -Ghk_15/50.fk.0\n\
   * To compute the static displacements from the same earthquake, use:\n\
	nawk \'$1==50\' st.out | syn -M4.5/355/80/-70 -A33.5\n\
   * To compute displacement from an explosion, use:\n\
   	syn -M3.3e20 -D1 -A33.5 -OPAS.z -Ghk_15/50.fk.a\n\
      or use an isotropic moment tensor:\n\
        syn -M3.3e20/1/0/0/1/0/1 -D1 -A33.5 -OPAS.z -Ghk_15/50.fk.0\n\
	\n",argv[0]);
    return -1;
  }

  if (dura>0.) {rise = dura*rise; dura = dura-rise;}

  if (src_type==3) {	// single force source
     nn = 2;
     sf_radiat(az-mt[0][0],mt[0][1],rad);
     m0 = m0*1.0e-15;
  } else {		// moment tensor sources (including EXP and DC)
     if (src_type>1&&src_type<7) m0 = pow(10.,1.5*m0+16.1);
     m0 = m0*1.0e-20;
     nn = 4;		// in the order of DD, DS, SS, and EXP
     if (src_type<6) mt[1][2] = 0.;	// default CLVD parameter chi=0
     if (src_type<5) mt[1][1] = 0.;	// default ISO parameter zeta=0
     if (src_type==1) nn=1; 		// explosion source
     if (src_type!=7) nmtensor(mt[1][1],mt[1][2],mt[0][0],mt[0][1],mt[0][2],vti,mt);
     if (az<0.) {
        nn = 6;		// 6 Mij components
        for(k=0,i=0;i<3;i++) for(j=i;j<3;j++,k++) rad[k][0]=rad[k][1]=rad[k][2]=mt[i][j];
     } else {
        mt_radiat(az,mt,rad);
     }
  }

  if (dynamic) ccc = strrchr(nam, (int) '.') + 1;
  for(j=0; j<3; j++) disp[j] = 0.;
  for(i=0; i<nn; i++) {
     for(j=0; j<3; j++) {
	coef = m0; if (nn>1) coef *= rad[i][j];
	if (!dynamic) {
	   if ((i==0||i==3) && j==0) scanf("%f",&dist);
	   scanf("%f",&tmp);
	   disp[j] += coef*tmp;
	   continue;
	}
        if ( (grn=read_sac(nam,&hd)) == NULL ) continue;
	if ( i==0 && j== 0 ) {
           npt = hd.npts;
	   dt = hd.delta;
	   t0 = hd.b;
	   tp = hd.t1; ts = hd.t2; ap = hd.user1; as = hd.user2;
  	   for(k=0; k<3; k++) syn[k]=(float *) calloc(npt, sizeof(float));
	   if (dura>0.) src = trap(dura,rise,dt,&ns);
	} else if (fabs(hd.b-t0)>0.5*dt || hd.npts < npt) {
	   fprintf(stderr,"t0 or npt in %s not agree with %f %d, skip\n",nam,t0,npt);
	   continue;
	}
        for(pt=syn[j],k=0;k<npt;k++,pt++) (*pt) += coef*grn[k];
	free(grn);
        (*ccc)++;
	if (*ccc == '9') (*ccc)+=40;	/* explosion components start at 'a' instead of '0' */
     }
  }

  if (!dynamic) {
     printf("%8.2f %8.2f %10.3e %10.3e %10.3e\n",dist,az,disp[0],disp[1],disp[2]);
     return 0;
  }

  /* convolve a source time function. integrate or filtering if needed */
#ifdef SAC_LIB
  if (filter) design(order, type, proto, 1., 1., f1, f2, (double) dt, sn, sd, &nsects);
#endif
  if (tstar>0.) fttq_(&dt,&tstar,&mftm,&nftm,ftm);
  for(j=0;j<3;j++) {
     if (intg) cumsum(syn[j],npt,dt);
     if (diff) diffrt(syn[j],npt,dt);
     if (ns > 0) conv(src,ns,syn[j],npt);
#ifdef SAC_LIB
     if (filter) apply(syn[j],npt,0,sn,sd,nsects);
#endif
     if (tstar>0.) conv(ftm,nftm,syn[j],npt);
  }

  /* output */
  ccc = outnm + strlen(outnm) - 1;
  if (ns == 0) strcat(outnm,"i");
  hd.npts = npt;
  hd.delta = dt;
  hd.b = t0-shift;
  hd.e = hd.b+(npt-1)*dt;
  hd.t1 = tp;
  hd.t2 = ts;
  hd.az = az;
  hd.user1 = ap; hd.user2 = as;
  for(j=0;j<3;j++) {
     *ccc = com[j];
     hd.cmpinc = cmpinc[j];
     hd.cmpaz  = cmpaz[j];
     write_sac(outnm,hd,syn[j]);
  }

  return 0;

}
