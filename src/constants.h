// Written by Peter Kutz.

#ifndef CONSTANTS_H
#define CONSTANTS_H

#ifndef __MINMAX_DEFINED
#  define max(a,b)    (((a) > (b)) ? (a) : (b))
#  define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif

#define INERT	  0
#define BURNING   1
#define SOOT      2

#define LERP(a,b,t) (1-t)*a + t*b



// Don't modify the values of these here.
// Modify the values of these in Constants.cpp instead.
extern const int theMillisecondsPerFrame;
extern const int theDim[3];
extern const double theCellSize;
extern const double theAirDensity;
extern const double theBuoyancyAlpha;
extern const double theBuoyancyBeta;	
extern const double theBuoyancyAmbientTemperature;
extern const double theVorticityEpsilon;

extern const double theParticleRad;
extern const double theFuelDrag;
extern const double theFuelMass;
extern const double theFuelThermalMass;
extern const double theFuelConductivity;

extern const double theSootMass;
extern const double theSootThermalMass;
extern const double theSootDrag;
extern const double theSootConductivity;

extern const double theCombustionIgnition;
extern const double theCombustionHeat;
extern const double theCombustionSoot;
extern const double theCombustionSootPoint;
extern const double theCombustionVolume;
extern const double theCombustionRate;





#endif