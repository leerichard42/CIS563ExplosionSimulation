// Written by Peter Kutz.

#include "constants.h"

const int theMillisecondsPerFrame = 10;

const int theDim[3] = {30, 30, 5};

const double theCellSize = 0.5;

const double theAirDensity = 1.0;

const double theBuoyancyAlpha = 0.08; // Gravity's effect on the smoke particles.
const double theBuoyancyBeta = 0.37; // Buoyancy's effect due to temperature difference.	
const double theBuoyancyAmbientTemperature = 0.0; // Ambient temperature.

const double theVorticityEpsilon = 0.10;

const double theParticleRad = 0.1;
const double theFuelDrag = 750.0;
const double theFuelMass = 4;
const double theFuelConductivity = 200;
const double theFuelThermalMass = 5;

const double theSootMass = 1;
const double theSootThermalMass = 5;
const double theSootDrag = 210;
const double theSootConductivity = 100;

const double theCombustionIgnition = 10;
const double theCombustionHeat = 50;
const double theCombustionSoot = 1;
const double theCombustionSootPoint = .5;
const double theCombustionVolume = 0.1;
const double theCombustionRate = 0.5;