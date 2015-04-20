#include "particle.h"
#include "constants.h"

Particle::Particle(void) : position(0), temperature(0), mass(0), thermalMass(0), density(0)
{
}

Particle::Particle(dvec3 startPos) : position(startPos*theCellSize), temperature(0), mass(0), thermalMass(0), density(0)
{
}

Particle::~Particle(void)
{
}

dvec3 Particle::getPos(){
	return position;
}

void Particle::setPos(dvec3 newPos){
	position = newPos;
}