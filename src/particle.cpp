#include "particle.h"
#include "constants.h"

Particle::Particle(void) : position(0), radius(theParticleRad), velocity(0), temperature(0), 
	mass(theFuelMass), thermalMass(theFuelThermalMass), density(0), state(INERT), drag(theFuelDrag), conductivity(theFuelConductivity), sootAccumulated(0)
{
}

Particle::Particle(dvec3 startPos) : position(startPos*theCellSize), radius(theParticleRad), velocity(0), temperature(0), 
	mass(theFuelMass), thermalMass(theFuelThermalMass), density(0), state(INERT), drag(theFuelDrag), conductivity(theFuelConductivity), sootAccumulated(0)
{
}

Particle::Particle(Particle& p) : position(p.position), radius(p.radius), velocity(p.velocity), temperature(p.temperature), 
	mass(theSootMass), thermalMass(theSootThermalMass), density(0), state(SOOT), drag(theSootDrag), conductivity(theSootConductivity), sootAccumulated(0)
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

dvec3 Particle::getVelocity(){
	return velocity;
}

void Particle::setVelocity(dvec3 newVel){
	velocity = newVel;
}

double Particle::getMass(){
	return mass;
}

void Particle::setMass(double newMass){
	mass = newMass;
}

double Particle::getRadius(){
	return radius;
}

void Particle::setRadius(double newRad){
	radius = newRad;
}

double Particle::getThermalMass(){
	return thermalMass;
}

void Particle::setThermalMass(double newThermalMass){
	thermalMass = newThermalMass;
}

double Particle::getTemp(){
	return temperature;
}

void Particle::setTemp(double newTemp){
	temperature = newTemp;
}

int Particle::getState(){
	return state;
}

void Particle::setState(int newState){
	state = newState;
}

double Particle::getDrag(){
	return drag;
}
	
void Particle::setDrag(double newDrag){
	drag = newDrag;
}

double Particle::getConductivity(){
	return conductivity;
}
	
void Particle::setConductivity(double newConductivity){
	conductivity = newConductivity;
}

double Particle::getSoot(){
	return sootAccumulated;
}
	
void Particle::setSoot(double newSoot){
	sootAccumulated = newSoot;
}