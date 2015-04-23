#ifndef Particle_H_
#define Particle_H_
#include <glm\glm.hpp>

using namespace glm;

class Particle
{
public:
	Particle(void);
	Particle(dvec3 startPos);
	Particle(Particle& p);
	~Particle(void);
	dvec3 getPos();
	void setPos(dvec3 newPos);

	dvec3 getVelocity();
	void setVelocity(dvec3 newVel);

	double getMass();
	void setMass(double newMass);

	double getThermalMass();
	void setThermalMass(double newThermalMass);

	double getTemp();
	void setTemp(double newTemp);

	double getRadius();
	void setRadius(double newRad);

	double getDrag();
	void setDrag(double newDrag);

	double getConductivity();
	void setConductivity(double newConductivity);

	double getSoot();
	void setSoot(double newSoot);

	int getState();
	void setState(int newState);

protected:
	dvec3 position;
	dvec3 velocity;
	double radius;
	double temperature;
	double mass;
	double thermalMass;
	double density;
	double drag;
	double conductivity;
	double sootAccumulated;
	int state;
};

#endif