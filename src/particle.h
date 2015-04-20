#ifndef Particle_H_
#define Particle_H_
#include <glm\glm.hpp>

using namespace glm;

class Particle
{
public:
	Particle(void);
	Particle(dvec3 startPos);
	~Particle(void);
	dvec3 getPos();
	void setPos(dvec3 newPos);

protected:
	dvec3 position;
	double temperature;
	double mass;
	double thermalMass;
	double density;
};

#endif