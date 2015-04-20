#ifndef MACGrid_H_
#define MACGrid_H_

#pragma warning(disable: 4244 4267 4996)

#include "open_gl_headers.h" // PETER KUTZ.

#include <glm/glm.hpp>

#include "grid_data.h"
#include "grid_data_matrix.h" // PETER KUTZ.
#include "constants.h"
#include "particle.h"
#include <vector>
class Camera;

class MACGrid
{

public:
	MACGrid();
	~MACGrid();
	MACGrid(const MACGrid& orig);
	MACGrid& operator=(const MACGrid& orig);

	void reset();

	void draw(const Camera& c);
	void updateSources();
	void advectVelocity(double dt);
	void addExternalForces(double dt);
	void project(double dt);
	void advectTemperature(double dt);
	void advectDensity(double dt);
	void advectParticles(double dt);

protected:

	// Setup
	void initialize();

	// Simulation
	void computeBouyancy(double dt);
	void computeVorticityConfinement(double dt);

	// Rendering
	struct Cube { glm::dvec3 pos; glm::dvec4 color; double dist; };
	void drawWireGrid();
	void drawSmokeCubes(const Camera& c);
	void drawSmoke(const Camera& c);
	void drawCube(const MACGrid::Cube& c);
	void drawFace(const MACGrid::Cube& c);
	void drawVelocities();
	glm::dvec4 getRenderColor(int i, int j, int k);
	glm::dvec4 getRenderColor(const glm::dvec3& pt);
	void drawZSheets(bool backToFront);
	void drawXSheets(bool backToFront);
	void drawParticle(Particle& p);

	// GridData accessors
	enum Direction { X, Y, Z };
	glm::dvec3 getVelocity(const glm::dvec3& pt);
	double getTemperature(const glm::dvec3& pt);
	double getDensity(const glm::dvec3& pt);
	glm::dvec3 getCenter(int i, int j, int k);
	bool isValidCell(int i, int j, int k);

  // Sets up A matrix for calculation
	void calculateAMatrix();
  // sanity check
	bool checkDivergence();
  // Conjugate Gradient stuff
	bool conjugateGradient(const GridDataMatrix & A, GridData & p, const GridData & d, int maxIterations, double tolerance);
	double dotProduct(const GridData & vector1, const GridData & vector2);
	void add(const GridData & vector1, const GridData & vector2, GridData & result);
	void subtract(const GridData & vector1, const GridData & vector2, GridData & result);
	void multiply(const double scalar, const GridData & vector, GridData & result);
	double maxMagnitude(const GridData & vector);
	void apply(const GridDataMatrix & matrix, const GridData & vector, GridData & result);


  // TODO : Fill in the necessary data structures to maintain velocity, pressure
  // and density
	GridDataX mVx; // X component of velocity, stored on X faces
	GridDataY mVy; // Y component of velocity, stored on Y faces
	GridDataZ mVz; // Z component of velocity, stored on Z faces
	GridData mPressure;  // Pressure, stored at grid centers
	GridData mDensity;  // Density, stored at grid centers
	GridData mTemperature;  // Temperature, stored at grid centers
	GridDataMatrix AMatrix;

	std::vector<Particle> particles;
public:

	enum RenderMode { CUBES, SHEETS };
	static RenderMode theRenderMode;
	static bool theDisplayVel;
	
	void saveSmoke(const char* fileName);
};

#endif
