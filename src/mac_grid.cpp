#include "mac_grid.h"
#include "open_gl_headers.h"
#include "camera.h"
#include "custom_output.h"
#include "constants.h"
#include "particle.h"

#include <math.h>
#include <map>
#include <stdio.h>
#include <vector>

#undef max
#undef min
#include <fstream>


// Globals
MACGrid target;


// NOTE: x -> cols, z -> rows, y -> stacks
MACGrid::RenderMode MACGrid::theRenderMode = SHEETS;
bool MACGrid::theDisplayVel = false;//true

#define FOR_EACH_CELL \
   for(int k = 0; k < theDim[MACGrid::Z]; k++)  \
      for(int j = 0; j < theDim[MACGrid::Y]; j++) \
         for(int i = 0; i < theDim[MACGrid::X]; i++) 

#define FOR_EACH_CELL_REVERSE \
   for(int k = theDim[MACGrid::Z] - 1; k >= 0; k--)  \
      for(int j = theDim[MACGrid::Y] - 1; j >= 0; j--) \
         for(int i = theDim[MACGrid::X] - 1; i >= 0; i--) 

#define FOR_EACH_FACE \
   for(int k = 0; k < theDim[MACGrid::Z]+1; k++) \
      for(int j = 0; j < theDim[MACGrid::Y]+1; j++) \
         for(int i = 0; i < theDim[MACGrid::X]+1; i++) 

#define FOR_EACH_FACE_X \
   for(int k = 0; k < theDim[MACGrid::Z]; k++) \
      for(int j = 0; j < theDim[MACGrid::Y]; j++) \
         for(int i = 0; i < theDim[MACGrid::X]+1; i++) 

#define FOR_EACH_FACE_Y \
   for(int k = 0; k < theDim[MACGrid::Z]; k++) \
      for(int j = 0; j < theDim[MACGrid::Y]+1; j++) \
         for(int i = 0; i < theDim[MACGrid::X]; i++) 

#define FOR_EACH_FACE_Z \
   for(int k = 0; k < theDim[MACGrid::Z]+1; k++) \
      for(int j = 0; j < theDim[MACGrid::Y]; j++) \
         for(int i = 0; i < theDim[MACGrid::X]; i++) 

MACGrid::MACGrid()
{
   initialize();
}

MACGrid::MACGrid(const MACGrid& orig)
{
  // TODO : Copy constructor for MAC Grid 
}

MACGrid& MACGrid::operator=(const MACGrid& orig)
{
  // TODO : Copy constructor for MAC Grid
  MACGrid m;
  return m;
}

MACGrid::~MACGrid()
{
}



void MACGrid::reset()
{
  // TODO : Initialize the MAC Grid.
	mVx.initialize();
	mVy.initialize();
	mVz.initialize();
	
	mPressure.initialize();
	mDensity.initialize();
	mTemperature.initialize(0.0);
	calculateAMatrix();

	//particles.push_back(Particle(dvec3(5,5,0)));

	for(int i = 0; i < 15; i++){
		particles.push_back(Particle(dvec3(10 + ((double)std::rand() / RAND_MAX),10 + ((double)std::rand() / RAND_MAX),0)));
	}
}

void MACGrid::initialize()
{
   reset();
}

void MACGrid::updateSources()
{
  // TODO: Set initial values for density, temperature, velocity
	//mVx(1,12,0) = 100;
	//mDensity(0,12,0) = 100.0;
	////mVy(0,2,0) = 0.5;
	////FOR_EACH_FACE_Y { mVy(i,j,k) = 1.0;}
	////mDensity(2,2,2) = 10.0;
	//mTemperature(10,5,0) = 50;
	//mTemperature(20,22,0) = 50;

	mVx(10,20,0) = 5;
	mVx(20,10,0) = -5;
	
	//mDensity(10,10,0) = 10.0;
}

void MACGrid::advectVelocity(double dt)
{

	target.mVx = mVx;
	target.mVy = mVy;
	target.mVz = mVz;
	double halfCell = theCellSize/2;
	FOR_EACH_FACE {
		//get face center points
		glm::dvec3 pt = getCenter(i,j,k);
		/*glm::dvec3 ptx = mVx.worldToSelf(pt);
		glm::dvec3 pty = mVy.worldToSelf(pt);
		glm::dvec3 ptz = mVz.worldToSelf(pt);
		*/
		glm::dvec3 ptx = glm::dvec3(i*theCellSize, j*theCellSize+halfCell, k*theCellSize+halfCell);
		glm::dvec3 pty = glm::dvec3(i*theCellSize+halfCell, j*theCellSize, k*theCellSize+halfCell);
		glm::dvec3 ptz = glm::dvec3(i*theCellSize+halfCell, j*theCellSize+halfCell, k*theCellSize);
		
		//use initial velocities
		glm::dvec3 velx = getVelocity(ptx);
		glm::dvec3 vely = getVelocity(pty);
		glm::dvec3 velz = getVelocity(ptz);

		//euler approximation
		glm::dvec3 eulerx = ptx-dt*velx;
		glm::dvec3 eulery = pty-dt*vely;
		glm::dvec3 eulerz = ptz-dt*velz;
		
		//get new velocities from our euler step
		glm::dvec3 new_velocity = glm::dvec3(getVelocity(eulerx)[0], getVelocity(eulery)[1], getVelocity(eulerz)[2]);	

		//check boundaries and set our new velocities
		target.mVx(i,j,k) = (i == 0 || i == theDim[X]-1) ? 0 : new_velocity[0];
		target.mVy(i,j,k) = (j == 0 || j == theDim[Y]-1) ? 0 : new_velocity[1];
		target.mVz(i,j,k) = (k == 0 || k == theDim[Z]-1) ? 0 : new_velocity[2];
		
	}
    mVx = target.mVx;
    mVy = target.mVy;
    mVz = target.mVz;
}

void MACGrid::advectTemperature(double dt)
{
    // TODO: Calculate new temp and store in target
	target.mTemperature = mTemperature;

	FOR_EACH_CELL {
		glm::dvec3 pt = getCenter(i,j,k);
		glm::dvec3 velocity = getVelocity(pt);
		glm::dvec3 euler = pt - dt * velocity;
		double new_temperature = getTemperature(euler);
		target.mTemperature(i,j,k) = new_temperature;
	}
	mTemperature = target.mTemperature;
}

void MACGrid::advectDensity(double dt)
{
    // TODO: Calculate new densitities and store in target
	target.mDensity = mDensity;
	FOR_EACH_CELL {
		glm::dvec3 pt = getCenter(i,j,k);
		glm::dvec3 velocity = getVelocity(pt);
		glm::dvec3 euler = pt - dt * velocity;
		double new_density = getDensity(euler);
		target.mDensity(i,j,k) = new_density;
	}
	mDensity = target.mDensity;
}

void MACGrid::advectParticles(double dt){
	for(int i = 0; i < particles.size(); i++){
		dvec3 pt = particles[i].getPos();
		dvec3 velocity = getVelocity(pt);
		dvec3 newPos = pt + dt * velocity;

		if(newPos.x < 0){
			newPos.x = 0;
			velocity.x = 0;
		}
		if(newPos.x > theDim[0] * theCellSize){
			newPos.x = theDim[0] * theCellSize;
			velocity.x = 0;
		}
		if(newPos.y < 0){
			newPos.y = 0;
			velocity.y = 0;
		}
		if(newPos.y > theDim[1] * theCellSize){
			newPos.y = theDim[1] * theCellSize;
			velocity.y = 0;
		}
		if(newPos.z < 0){
			newPos.z = 0;
			velocity.z = 0;
		}
		if(newPos.z > theDim[2] * theCellSize){
			newPos.z = theDim[2] * theCellSize;
			velocity.z = 0;
		}


		particles[i].setPos(newPos);
		//printf("DIM: (%d, %d, %d)\n", theDim[0], theDim[1], theDim[2]);
		//printf("POS: (%f, %f, %f)\n", pt.x, pt.y, pt.z);
		//printf("VELOCITY: (%f, %f, %f)\n", velocity.x, velocity.y, velocity.z);
		//printf("NEWPOS: (%f, %f, %f)\n", newPos.x, newPos.y, newPos.z);
	}
}

void MACGrid::computeBouyancy(double dt)
{
  	// TODO: Calculate bouyancy and store in target
	target.mVy = mVy;
	double mass = 1.0;
	FOR_EACH_FACE_Y {
		if (j == 0 || j == theDim[Y]) continue;
		glm::dvec3 pt = mVy.worldToSelf(getCenter(i,j,k));
		double temperature = getTemperature(pt);
		double density = getDensity(pt);
		double force = -theBuoyancyAlpha * density + theBuoyancyBeta * (temperature - theBuoyancyAmbientTemperature);
		//if (i == 0 || j == 12 || k == 0) printf("%f\n", force);
		target.mVy(i,j,k) = mVy(i,j,k) + dt*force;
	}
}

void MACGrid::computeVorticityConfinement(double dt)
{
   // TODO: Calculate vorticity confinement forces
   // Apply the forces to the current velocity and store the result in target

	//calculate vorticity values
	int dataSize = theDim[X]*theDim[Y]*theDim[Z];
	GridData vorticity[3] = {GridDataX(), GridDataY(), GridDataZ()};
	for (int i = 0; i < 3; i++) vorticity[i].data().resize(dataSize);
	FOR_EACH_CELL {
		//getting gradients
		glm::dvec3 iPrev = getCenter(i-1,j,k), iNext(i+1,j,k), jPrev(i,j-1,k), jNext(i,j+1,k), kPrev(i,j,k-1), kNext(i,j,k+1);
		glm::dvec3 viNext = getVelocity(iNext), viPrev = getVelocity(iPrev), 
				   vjNext = getVelocity(jNext), vjPrev = getVelocity(jPrev), 
				   vkNext = getVelocity(kNext), vkPrev = getVelocity(kPrev);
		double x1 = vkNext[2] - vkPrev[2];
		double x2 = vjNext[1] - vjPrev[1];
		double y1 = viNext[0] - viPrev[0];
		double y2 = vkNext[2] - vkPrev[2];
		double z1 = vjNext[1] - vjPrev[1];
		double z2 = viNext[0] - viPrev[0];

		vorticity[0](i,j,k) = (x1 - x2)/theCellSize;
		vorticity[1](i,j,k) = (y1 - y2)/theCellSize;
		vorticity[2](i,j,k) = (z1 - z2)/theCellSize;
	}
	//vorticity confinement
	GridData confinement[3] = {GridData(), GridData(), GridData()};
	for (int i = 0; i < 3; i++) {
		confinement[i].data().resize(dataSize);
		confinement[i].initialize();
	}
	FOR_EACH_CELL {
		if (i == 0 || j == 0 || k == 0) continue;
		if (i == theDim[X]-1 || j == theDim[Y]-1 || k == theDim[Z]-1) continue;
		//calculate differentials to derive gradient
		double d1, d2;
		glm::dvec3 gradient;
		d1 = glm::dvec3(vorticity[0](i+1,j,k), vorticity[1](i+1,j,k), vorticity[2](i+1,j,k)).length();
		d2 = glm::dvec3(vorticity[0](i-1,j,k), vorticity[1](i-1,j,k), vorticity[2](i-1,j,k)).length();
		gradient[0] = (d1 - d2) / (2.0 * theCellSize);

		d1 = glm::dvec3(vorticity[0](i,j+1,k), vorticity[1](i,j+1,k), vorticity[2](i,j+1,k)).length();
		d2 = glm::dvec3(vorticity[0](i,j-1,k), vorticity[1](i,j-1,k), vorticity[2](i,j-1,k)).length();
		gradient[1] = (d1 - d2) / (2.0 * theCellSize);

		d1 = glm::dvec3(vorticity[0](i,j,k+1), vorticity[1](i,j,k+1), vorticity[2](i,j,k+1)).length();
		d2 = glm::dvec3(vorticity[0](i,j,k-1), vorticity[1](i,j,k-1), vorticity[2](i,j,k-1)).length();
		gradient[2] = (d1 - d2) / (2.0 * theCellSize);
		//compute N
		glm::dvec3 N = gradient / (gradient.length() + 10e-20); //prevent divide by zero
		glm::dvec3 _vorticity = glm::dvec3(vorticity[0](i,j,k), vorticity[1](i,j,k), vorticity[2](i,j,k));
		glm::dvec3 _confinement = theVorticityEpsilon * theCellSize * glm::cross(_vorticity, N);
		confinement[0](i,j,k) = _confinement[X];
		confinement[1](i,j,k) = _confinement[Y];
		confinement[2](i,j,k) = _confinement[Z];
	}
	FOR_EACH_FACE_X {
		if (i == 0 || i == theDim[X]) continue;
		target.mVx(i,j,k) += (confinement[0](i,j,k) + confinement[0](i-1,j,k))/2.0f;
	}
	FOR_EACH_FACE_Y {
		if (j == 0 || j == theDim[Y]) continue;
		target.mVy(i,j,k) += (confinement[1](i,j,k) + confinement[1](i,j-1,k))/2.0f;
	}
	FOR_EACH_FACE_Z {
		if (k == 0 || k == theDim[X]) continue;
		target.mVz(i,j,k) += (confinement[2](i,j,k) + confinement[2](i,j,k-1))/2.0f;
	}
	mVx = target.mVx;
	mVy = target.mVy;
	mVz = target.mVz;
}

void MACGrid::addExternalForces(double dt)
{
   computeBouyancy(dt);
   computeVorticityConfinement(dt);
}

void MACGrid::project(double dt)
{
   // TODO: Solve Ax = b for pressure
   // 1. Contruct b
   // 2. Construct A 
   // 3. Solve for p
   // Subtract pressure from our velocity and save in target
   // Then save the result to our object
	double rho = 1.0;

	// 1. Construct b
	GridData b;
	b.initialize();

	double density = rho;
	double mVxltiplier = -(((theCellSize*theCellSize)*density)/dt);
	//get div matrix
	FOR_EACH_CELL {
		glm::dvec3 pX(i,j,k);
		double divX, divY, divZ;
		//divergence of x
		if (i+1 == theDim[MACGrid::X]) divX = -mVx(i,j,k) / theCellSize; 
		else if ( i == 0)              divX = mVx(i+1,j,k) / theCellSize;
		else                           divX = (mVx(i+1,j,k) - mVx(i,j,k)) / theCellSize;

		//divergence of y
		if (j+1 == theDim[MACGrid::Y]) divY = -mVy(i,j,k) / theCellSize;
		else if (j == 0 )              divY = mVy(i,j+1,k) / theCellSize;
		else                           divY = (mVy(i,j+1,k) - mVy(i,j,k)) / theCellSize; 

		//divergence of z
		if (k+1 == theDim[MACGrid::Z]) divZ = mVz(i,j,k) / theCellSize;
		else if (k == 0)               divZ = mVz(i,j,k+1) / theCellSize;
		else                           divZ = (mVz(i,j,k+1) - mVz(i,j,k)) / theCellSize; 

		//update b
		double div = mVxltiplier*(divX + divY + divZ);
		b(i, j, k) = div;
	}


	// 2. Construct A
	GridDataMatrix A = this->AMatrix;

	// 3. Solve for p - store in target.mPressure (pressure)
	int iterations = 400;
	double tolerance = 0.01;

	conjugateGradient(A, target.mPressure, b, iterations, tolerance);

	mPressure = target.mPressure;
	target.mVx = mVx;
	target.mVy = mVy;
	target.mVz = mVz;

	double m = dt/rho;

	// X FACES
	FOR_EACH_FACE_X {  	
		if (i == 0 || i == theDim[X]-1) {
			target.mVx(i,j,k) = 0.0;
			continue;
		}
		double dP = (m * (mPressure(i,j,k) - mPressure(i-1,j,k)))/theCellSize;
		target.mVx(i,j,k) = mVx(i,j,k) - dP;
	}
	// Y FACES
	FOR_EACH_FACE_Y {
		if (i == 0 || i == theDim[Y]-1) {
			target.mVy(i,j,k) = 0.0;
			continue;
		}
		double dP = (m * (mPressure(i,j,k) - mPressure(i,j-1,k)))/theCellSize;
		target.mVy(i,j,k) = mVy(i,j,k) - dP;
	}
	//Z FACES
	FOR_EACH_FACE_Z {
		if (i == 0 || i == theDim[Z]-1) {
			target.mVz(i,j,k) = 0.0;
			continue;
		}
		double dP = (m * (mPressure(i,j,k) - mPressure(i,j,k-1)))/theCellSize;
		target.mVz(i,j,k) = mVz(i,j,k) - dP;	
	}

	mPressure = target.mPressure;
	mVx = target.mVx;
	mVy = target.mVy;
	mVz = target.mVz;
   return;
}

bool MACGrid::checkDivergence() {
	double sum = 0;
	FOR_EACH_FACE_X {
		sum += mVx(i,j,k);
	}
	FOR_EACH_FACE_Y {
		sum += mVy(i,j,k);
	}
	FOR_EACH_FACE_Z {
		sum += mVz(i,j,k);
	}
	if (abs(sum) > 10e-6) {
		return false;
	}
	return true;
}

//accessors

glm::dvec3 MACGrid::getVelocity(const glm::dvec3& pt) {
  // TODO : Given a point in space, give the 3D velocity field at the point
	glm::dvec3 vel;
	vel[0] = mVx.interpolate(pt);
	vel[1] = mVy.interpolate(pt);
	vel[2] = mVz.interpolate(pt);
	return vel;
}

double MACGrid::getTemperature(const glm::dvec3& pt) {
	return mTemperature.interpolate(pt);
}

double MACGrid::getDensity(const glm::dvec3& pt) {
	return mDensity.interpolate(pt);
}

glm::dvec3 MACGrid::getCenter(int i, int j, int k)
{
   double xstart = theCellSize/2.0;
   double ystart = theCellSize/2.0;
   double zstart = theCellSize/2.0;

   double x = xstart + i*theCellSize;
   double y = ystart + j*theCellSize;
   double z = zstart + k*theCellSize;
   return glm::dvec3(x, y, z);
}

bool MACGrid::isValidCell(int i, int j, int k)
{
	if (i >= theDim[MACGrid::X] || j >= theDim[MACGrid::Y] || k >= theDim[MACGrid::Z]) {
		return false;
	}

	if (i < 0 || j < 0 || k < 0) {
		return false;
	}

	return true;
}

void MACGrid::calculateAMatrix() {

	FOR_EACH_CELL {

		int numFluidNeighbors = 0;
		if (i-1 >= 0) {
			AMatrix.plusI(i-1,j,k) = -1;
			numFluidNeighbors++;
		}
		if (i+1 < theDim[MACGrid::X]) {
			AMatrix.plusI(i,j,k) = -1;
			numFluidNeighbors++;
		}
		if (j-1 >= 0) {
			AMatrix.plusJ(i,j-1,k) = -1;
			numFluidNeighbors++;
		}
		if (j+1 < theDim[MACGrid::Y]) {
			AMatrix.plusJ(i,j,k) = -1;
			numFluidNeighbors++;
		}
		if (k-1 >= 0) {
			AMatrix.plusK(i,j,k-1) = -1;
			numFluidNeighbors++;
		}
		if (k+1 < theDim[MACGrid::Z]) {
			AMatrix.plusK(i,j,k) = -1;
			numFluidNeighbors++;
		}
		// Set the diagonal:
		AMatrix.diag(i,j,k) = numFluidNeighbors;
	}
}

bool MACGrid::conjugateGradient(const GridDataMatrix & A, GridData & p, const GridData & d, int maxIterations, double tolerance) {
	// Solves Ap = d for p.

	FOR_EACH_CELL {
		p(i,j,k) = 0.0; // Initial guess p = 0.	
	}

	GridData r = d; // Residual vector.

	GridData z; z.initialize();
  // TODO : Apply preconditioner; for now, bypass the preconditioner
  z = r;

	GridData s = z; // Search vector;

	double sigma = dotProduct(z, r);

	for (int iteration = 0; iteration < maxIterations; iteration++) {

		double rho = sigma; // According to Aline. Here???

		apply(A, s, z); // z = applyA(s);

		double alpha = rho/dotProduct(z, s);

		GridData alphaTimesS; alphaTimesS.initialize();
		multiply(alpha, s, alphaTimesS);
		add(p, alphaTimesS, p);

		GridData alphaTimesZ; alphaTimesZ.initialize();
		multiply(alpha, z, alphaTimesZ);
		subtract(r, alphaTimesZ, r);

		if (maxMagnitude(r) <= tolerance) {
			//PRINT_LINE("PCG converged in " << (iteration + 1) << " iterations.");
			return true; //return p;
		}


    // TODO : Apply preconditioner; for now, bypass the preconditioner
		z = r;		
		double sigmaNew = dotProduct(z, r);
		double beta = sigmaNew / rho;

		GridData betaTimesS; betaTimesS.initialize();
		multiply(beta, s, betaTimesS);
		add(z, betaTimesS, s);

		sigma = sigmaNew;
	}

	PRINT_LINE( "PCG didn't converge!" );
	return false;

}

double MACGrid::dotProduct(const GridData & vector1, const GridData & vector2) {
	
	double result = 0.0;

	FOR_EACH_CELL {
		result += vector1(i,j,k) * vector2(i,j,k);
	}

	return result;
}

void MACGrid::add(const GridData & vector1, const GridData & vector2, GridData & result) {
	
	FOR_EACH_CELL {
		result(i,j,k) = vector1(i,j,k) + vector2(i,j,k);
	}

}

void MACGrid::subtract(const GridData & vector1, const GridData & vector2, GridData & result) {
	
	FOR_EACH_CELL {
		result(i,j,k) = vector1(i,j,k) - vector2(i,j,k);
	}

}

void MACGrid::multiply(const double scalar, const GridData & vector, GridData & result) {
	
	FOR_EACH_CELL {
		result(i,j,k) = scalar * vector(i,j,k);
	}

}

double MACGrid::maxMagnitude(const GridData & vector) {
	
	double result = 0.0;

	FOR_EACH_CELL {
		if (abs(vector(i,j,k)) > result) result = abs(vector(i,j,k));
	}

	return result;
}

void MACGrid::apply(const GridDataMatrix & matrix, const GridData & vector, GridData & result) {
	
	FOR_EACH_CELL { // For each row of the matrix.

		double diag = 0;
		double plusI = 0;
		double plusJ = 0;
		double plusK = 0;
		double minusI = 0;
		double minusJ = 0;
		double minusK = 0;

		diag = matrix.diag(i,j,k) * vector(i,j,k);
		if (isValidCell(i+1,j,k)) plusI = matrix.plusI(i,j,k) * vector(i+1,j,k);
		if (isValidCell(i,j+1,k)) plusJ = matrix.plusJ(i,j,k) * vector(i,j+1,k);
		if (isValidCell(i,j,k+1)) plusK = matrix.plusK(i,j,k) * vector(i,j,k+1);
		if (isValidCell(i-1,j,k)) minusI = matrix.plusI(i-1,j,k) * vector(i-1,j,k);
		if (isValidCell(i,j-1,k)) minusJ = matrix.plusJ(i,j-1,k) * vector(i,j-1,k);
		if (isValidCell(i,j,k-1)) minusK = matrix.plusK(i,j,k-1) * vector(i,j,k-1);

		result(i,j,k) = diag + plusI + plusJ + plusK + minusI + minusJ + minusK;
	}

}

void MACGrid::saveSmoke(const char* fileName) {
	std::ofstream fileOut(fileName);
	if (fileOut.is_open()) {
		FOR_EACH_CELL {
			fileOut << mDensity(i,j,k) <<std::endl;
		}
		fileOut.close();
	}
}

void MACGrid::draw(const Camera& c)
{   
   drawWireGrid();
   if (theDisplayVel) drawVelocities();   
   if (theRenderMode == CUBES) drawSmokeCubes(c);
   else drawSmoke(c);

   for(int i = 0; i < particles.size(); i++){
	   drawParticle(particles[i]);
   }
}

void MACGrid::drawVelocities()
{
   // draw line at each center
	//printf("0 1 0 = %f  0 2 0 = %f\n", mVy(0,1,0), mVy(0,2,0));
   glBegin(GL_LINES);
      FOR_EACH_CELL
      {
         glm::dvec3 pos = getCenter(i,j,k);
         glm::dvec3 vel = getVelocity(pos);
         if (glm::length(vel) > 0.0001)
         {
           //vel.Normalize(); // PETER KUTZ.
           vel *= theCellSize/2.0;
           vel += pos;
		   glColor4f(1.0, 1.0, 0.0, 1.0);

           GLdouble doublePos[3];
           doublePos[0] = pos.x, doublePos[1] = pos.y, doublePos[2] = pos.z;
           glVertex3dv(doublePos);
		       
           GLdouble doubleVel[3];
           glColor4f(1.0 * mTemperature(i,j,k)/10, 0.0 + (1 - mTemperature(i,j,k)/10), 0.0, 1.0);
           doubleVel[0] = vel.x, doubleVel[1] = vel.y, doubleVel[2] = vel.z;
           glVertex3dv(doubleVel);
         }
      }
   glEnd();
}

glm::dvec4 MACGrid::getRenderColor(int i, int j, int k)
{
  // TODO : get density value and display as alpha value
	double value = mDensity(i,j,k);
	return glm::dvec4(1.0, 1.0, 1.0, value);
}

glm::dvec4 MACGrid::getRenderColor(const glm::dvec3& pt)
{
	// TODO : get desnity value and display as alpha value
	double value = getDensity(pt);
	return glm::dvec4(1.0, 1.0 , 1.0, value);
}

void MACGrid::drawZSheets(bool backToFront)
{
   // Draw K Sheets from back to front
   double back =  (theDim[2])*theCellSize;
   double top  =  (theDim[1])*theCellSize;
   double right = (theDim[0])*theCellSize;
  
   double stepsize = theCellSize*0.25;

   double startk = back - stepsize;
   double endk = 0;
   double stepk = -theCellSize;

   if (!backToFront)
   {
      startk = 0;
      endk = back;   
      stepk = theCellSize;
   }

   for (double k = startk; backToFront? k > endk : k < endk; k += stepk)
   {
     for (double j = 0.0; j < top; )
      {
         glBegin(GL_QUAD_STRIP);
         for (double i = 0.0; i <= right; i += stepsize)
         {
            glm::dvec3 pos1 = glm::dvec3(i,j,k); 
            glm::dvec3 pos2 = glm::dvec3(i, j+stepsize, k); 

            glm::dvec4 color1 = getRenderColor(pos1);
            glm::dvec4 color2 = getRenderColor(pos2);

            glColor4dv(glm::value_ptr(color1));
            glVertex3dv(glm::value_ptr(pos1));

            glColor4dv(glm::value_ptr(color2));
            glVertex3dv(glm::value_ptr(pos2));
         } 
         glEnd();
         j+=stepsize;

         glBegin(GL_QUAD_STRIP);
         for (double i = right; i >= 0.0; i -= stepsize)
         {
            glm::dvec3 pos1 = glm::dvec3(i,j,k); 
            glm::dvec3 pos2 = glm::dvec3(i, j+stepsize, k); 

            glm::dvec4 color1 = getRenderColor(pos1);
            glm::dvec4 color2 = getRenderColor(pos2);

            glColor4dv(glm::value_ptr(color1));
            glVertex3dv(glm::value_ptr(pos1));

            glColor4dv(glm::value_ptr(color2));
            glVertex3dv(glm::value_ptr(pos2));
         } 
         glEnd();
         j+=stepsize;
      }
   }
}

void MACGrid::drawXSheets(bool backToFront)
{
   // Draw K Sheets from back to front
   double back =  (theDim[2])*theCellSize;
   double top  =  (theDim[1])*theCellSize;
   double right = (theDim[0])*theCellSize;
  
   double stepsize = theCellSize*0.25;

   double starti = right - stepsize;
   double endi = 0;
   double stepi = -theCellSize;

   if (!backToFront)
   {
      starti = 0;
      endi = right;   
      stepi = theCellSize;
   }

   for (double i = starti; backToFront? i > endi : i < endi; i += stepi)
   {
     for (double j = 0.0; j < top; )
      {
         glBegin(GL_QUAD_STRIP);
         for (double k = 0.0; k <= back; k += stepsize)
         {
            glm::dvec3 pos1 = glm::dvec3(i,j,k); 
            glm::dvec3 pos2 = glm::dvec3(i, j+stepsize, k); 

            glm::dvec4 color1 = getRenderColor(pos1);
            glm::dvec4 color2 = getRenderColor(pos2);

            glColor4dv(glm::value_ptr(color1));
            glVertex3dv(glm::value_ptr(pos1));

            glColor4dv(glm::value_ptr(color2));
            glVertex3dv(glm::value_ptr(pos2));
         } 
         glEnd();
         j+=stepsize;

         glBegin(GL_QUAD_STRIP);
         for (double k = back; k >= 0.0; k -= stepsize)
         {
            glm::dvec3 pos1 = glm::dvec3(i,j,k); 
            glm::dvec3 pos2 = glm::dvec3(i, j+stepsize, k); 

            glm::dvec4 color1 = getRenderColor(pos1);
            glm::dvec4 color2 = getRenderColor(pos2);

            glColor4dv(glm::value_ptr(color1));
            glVertex3dv(glm::value_ptr(pos1));

            glColor4dv(glm::value_ptr(color2));
            glVertex3dv(glm::value_ptr(pos2));
         } 
         glEnd();
         j+=stepsize;
      }
   }
}


void MACGrid::drawSmoke(const Camera& c)
{
   glm::dvec3 eyeDir = c.getBackward();
   double zresult = fabs(glm::dot(eyeDir, glm::dvec3(1,0,0)));
   double xresult = fabs(glm::dot(eyeDir, glm::dvec3(0,0,1)));
   //double yresult = fabs(Dot(eyeDir, vec3(0,1,0)));

   if (zresult < xresult)
   {      
      drawZSheets(c.getPosition()[2] < 0);
   }
   else 
   {
      drawXSheets(c.getPosition()[0] < 0);
   }
}

void MACGrid::drawSmokeCubes(const Camera& c)
{
   std::multimap<double, MACGrid::Cube, std::greater<double> > cubes;
   FOR_EACH_CELL
   {
      MACGrid::Cube cube;
      cube.color = getRenderColor(i,j,k);
      cube.pos = getCenter(i,j,k);
      cube.dist = glm::length((cube.pos - c.getPosition()));
      cubes.insert(std::make_pair(cube.dist, cube));
   } 

   // Draw cubes from back to front
   std::multimap<double, MACGrid::Cube, std::greater<double> >::const_iterator it;
   for (it = cubes.begin(); it != cubes.end(); ++it)
   {
      drawCube(it->second);
   }
}

void MACGrid::drawWireGrid()
{
   // Display grid in light grey, draw top & bottom

   double xstart = 0.0;
   double ystart = 0.0;
   double zstart = 0.0;
   double xend = theDim[0]*theCellSize;
   double yend = theDim[1]*theCellSize;
   double zend = theDim[2]*theCellSize;

   glPushAttrib(GL_LIGHTING_BIT | GL_LINE_BIT);
      glDisable(GL_LIGHTING);
      glColor3f(0.25, 0.25, 0.25);

      glBegin(GL_LINES);
      for (int i = 0; i <= theDim[0]; i++)
      {
         double x = xstart + i*theCellSize;
         glVertex3d(x, ystart, zstart);
         glVertex3d(x, ystart, zend);

         glVertex3d(x, yend, zstart);
         glVertex3d(x, yend, zend);
      }

      for (int i = 0; i <= theDim[2]; i++)
      {
         double z = zstart + i*theCellSize;
         glVertex3d(xstart, ystart, z);
         glVertex3d(xend, ystart, z);

         glVertex3d(xstart, yend, z);
         glVertex3d(xend, yend, z);
      }

      glVertex3d(xstart, ystart, zstart);
      glVertex3d(xstart, yend, zstart);

      glVertex3d(xend, ystart, zstart);
      glVertex3d(xend, yend, zstart);

      glVertex3d(xstart, ystart, zend);
      glVertex3d(xstart, yend, zend);

      glVertex3d(xend, ystart, zend);
      glVertex3d(xend, yend, zend);
      glEnd();
   glPopAttrib();

   glEnd();
}

#define LEN 0.5
void MACGrid::drawFace(const MACGrid::Cube& cube)
{
   glColor4dv(glm::value_ptr(cube.color));
   glPushMatrix();
      glTranslated(cube.pos[0], cube.pos[1], cube.pos[2]);      
      glScaled(theCellSize, theCellSize, theCellSize);
      glBegin(GL_QUADS);
         glNormal3d( 0.0,  0.0, 1.0);
         glVertex3d(-LEN, -LEN, LEN);
         glVertex3d(-LEN,  LEN, LEN);
         glVertex3d( LEN,  LEN, LEN);
         glVertex3d( LEN, -LEN, LEN);
      glEnd();
   glPopMatrix();
}

void MACGrid::drawCube(const MACGrid::Cube& cube)
{
   glColor4dv(glm::value_ptr(cube.color));
   glPushMatrix();
      glTranslated(cube.pos[0], cube.pos[1], cube.pos[2]);      
      glScaled(theCellSize, theCellSize, theCellSize);
      glBegin(GL_QUADS);
         glNormal3d( 0.0, -1.0,  0.0);
         glVertex3d(-LEN, -LEN, -LEN);
         glVertex3d(-LEN, -LEN,  LEN);
         glVertex3d( LEN, -LEN,  LEN);
         glVertex3d( LEN, -LEN, -LEN);         

         glNormal3d( 0.0,  0.0, -0.0);
         glVertex3d(-LEN, -LEN, -LEN);
         glVertex3d(-LEN,  LEN, -LEN);
         glVertex3d( LEN,  LEN, -LEN);
         glVertex3d( LEN, -LEN, -LEN);

         glNormal3d(-1.0,  0.0,  0.0);
         glVertex3d(-LEN, -LEN, -LEN);
         glVertex3d(-LEN, -LEN,  LEN);
         glVertex3d(-LEN,  LEN,  LEN);
         glVertex3d(-LEN,  LEN, -LEN);

         glNormal3d( 0.0, 1.0,  0.0);
         glVertex3d(-LEN, LEN, -LEN);
         glVertex3d(-LEN, LEN,  LEN);
         glVertex3d( LEN, LEN,  LEN);
         glVertex3d( LEN, LEN, -LEN);

         glNormal3d( 0.0,  0.0, 1.0);
         glVertex3d(-LEN, -LEN, LEN);
         glVertex3d(-LEN,  LEN, LEN);
         glVertex3d( LEN,  LEN, LEN);
         glVertex3d( LEN, -LEN, LEN);

         glNormal3d(1.0,  0.0,  0.0);
         glVertex3d(LEN, -LEN, -LEN);
         glVertex3d(LEN, -LEN,  LEN);
         glVertex3d(LEN,  LEN,  LEN);
         glVertex3d(LEN,  LEN, -LEN);
      glEnd();
   glPopMatrix();
}

void MACGrid::drawParticle(Particle& p){
	double PLEN = LEN / 10.0;
	glColor4dv(glm::value_ptr(dvec4(1,0,0,1)));
	glPushMatrix();
	dvec3 pos = p.getPos();
		glTranslated(pos[0], pos[1], pos[2]);      
		glBegin(GL_QUADS);
			glNormal3d( 0.0, -1.0,  0.0);
			glVertex3d(-PLEN, -PLEN, -PLEN);
			glVertex3d(-PLEN, -PLEN,  PLEN);
			glVertex3d( PLEN, -PLEN,  PLEN);
			glVertex3d( PLEN, -PLEN, -PLEN);         

			glNormal3d( 0.0,  0.0, -0.0);
			glVertex3d(-PLEN, -PLEN, -PLEN);
			glVertex3d(-PLEN,  PLEN, -PLEN);
			glVertex3d( PLEN,  PLEN, -PLEN);
			glVertex3d( PLEN, -PLEN, -PLEN);

			glNormal3d(-1.0,  0.0,  0.0);
			glVertex3d(-PLEN, -PLEN, -PLEN);
			glVertex3d(-PLEN, -PLEN,  PLEN);
			glVertex3d(-PLEN,  PLEN,  PLEN);
			glVertex3d(-PLEN,  PLEN, -PLEN);

			glNormal3d( 0.0, 1.0,  0.0);
			glVertex3d(-PLEN, PLEN, -PLEN);
			glVertex3d(-PLEN, PLEN,  PLEN);
			glVertex3d( PLEN, PLEN,  PLEN);
			glVertex3d( PLEN, PLEN, -PLEN);

			glNormal3d( 0.0,  0.0, 1.0);
			glVertex3d(-PLEN, -PLEN, PLEN);
			glVertex3d(-PLEN,  PLEN, PLEN);
			glVertex3d( PLEN,  PLEN, PLEN);
			glVertex3d( PLEN, -PLEN, PLEN);

			glNormal3d(1.0,  0.0,  0.0);
			glVertex3d(PLEN, -PLEN, -PLEN);
			glVertex3d(PLEN, -PLEN,  PLEN);
			glVertex3d(PLEN,  PLEN,  PLEN);
			glVertex3d(PLEN,  PLEN, -PLEN);
		glEnd();
	glPopMatrix();
}