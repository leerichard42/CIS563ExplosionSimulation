#include "grid_data.h"

GridData::GridData() :
	mDfltValue(0.0),
	mMax(0.0,0.0,0.0) {
  // TODO : GridData constructor
}

GridData::GridData(const GridData& orig) {
  // TODO : GridData copy constructor
	mDfltValue = orig.mDfltValue;
	mData = orig.mData;
	mMax = orig.mMax;
}

GridData::~GridData() {
  // TODO : GridData destructor
}

GridData& GridData::operator=(const GridData& orig) {
  // TODO : Override GridData '=' operator with copy 
	if (this == &orig) return *this;
	mDfltValue = orig.mDfltValue;
	mMax = orig.mMax;
	mData = orig.mData;
	return *this;
}

void GridData::initialize(double dfltValue) {
  // TODO : Initialize the grid to a default value
	mDfltValue = dfltValue;
	mMax[0] = theCellSize*theDim[0];
	mMax[1] = theCellSize*theDim[1];
	mMax[2] = theCellSize*theDim[2];
	mData.resize(theDim[0]*theDim[1]*theDim[2], false);
    std::fill(mData.begin(), mData.end(), mDfltValue);
}

double& GridData::operator()(int i, int j, int k) {
  // TODO : Grid accessor that allows for client to access and set cell data
	static double dflt = 0;
	dflt = mDfltValue;

    if (i< 0 || j<0 || k<0 || 
		i > theDim[0]-1 || 
		j > theDim[1]-1 || 
		k > theDim[2]-1) return dflt;

	int col = i;
	int row = k*theDim[0];
	int stack = j*theDim[0]*theDim[2];

	return mData[col+row+stack];
}

const double GridData::operator()(int i, int j, int k) const {
  // TODO : Grid accessor
	static double dflt = 0;
	dflt = mDfltValue;

    if (i< 0 || j<0 || k<0 || 
		i > theDim[0]-1 || 
		j > theDim[1]-1 || 
		k > theDim[2]-1) return dflt;

	int col = i;
	int row = k*theDim[0];
	int stack = j*theDim[0]*theDim[2];

	return mData[col+row+stack];
}

double GridData::interpolate(const glm::dvec3& pt) {
  // TODO : Given a point, interpolate the value in the grid at that point
	glm::dvec3 pos = worldToSelf(pt);
	int i = (int) (pos[0]/theCellSize);
	int j = (int) (pos[1]/theCellSize);
	int k = (int) (pos[2]/theCellSize);
	double scale = 1.0/theCellSize;
	glm::dvec3 weight(scale * (pos[0] - i * theCellSize), scale * (pos[1] - j * theCellSize), scale * (pos[2] - k * theCellSize));
	//interpolate along y
	double y1 = LERP((*this)(i,j,k), (*this)(i,j+1,k), weight[1]);
	double y2 = LERP((*this)(i+1,j,k), (*this)(i+1,j+1,k), weight[1]);
	double y3 = LERP((*this)(i,j,k+1), (*this)(i,j+1,k+1), weight[1]);
	double y4 = LERP((*this)(i+1,j,k+1), (*this)(i+1,j+1,k+1), weight[1]);
	//interpolate along x
	double x1 = LERP(y1, y2, weight[0]);
	double x2 = LERP(y3, y4, weight[0]);
	//interpolate along z
	double z = LERP(x1, x2, weight[2]);
	return z;
	
}

std::vector<double>& GridData::data() {
  // TODO : Return underlying data structure (you may change the method header
  // to fit whatever design you choose).
  return mData;
}

void GridData::getCell(const glm::dvec3& pt, int& i, int& j, int& k) {
  // TODO : Given a point in world coordinates, return the cell index
  // corresponding to it.
	glm::dvec3 pos = worldToSelf(pt);
	i = (int) (pos[0]/theCellSize);
	j = (int) (pos[1]/theCellSize);
	k = (int) (pos[2]/theCellSize);
}

glm::dvec3 GridData::worldToSelf(const glm::dvec3& pt) const {
  // TODO : Given a point, returns the cell index that the grid uses in its own
  // space.
	glm::dvec3 out;
	out[0] = min(max(0.0, pt[0] - theCellSize*0.5), mMax[0]);
	out[1] = min(max(0.0, pt[1] - theCellSize*0.5), mMax[1]);
	out[2] = min(max(0.0, pt[2] - theCellSize*0.5), mMax[2]);
	return out;
}

GridDataX::GridDataX() : GridData() {
}

GridDataX::~GridDataX() {
}

void GridDataX::initialize(double dfltValue) {
  // TODO : Intialize GridDataX
	GridData::initialize(dfltValue);
	mMax[0] = theCellSize*(theDim[0]+1);
	mMax[1] = theCellSize*theDim[1];
	mMax[2] = theCellSize*theDim[2];
	mData.resize((theDim[0]+1)*theDim[1]*theDim[2], false);
	std::fill(mData.begin(), mData.end(), mDfltValue);
}

double& GridDataX::operator()(int i, int j, int k) {
  // TODO : GridX accessor
	static double dflt = 0;
    dflt = mDfltValue;
	if (i < 0 || i > theDim[0]) return dflt;
	if (j < 0) j = 0;
	if (j > theDim[1]-1) j = theDim[1]-1;
	if (k < 0) k = 0;
	if (k > theDim[2]-1) k = theDim[2]-1;
	
    int col = i;
    int row = k * (theDim[0]+1);
    int stack = j * (theDim[0]+1) * theDim[2];
    return mData[col+row+stack];
}

const double GridDataX::operator()(int i, int j, int k) const {
  // TODO : GridX accessor
	static double dflt = 0;
    dflt = mDfltValue;
	if (i < 0 || i > theDim[0]) return dflt;
	if (j < 0) j = 0;
	if (j > theDim[1]-1) j = theDim[1]-1;
	if (k < 0) k = 0;
	if (k > theDim[2]-1) k = theDim[2]-1;
	
    int col = i;
    int row = k * (theDim[0]+1);
    int stack = j * (theDim[0]+1) * theDim[2];
    return mData[col+row+stack];
}

glm::dvec3 GridDataX::worldToSelf(const glm::dvec3& pt) const {
  // TODO : Given a point, returns the cell index that the grid uses in its own
  // space
	glm::dvec3 out;
	out[0] = min(max(0.0, pt[0]), mMax[0]);
	out[1] = min(max(0.0, pt[1] - theCellSize*0.5), mMax[1]);
	out[2] = min(max(0.0, pt[2] - theCellSize*0.5), mMax[2]);
	return out;
}

GridDataY::GridDataY() : GridData() {
}

GridDataY::~GridDataY() {
}

void GridDataY::initialize(double dfltValue) {
  // TODO : Intialize GridDataY
	GridData::initialize(dfltValue);
	mMax[0] = theCellSize*theDim[0];
	mMax[1] = theCellSize*(theDim[1]+1);
	mMax[2] = theCellSize*theDim[2];
	mData.resize(theDim[0]*(theDim[1]+1)*theDim[2], false);
	std::fill(mData.begin(), mData.end(), mDfltValue);
}

double& GridDataY::operator()(int i, int j, int k) {
  // TODO : GridY accessor
	static double dflt = 0;
    dflt = mDfltValue;
	if (j < 0 || j > theDim[1]) return dflt;
	if (i < 0) i = 0;
	if (i > theDim[0]-1) i = theDim[0]-1;
	if (k < 0) k = 0;
	if (k > theDim[2]-1) k = theDim[2]-1;
	
    int col = i;
    int row = k * theDim[0];
    int stack = j * theDim[0] * theDim[2];
    return mData[col+row+stack];
}

const double GridDataY::operator()(int i, int j, int k) const {
  // TODO : GridY accessor
	static double dflt = 0;
    dflt = mDfltValue;
	if (j < 0 || j > theDim[1]) return dflt;
	if (i < 0) i = 0;
	if (i > theDim[0]-1) i = theDim[0]-1;
	if (k < 0) k = 0;
	if (k > theDim[2]-1) k = theDim[2]-1;
	
    int col = i;
    int row = k * theDim[0];
    int stack = j * theDim[0] * theDim[2];
    return mData[col+row+stack];
}

glm::dvec3 GridDataY::worldToSelf(const glm::dvec3& pt) const {
  // TODO : Given a point, returns the cell index that the grid uses in its own
  // space
	glm::dvec3 out;
	out[0] = min(max(0.0, pt[0] - theCellSize*0.5), mMax[0]);
	out[1] = min(max(0.0, pt[1]), mMax[1]);
	out[2] = min(max(0.0, pt[2] - theCellSize*0.5), mMax[2]);
	return out;
}

GridDataZ::GridDataZ() : GridData() {
}

GridDataZ::~GridDataZ() {
}

void GridDataZ::initialize(double dfltValue) {
  // TODO : Intialize GridDataZ
	GridData::initialize(dfltValue);
	mMax[0] = theCellSize*theDim[0];
	mMax[1] = theCellSize*theDim[1];
	mMax[2] = theCellSize*(theDim[2]+1);
	mData.resize(theDim[0]*theDim[1]*(theDim[2]+1), false);
	std::fill(mData.begin(), mData.end(), mDfltValue);
}

double& GridDataZ::operator()(int i, int j, int k) {
  // TODO : GridZ accessor
	static double dflt = 0;
    dflt = mDfltValue;
	if (k < 0 || k > theDim[2]) return dflt;
	if (i < 0) i = 0;
	if (i > theDim[0]-1) i = theDim[0]-1;
	if (j < 0) j = 0;
	if (j > theDim[1]-1) j = theDim[1]-1;
	
    int col = i;
    int row = k * theDim[0];
    int stack = j * theDim[0] * (theDim[2]+1);
    return mData[col+row+stack];
}

const double GridDataZ::operator()(int i, int j, int k) const {
  // TODO : GridY accessor
	static double dflt = 0;
    dflt = mDfltValue;
	if (k < 0 || k > theDim[2]) return dflt;
	if (i < 0) i = 0;
	if (i > theDim[0]-1) i = theDim[0]-1;
	if (j < 0) j = 0;
	if (j > theDim[1]-1) j = theDim[1]-1;
	
    int col = i;
    int row = k * theDim[0];
    int stack = j * theDim[0] * (theDim[2]+1);
    return mData[col+row+stack];
}

glm::dvec3 GridDataZ::worldToSelf(const glm::dvec3& pt) const {
  // TODO : Given a point, returns the cell index that the grid uses in its own
  // space
	glm::dvec3 out;
	out[0] = min(max(0.0, pt[0] - theCellSize*0.5), mMax[0]);
	out[1] = min(max(0.0, pt[1] - theCellSize*0.5), mMax[1]);
	out[2] = min(max(0.0, pt[2]), mMax[2]);
	return out;
}
