#ifndef Track_HH
#define Track_HH

#include <vector>

//
// A simple track data structure to keep positon, direction and accumulated track
// length information.
class Track {
public:
  float fPosition[3];   // rx, ry, rz
  float fDirection[3];  // dx, dy, dz normalised to 1
  int    fBoxIndx[3];    // x,y and z indices of the current box

  int    fType;          // -1 for e-, 0 for gamma and +1 for e+
  int    fMatIndx;       // material index
  float fTrackLength;   // cummulative track length
  float fEkin;          // kinetic energy
  float fStepLenght;    // current step length
  float fEdep;          // energy deposit in the given step
  //
  __host__ __device__
  Track() {
    Reset();
  }

  // copy ctr
  __host__ __device__
  Track(const Track& o) {
    fPosition[0]  = o.fPosition[0];
    fPosition[1]  = o.fPosition[1];
    fPosition[2]  = o.fPosition[2];
    //
    fDirection[0] = o.fDirection[0];
    fDirection[1] = o.fDirection[1];
    fDirection[2] = o.fDirection[2];
    //
    fBoxIndx[0]   = o.fBoxIndx[0];
    fBoxIndx[1]   = o.fBoxIndx[1];
    fBoxIndx[2]   = o.fBoxIndx[2];
    //
    fType         = o.fType;
    fMatIndx      = o.fMatIndx;
    fTrackLength  = o.fTrackLength;
    fEkin         = o.fEkin;
    fStepLenght   = o.fStepLenght;
    fEdep         = o.fEdep;
  }

  //
  __host__ __device__
  void Reset() {
    fPosition[0]  = 0.0;
    fPosition[1]  = 0.0;
    fPosition[2]  = 0.0;
    //
    fDirection[0] = 0.0;
    fDirection[1] = 0.0;
    fDirection[2] = 1.0;
    //
    fBoxIndx[0]   = 0;
    fBoxIndx[1]   = 0;
    fBoxIndx[2]   = 0;
    //
    fType         = -3;
    fMatIndx      = -1;
    fTrackLength  = 0.0;
    fEkin         = 0.0;
    fStepLenght   = 0.0;
    fEdep         = 0.0;
  }
  __host__ __device__
  void Copy(Track& in){
    in.fPosition[0]  = fPosition[0];
    in.fPosition[1]  = fPosition[1];
    in.fPosition[2]  = fPosition[2];
    //
    in.fDirection[0] = fDirection[0];
    in.fDirection[1] = fDirection[1];
    in.fDirection[2] = fDirection[2];
    //
    in.fBoxIndx[0]   = fBoxIndx[0];
    in.fBoxIndx[1]   = fBoxIndx[1];
    in.fBoxIndx[2]   = fBoxIndx[2];
    //
    in.fType         = fType;
    in.fMatIndx      = fMatIndx;
    in.fTrackLength  = fTrackLength;
    in.fEkin         = fEkin;
    in.fStepLenght   = fStepLenght;
    in.fEdep         = fEdep;
  }
};

#endif // Track_HH
