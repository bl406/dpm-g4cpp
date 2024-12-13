
#include "SimMaterialData.hh"
#include "SimElectronData.hh"
#include "SimPhotonData.hh"

#include "SimDPMLike.hh"

#include "Configuration.hh"

#include <iostream>
#include <iomanip>
#include <string>

#include <getopt.h>
#include "Source.hh"
#include "Geom.hh"

//
// M. Novak: 2021
//
// The simulation phase of the `dpm++` prototype. This can be executed only after
// the pre-init/data generation part (i.e. `dpm_GenerateData`) since this relies
// on all the data files generated during the pre-init phase while performs the
// simulation of with the specified configuration using the given input arguments.
//

//
// Input arguments for the simulation phase with their default values.
static std::string   gInputDataDir("./phydata");       // location of the pre-generated data (currently config0)
static std::string   gOutputFileName("dpm");   // the output filename
static float         gVoxelSize         =  1.0f;     // geometry voxel/box size in [mm]
static int           gNumPrimaries      =  (int)1.0E+5;  // simulate 100 000 primary events
static int           gConfigIndex       =  0;       // 0 that corresponds to a homogeneous Water geometry
//
static struct option options[] = {
  {"number-of-histories (number of primary events to simulate)          - default: 1.0E+5" , required_argument, 0, 'n'},
  {"configuration-index (one of the pre-defined configuration index)    - default: 0"      , required_argument, 0, 'c'},
  {"output-filename     (the filename of the result)                    - default: hist.sim"      , required_argument, 0, 'o'},
  {"help"                                                                                  , no_argument      , 0, 'h'},
  {0, 0, 0, 0}
};
// auxiliary functions for obtaining input arguments
void Help();
void GetOpt(int argc, char *argv[]);


//
//
// The main: obtaines the input arguments, creates the selected configuration
//           settings (mainly for infomation only since only its `fGeomIndex`
//           member is need now), loads all the data needed for the simulation
//           and executes the simulation with the given configuration and input
//           arguments. At termination, the `hist.sim` file contains the simulated
//           depth dose distribution.
int main(int argc, char *argv[]) {
  //
  // get the input arguments
  GetOpt(argc, argv);
  
  //
  // Load data for simulation:
  // - electron related data
  SimElectronData theSimElectronData;
  theSimElectronData.Load(gInputDataDir);
  // - photon related data
  SimPhotonData theSimPhotonData;
  theSimPhotonData.Load(gInputDataDir);
  // - configuration and material related data
  SimMaterialData theSimMaterialData;
  theSimMaterialData.Load(gInputDataDir);

  SimpleSource theSource(gVoxelSize);

  // create the simple geometry
  Geom geom(gVoxelSize, &theSimMaterialData, gConfigIndex);
  geom.InitGeom();
  geom.InitScore();

  //
  // Execute the simulation according to the iput arguments
  int nbatch = 10;
  gNumPrimaries = Simulate(gNumPrimaries, nbatch, &theSource, gVoxelSize, theSimMaterialData, geom);

  geom.Write(gOutputFileName, gNumPrimaries, nbatch);
  //
  return 0;
}


//
// Inplementation of the auxiliary functions for obtaining input ragumets
//
void Help() {
  std::cout<<"\n "<<std::setw(120)<<std::setfill('=')<<""<<std::setfill(' ')<<std::endl;
  std::cout<<"  The dpm++ simulation phase."
           << std::endl;
  std::cout<<"\n  Usage: dpm_Simulate [OPTIONS] \n"<<std::endl;
  for (int i = 0; options[i].name != NULL; i++) {
    printf("\t-%c  --%s\n", options[i].val, options[i].name);
  }
  std::cout<<"\n "<<std::setw(120)<<std::setfill('=')<<""<<std::setfill(' ')<<std::endl;
}


void GetOpt(int argc, char *argv[]) {
  while (true) {
    int c, optidx = 0;
    c = getopt_long(argc, argv, "hp:e:n:c:o:", options, &optidx);
    if (c == -1)
      break;
    switch (c) {
    case 0:
       c = options[optidx].val;
       /* fall through */
    case 'o':
		gOutputFileName = optarg;
		break;
    case 'n':
       gNumPrimaries  = std::stoi(optarg);
       break;
    case 'c':
       gConfigIndex   = std::stoi(optarg);
       break;
    case 'h':
       Help();
       exit(-1);
       break;
    default:
      printf(" *** Unknown input argument: %c\n",c);
      Help();
      exit(-1);
    }
   }
   std::cout << "\n === The dpm++ simulation confguration: \n"
            << "\n     - input data directory  : " << gInputDataDir
            << "\n     - output filename  : " << gOutputFileName
            << "\n     - number of histories   : " << gNumPrimaries
            << "\n     - geometry voxel size   : " << gVoxelSize  << " [mm]"
            << "\n     - confoguration index   : " << gConfigIndex
            << std::endl;
}
