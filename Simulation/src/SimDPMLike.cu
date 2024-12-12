#include "SimDPMLike.hh"

#include <cstdio>

#include "Geom.hh"
#include "SimMaterialData.hh"
#include "SimElectronData.hh"
#include "SimITr1MFPElastic.hh"
#include "SimMaxScatStrength.hh"
#include "SimIMFPMoller.hh"
#include "SimIMFPBrem.hh"
#include "SimStoppingPower.hh"
#include "SimMollerTables.hh"
#include "SimSBTables.hh"
#include "SimGSTables.hh"
#include "SimPhotonData.hh"
#include "SimIMFPMaxPhoton.hh"
#include "SimIMFPPhoton.hh"
#include "SimKNTables.hh"
#include "Random.hh"
#include "Track.hh"
#include "TrackStack.hh"
#include "Source.hh"
#include "config.hh"
#include "error_checking.h"
#include "Utils.h"

#define DEBUG_LOG

__global__ void Simulate_kernel(int i)
{
	Track& track = d_TrackSeq.fData[i];

	// compute the distance to the boundary
	// (This also sets the box indices so the material index can be obtained)
	// init the step length to this distance to boundary
    //          float step =
	Geometry::DistanceToBoundary(track.fPosition, track.fDirection, track.fBoxIndx);
	// get the current material index: i.e. the base material of the voxel
	// NOTE: vacuum is indicated by the special voxel material index of -1
	//       Stop tracking if the track is in a voxel filled with vacuum.
	int theVoxelMatIndx = Geometry::GetMaterialIndex(track.fBoxIndx);
	if (theVoxelMatIndx < 0) {
		return;
	}
	// set the material index otherwise
	track.fMatIndx = theVoxelMatIndx;
	// Use the dedicated tracking for photons if we have one in hand now:
	if (track.fType == 0) {
		//KeepTrackingPhoton(track);
		return;
	}
	//
	// Track e-/e+ ortherwise:
	//
	// WE ASSUME HERE NOW THAT EACH VOXEL IS A CLEAR MATERIAL SO WE WILL
	// USE theVoxelMaterialDensity = theVoxelBaseMaterialDensity. HOWEVER,
	// THIS MIGH BE CHANGED ANYTIME WHEN THE GEOMETRY CAN PROVIDE A SUITABLE
	// VOXEL MATERIAL DENSITY.
	//
	// NOTE: density must be in g/cm3 units. (cannot be vacuum at this point)
	float theVoxelMatDensity = Geometry::GetVoxelMaterialDensity(theVoxelMatIndx);
	//
	// this will be used to alter between an hinge and remaining sub MSC-step
	bool   isMSCHinge = true;
	// this will be used to keep track of the pre-step point energy that we need
	// when we reach the msc hinge point (init. is not important cause will be set below)
	float theEkin0 = track.fEkin;
	//
	//
	// Compute the initial number of mfp left till the different interactions
	// which is -ln(R) = s/mfp or U(R) = s/tr1-mfp for the MSC hinge interaction:
	//
	// 1. elastic interaction with msc: s/tr1mfp and sample the hinge position as well
	//    NOTE: data are used for the reference material and not scalled by the density
	//          so numTr1MFP ~ s_max(E)/tr1-mfp(E) [no units]
	float numTr1MFP = MaxScatStrength::GetMaxScatStrength(track.fEkin);
	// travell this #tr1-mfp after the MSC-hinge took place
	float numTr1MFP0 = CuRand::rand() * numTr1MFP;
	// travel this #tr1-mfp till the MSC-hinge
	numTr1MFP -= numTr1MFP0;
	//
	// 2. Moller:
	// NOTE: as in DPM, the mfp for Moller will be assumed to have no energy dependence!!!
	//       So the update of the number-of-mfp-left will contain only the material
	//       dependence related scaling (that's again approximate but DPM does this).
	//       This material dependent update(scaling) relies on the \lam ~ [A/(Z\rho)]
	//       dependence: \lam' = \lam_ref(E') [Z\rho/A]_ref [A/(Z\rho)]_actual (such
	//       that in case of Moller (in DPM) even E'=E_0 in each step as dicussed above).
	float numMollerMFP = -std::log(CuRand::rand());
	// again, the reference material Moller IMFP value is used
	float invMollerMFP = IMFPMoller::GetIMFPPerDensity(track.fEkin);
	//
	// 3. bremsstrahlung:
	// NOTE: IMFP for bremsstrahlung is important so it's interpolated by using values for
	//       the actual material and kinetic energy in the `KeepTracking` code. Here we
	//       sample a `number-of-interaction-left` value, i.e. -ln(R) only (=s/imfp(E,mat))
	float numBremMFP = -std::log(CuRand::rand());
	//
	// Start and keep tracking this e- or e+ track while its stopped, i.e. its energy becomes
	// zero or goes away (in vacuum)
	while (track.fEkin > 0.0) {
		//
		// Now we can keep tracking this e- (even trhough boxes with different materials),
		// by decreasing these 3 above initial number-of-mfp/tr1-mfp-left at each step by the
		// correspnding n' = n - dS/mfp' or - dS/tr1-mfp as long as:
		// a.  any of these above 3 number of mfp/tr1-mfp goes to zero ==> the correspnding
		//     discrete Brem.(1), Moller(2) interaction happens or (3) either the MSC
		//     hinge or MSC step end point is reached
		// b.  (4) the e- energy drops below the e- tracking cut which is equal to the
		//     secondary e- production threshold (end of this history)
		// c.  the e- leaves the geometry, i.e. goes into vacuum (its energy set to 0 so return with 4)
		int whatHappend = KeepTrackingElectron(track, numTr1MFP, numMollerMFP, invMollerMFP, numBremMFP);
#ifdef DEBUG_LOG
        printf("whatHapped=%d track.fPostion=[%f %f %f] track.fTrackLength=%f track.fEdep=%f\n", 
            whatHappend, track.fPosition[0], track.fPosition[1], track.fPosition[2], track.fTrackLength, track.fEdep);
#endif
		//
		theVoxelMatIndx = track.fMatIndx;
		// terminate if the tarck is in vacuum (its energy has been set to zero)
		if (theVoxelMatIndx < 0) {
			continue;
		}
		theVoxelMatDensity = Geometry::GetVoxelMaterialDensity(theVoxelMatIndx);
		switch (whatHappend) {
			// (1) discrete bremsstrahlung interaction should be sampled:
			//     - sample energy transfer to the photon (if any)
		case 1: {
			// perform bremsstrahlung interaction but only if E0 > gcut
			if (track.fEkin > Geometry::GammaCut) {
				PerformBrem(track);
			}
			// check if the post-interaction electron energy dropped
			// below the tracking cut and stop tracking if yes
			if (track.fEkin < Geometry::ElectronCut) {
				// deposit the whole energy and stop tracking
				Geometry::Score(track.fEkin, track.fBoxIndx[2]);
				track.fEkin = 0.0;
				// perform annihilation in case of e+
				if (track.fType == +1) {
					PerformAnnihilation(track);
				}
				break;
			}
			// if the primary electron (or e+) survived, i.e. if we are here
			// then, re-sample the #mfp to travel till the next brem event
			// and break
			numBremMFP = -std::log(CuRand::rand());
			break;
		}
			  // (2) discrete Moller interaction is sampled:
			  // NOTE: no energy dependence is considered in case of Moller in DPM so
			  //       the numMollerMFP is evaluated only at this point and assumed to be
			  //       constant along the entire `KeepTracking` part (only material scaling is applied).
			  //       Furthermore, the kinetic energies of both the post interaction priary
			  //       and seconday electrons are guarantied to be above the secondary electron
			  //       production threshold so no need to check if their energy droppe below after
			  //       the interaction
			  //       Furthermore, note that Moller interaction is independent from Z
		case 2: {
			// perform ionisation (Moller) intraction but only if E0 > 2cut
			if (track.fEkin > 2. * Geometry::ElectronCut) {
				PerformMoller(track);
			}
			// Resample #mfp left and interpolate the IMFP since the enrgy has been changed.
			// Again, the reference material Moller IMFP value is used
			numMollerMFP = -std::log(CuRand::rand());
			invMollerMFP = IMFPMoller::GetIMFPPerDensity(track.fEkin);
			break;
		}
			  // (3) msc interaction happend: either hinge or just or end of an MSC step
		case 3: {
			if (isMSCHinge) {
				// Sample angular deflection from GS distr. and apply it
				// -----------------------------------------------------
				PerformMSCAngularDeflection(track, theEkin0);
				// -----------------------------------------------------
				// set the #tr1-mfp left to the remaining, i.e. after hinge part
				numTr1MFP = numTr1MFP0;
				// the end point is the next msc stop and not the hinge
				isMSCHinge = false;
			}
			else {
				// end point so resample #tr1-mfp left and the hinge point
				theEkin0 = track.fEkin;
				// again, the reference material K_1(E) is used
				numTr1MFP = MaxScatStrength::GetMaxScatStrength(theEkin0);
				// travell this #tr1-mfp after the MSC-hinge took place
				numTr1MFP0 = CuRand::rand() * numTr1MFP;
				// travell this #tr1-mfp till the MSC-hinge
				numTr1MFP -= numTr1MFP0;
				// hinge will be the next msc stop
				isMSCHinge = true;
			}
			break;
		}
			  // (4) the kinetic energy dropped below the tracking cut so the particle is stopped
		case 4:  // nothng to do now: track.fEkin should be zero et this point so the
			// "traking" while loop should be terminated after this `break`
			break;
		}
	} // end of tracking while loop
}

void Simulate(int nprimary, const Source* source)
{
    int nbatch = 1;
    if (nprimary / nbatch == 0) {
        nprimary = nbatch;
    }
    // nhist对nperbatch向上取整
    int nperbatch = nprimary / nbatch;
    if (nprimary % nperbatch != 0) {
        nbatch++;
    }
    nprimary = nperbatch * nbatch;

	int seq_size = nprimary;
	int stack_size = seq_size * 16;
    int nblocks = divUp(seq_size, THREADS_PER_BLOCK);

    CuRand::initCurand(nblocks, THREADS_PER_BLOCK);

    h_PhotonStack.init(stack_size);
	h_ElectronStack.init(stack_size);
	h_TrackSeq.init(seq_size);
    CudaCheckError();

    for (int ibatch = 0; ibatch < nbatch; ++ibatch){
        int nSimulatedPri = 0;

        while (1) {
            if (nSimulatedPri >= nperbatch) {
                if (!h_PhotonStack.empty() || !h_ElectronStack.empty()) {
                    if (h_PhotonStack.size() >= seq_size) {
                        h_TrackSeq.add_secondary(&h_PhotonStack);
                    }
                    else if (h_ElectronStack.size() >= seq_size) {
                        h_TrackSeq.add_secondary(&h_ElectronStack);
                    }
                    else {
                        if (h_PhotonStack.size() + h_ElectronStack.size() < h_TrackSeq.fCapacity) {
                            h_TrackSeq.add_secondary(&h_PhotonStack);
                            h_TrackSeq.add_secondary(&h_ElectronStack);
                        }
                        else if (h_PhotonStack.size() > 0) {
                            h_TrackSeq.add_secondary(&h_PhotonStack);
                        }
                        else if (h_ElectronStack.size() > 0) {
                            h_TrackSeq.add_secondary(&h_ElectronStack);
                        }
                    }
                }
                else {
                    break;
                }
            }
            else {
                if (h_PhotonStack.size() >= seq_size) {
                    h_TrackSeq.add_secondary(&h_PhotonStack);
                }
                else if (h_ElectronStack.size() >= seq_size) {
                    h_TrackSeq.add_secondary(&h_ElectronStack);
                }
                else {
                    h_TrackSeq.add_n_primary(seq_size, source);
                    nSimulatedPri += h_TrackSeq.fSize;
                }
            }
            CudaCheckError();
            cudaMemcpyToSymbol(d_TrackSeq, &h_TrackSeq, sizeof(TrackSeq));
            cudaMemcpyToSymbol(d_PhotonStack, &h_PhotonStack, sizeof(TrackStack));
            cudaMemcpyToSymbol(d_ElectronStack, &h_ElectronStack, sizeof(TrackStack));
            CudaCheckError();
            for (int i = 0; i < nprimary; ++i) {
                Simulate_kernel << <1, 1 >> > (i);
            }                       
            return;
            CudaCheckError();            
            h_TrackSeq.fSize = 0;
            cudaMemcpyFromSymbol(&h_PhotonStack, d_PhotonStack, sizeof(TrackStack));
            cudaMemcpyFromSymbol(&h_ElectronStack, d_ElectronStack, sizeof(TrackStack));
            CudaCheckError();
        }
        std::cout << "\n === End simulation of N = " << (ibatch+1) * nperbatch << " events === \n" << std::endl;
    }
}

__device__  int KeepTrackingElectron(Track& track, float& numTr1MFP, float& numMollerMFP, float invMollerMFP, float& numBremMFP) {
  int whatHappend = 0;
  // compute the distance to boundary: this will be the current value of the maximal step length
  float stepGeom = Geometry::DistanceToBoundary(track.fPosition, track.fDirection, track.fBoxIndx);
  // When the particle goes further than the pre-defined (+-10 [cm] along the xy plane, +10 cm
  // in the +z and the voxel-size in the -z direction) then `DistanceToBoundary` returns -1.0
  // indicating that the particle left the geometry: we stop tracking this particle
  if (stepGeom < 0.0) {
    // left the geometry
    track.fEkin = 0.0;
    return 4;
  }
  //
  const float theElectronCut = Geometry::ElectronCut;
  //
  // The Moller mfp energy dependence is not considered in DPM so we do the same:
  // we will keep using the Moller mfp evaluated at the initial energy for the reference
  // material. Only the material scaling will be applied below:
  // \lam' = \lam_ref(E') [Z\rho/A]_ref [A/(Z\rho)]_actual (for 1/\lam' to compute delta #mfp)
  while (whatHappend==0) {
    // init the current step lenght to that of the distance to boundary: we might
    // or might not go that far (depending on what the 3 physics tells) but for
    // sure that is the maximum step length becasue 'boundary crossing interaction'
    // happens after travelling that far. So set what happen to that (i.e. = 0.)
    float stepLength = stepGeom;
    whatHappend = 0;
    // get the current material index (when computing distance to boundary track.fBoxIndx
    // is updated if needed, i.e. if we crossed a boundary)
    int theVoxelMatIndx = Geometry::GetMaterialIndex(track.fBoxIndx);
    // stop tracking if the track entered to vacuum i.e. voxel with material index of -1
    if (theVoxelMatIndx < 0) {
      track.fEkin = 0.0;
      return 4;
    }
    // set the material index otherwise
    track.fMatIndx      = theVoxelMatIndx;
    // the material scaling factor for the Moller inverse-mf: [A/(Z\rho/)]_ref [(Z\rho)/A]_actual
    // or more exactly its [A/Z)]_ref [(Z)/A]_actual part
    float scalMolMFP = Geometry::GetVoxelMollerIMFPScaling(theVoxelMatIndx);
    // WE ASSUME HERE NOW THAT EACH VOXEL IS A CLEAR MATERIAL SO WE WILL
    // USE theVoxelMaterialDensity = theVoxelBaseMaterialDensity. HOWEVER,
    // THIS MIGH BE CHANGED ANYTIME WHEN THE GEOMETRY CAN PROVIDE A SUITABLE
    // VOXEL MATERIAL DENSITY.
    //
    // NOTE: density must be in g/cm3 units !!!!
    float theVoxelMatDensity = Geometry::GetVoxelMaterialDensity(theVoxelMatIndx);
    //
    // Here we compute the decrese of the #mfp/#tr1-mfp for the 3 interaction with the current,
    // maximum step length (i.e. the distance to boundary): #mfp' = #mfp - ds/mfp' or - ds/tr1_mfp
    // for MSC (where ' indicates values at the end point)
    //
    // compute the mid-point energy along this step by assuming:
    // - constant dEdx along the step, i.e. dEdx=dEdx(E_0) and dE = s dEdx --> E_mid = E_0 - 0.5 s dEdx
    // - the step equal to the current one, i.e. `stepLength` (dist. to boundary)
    // the restricted stopping power for this material: for the referecne material and scalled with the current density
    float theDEDX    = StoppingPower::GetDEDXPerDensity(track.fEkin, theVoxelMatIndx)*theVoxelMatDensity;
    // make sure that do not go below the minim e- energy
    float midStepE   = fmax(track.fEkin-0.5f*stepLength*theDEDX, theElectronCut );
    // elastic: #tr1-mfp' = #tr1-mfp - ds/tr1-mfp' so the change in #tr1-mfp is ds/tr1-mfp' and
    //          1/mfp' is computed here
    float delNumTr1MFP    = ITr1MFPElastic::GetITr1MFPPerDensity(midStepE, theVoxelMatIndx)*theVoxelMatDensity;
    // moller: see above the details
    float delNumMollerMFP = invMollerMFP*scalMolMFP*theVoxelMatDensity;
    // brem: #mfp' = #mfp - ds/mfp' with mfp = brem_mfp so the change in #mfp is ds/mfp' and
    //       1/mfp' is computed here
    float delNumBremMFP   = IMFPBrem::GetIMFPPerDensity(midStepE, theVoxelMatIndx)*theVoxelMatDensity;
    //
    //
    // Now we will see how far actually we go by trying to decrese each of the 3 #mfp/#tr1-mfp
    // by the changes in the number of mfp/tr1-mfp computed above as `delNum` :
    // - if we could manage to decrese all the 3 #mfp/tr1-mfp such that they are > 0
    //   then actually we reached the boundary: we cross and perform an other step.
    // - if any of the 3 #mfp/#tr1-mfp goes down to zero, then the lowest (i.e. shortest
    //   path to) will be considered to happen: the given inetraction need to beinvoked
    // In all cases, the energy loss along the given step needs to be computed!
    //
    // In DMP, the #mfp are tried to decresed with the current step length (that
    // is always the current shortest) as #mfp: n' = n - ds/mfp' then corrected
    // back if goes below zero, etc...
    // We compute the step length ds = n x mfp' (or n x tr1-mfp') i.e. that
    // would bring the given number #mfp/#tr1-mfp down to 0 (since n'=n-ds/mfp' or ds/tr1-mfp).
    // If this step is the shortest then we take this as current step length and we determine
    // the overal shortest and the corresponding interaction will happen (including boundary crossing).
    //
    float stepBrem    = numBremMFP/delNumBremMFP;
    if (stepBrem < stepLength) {
      // discrete bremsstrahlung (might) happen before reaching the boundary:
      stepLength  = stepBrem;
      whatHappend = 1;
    }
    float stepMoller  = numMollerMFP/delNumMollerMFP;
    if (stepMoller < stepLength) {
      // discrete Moller (might) happen even before bremsstrahlung:
      stepLength  = stepMoller;
      whatHappend = 2;
    }
    float stepElastic = numTr1MFP/delNumTr1MFP;
    if (stepElastic < stepLength) {
      // elastic interaction happens:
      // - either the hinge: sample and apply deflection and update numElMFP to the
      //                     remaining part
      // - end of 2nd step : nothing to do just resample the #mfp left since all
      //                     has been eaten up by the step lenght travelled so far
      // Before anything, refine the comutation of the mfp (i.e. the first transprt
      // mean free path in acse of elastic) regarding its energy dependence.
      // NOTE: that the 1/mfp values were computed at the mid-point energy (brem
      //       and elatsic since Moller is assumed to be constant), assuming that
      //       the geometry step will be taken and the dEdx is constant along this
      //       step (i.e. no energy dependence).
      //       Here we know that actually not the geometry step, but the stepElastic
      //       is taken since that is the shortest. So we recompute the mid-step-point
      //       energy according to the step lenght of stepElastic and re-evaluate
      //       the 1./mfp i.e. 1/tr1mfp at this energy value
      stepElastic = numTr1MFP/(ITr1MFPElastic::GetITr1MFPPerDensity(track.fEkin, theVoxelMatIndx)*theVoxelMatDensity);
      midStepE    = fmax( track.fEkin-0.5f*stepElastic*theDEDX, theElectronCut );
      delNumTr1MFP = ITr1MFPElastic::GetITr1MFPPerDensity(midStepE, theVoxelMatIndx)*theVoxelMatDensity;
      // don't let longer than the original in order to make sure that it is still the
      // minimum of all step lenghts
      stepElastic = fmin(stepLength, numTr1MFP/delNumTr1MFP);
      stepLength  = stepElastic;
      whatHappend = 3;
    }
    //
    // At this point, we know the step lenght so we can decrease all #mfp by
    // substracting the delta #mfp = ds/#mfp that correspond to this final stepLength
    numBremMFP   = (whatHappend == 1) ? 0 : numBremMFP   - stepLength*delNumBremMFP;
    numMollerMFP = (whatHappend == 2) ? 0 : numMollerMFP - stepLength*delNumMollerMFP;
    numTr1MFP    = (whatHappend == 3) ? 0 : numTr1MFP    - stepLength*delNumTr1MFP;


    //
    // Compte the (sub-treshold, i.e. along step) energy loss:
    // - first the mid-step energy using the final value of the step lenght and the
    //   pre-step point dEdx (assumed to be constant along the step).
    midStepE     = fmax( track.fEkin-0.5f*stepLength*theDEDX, theElectronCut );
    // - then the dEdx at this energy
    theDEDX      = StoppingPower::GetDEDXPerDensity(midStepE, theVoxelMatIndx)*theVoxelMatDensity;
    // - then the energy loss along the step using the mid-step dEdx (as constant)
    //   and the final energy
    float deltE = stepLength*theDEDX;
    float eFinal= track.fEkin-deltE;
    // check if energy dropped below tracking cut, i.e. below seconday e- production threshold
    // NOTE: HERE THERE IS A SUB-THRESHOLD TRACKING CONDITION IN DPM BUT WE WILL NEED TO SEE THAT !!!
    // ALSO: IF THE SELECTED STEP LENGHT BRINGS THE EKIN BELOW THRESHOLD WE DON'T TRY TO FIND THE
    //       STEP LENGTH (a smaller than selected) THAT ACTUALLY BRINGS EKIN EXACTLY TO THE THRESHOLD.
    //       SO THE TRACK LENGTH IS NOT PRECISE, SINCE WE KNOW THAT THE e- WAS STOPPED IN THIS VOLUME/BOX
    //       BUT WE DON'T COMPUTE THE EXCT POSITION
    if (eFinal < theElectronCut) {
      // deposit the whole energy and stop tracking
      track.fEdep  = track.fEkin;
      track.fEkin  = 0.0;
      // perform e+e- annihilation in case of e+
      if (track.fType == +1) {
        // annihilate the e+ at the correct position !!!
        track.fPosition[0] += track.fDirection[0]*stepLength;
        track.fPosition[1] += track.fDirection[1]*stepLength;
        track.fPosition[2] += track.fDirection[2]*stepLength;
        // make sure that the mat index is up to date
        Geometry::DistanceToBoundary(track.fPosition, track.fDirection, track.fBoxIndx);
        theVoxelMatIndx = Geometry::GetMaterialIndex(track.fBoxIndx);
        // check if we are in vacuum: nothing happens in that case
        if (theVoxelMatIndx < 0) {
          // kinetic energy is already zero
          return 4;
        }
        track.fMatIndx  = theVoxelMatIndx;
        PerformAnnihilation(track);
      }
      whatHappend  = 4;
    } else {
      track.fEkin  = eFinal;
      track.fEdep  = deltE;
    }
    //
    // Update particle position, track length etc.
    track.fPosition[0] += track.fDirection[0]*stepLength;
    track.fPosition[1] += track.fDirection[1]*stepLength;
    track.fPosition[2] += track.fDirection[2]*stepLength;
    // update cummulative track length
    track.fStepLenght   = stepLength;
    track.fTrackLength += stepLength;
    //
    // Score the continuous energy loss before going back to perform the discrete
    // interaction (whatHappend={1,2,3}) OR to terminate the tracking (whatHappend=4)
    // NOTE: we need to score before calling DistanceToBoundary again because that
    //       might positon the particle to the next volume.
    Geometry::Score(track.fEdep, track.fBoxIndx[2]);
    //
    // Compute distance to boundary if geometry limited this step:
    // - if geometry limited the step, the current position above is on a
    //   volume/box boundary
    // - when calling DistanceToBoundary such that the position is within half
    //   tolerance to a boudnary, the track position is updated to be on the
    //   other side, the distance to boundary from the new positon is computed
    //   and the x,y and z box coordinate indices are updated (so the material
    //   index will be the new one)
    if (whatHappend==0) {
      stepGeom = Geometry::DistanceToBoundary(track.fPosition, track.fDirection, track.fBoxIndx);
      if (stepGeom<0.0) {
        // left the geometry
        track.fEkin = 0.0;
        return 4;
      }
    }
  }
  return whatHappend;
}


__device__ void KeepTrackingPhoton(Track& track) {
  const float kPI      = 3.1415926535897932f;
  const float kEMC2    = 0.510991f;
  const float kInvEMC2 = 1.0/kEMC2;
  //
  const float theElectronCut = Geometry::ElectronCut;
  const float theGammaCut    = Geometry::GammaCut;
  //
  while (track.fEkin > 0.0f) {
    // get the global max-macroscopic cross section and use it for samppling the
    // the length till the next photon intercation (that includes delta interaction
    // as well)
    float globalMaxMFP   = 1.0f / IMFPMaxPhoton::GetIMFP(track.fEkin);
    float     stepLength = -globalMaxMFP*std::log(CuRand::rand());
    // Update particle position, track length etc.
    track.fPosition[0] += track.fDirection[0]*stepLength;
    track.fPosition[1] += track.fDirection[1]*stepLength;
    track.fPosition[2] += track.fDirection[2]*stepLength;
    // update cummulative track length
    track.fStepLenght   = stepLength;
    track.fTrackLength += stepLength;
    // determine currecnt voxel index
    if (Geometry::DistanceToBoundary(track.fPosition, track.fDirection, track.fBoxIndx) < 0) {
      // left the geometry
      track.fEkin = 0.0;
      return;
    }
    //
    // check if any interaction happened
    int theVoxelMatIndx = Geometry::GetMaterialIndex(track.fBoxIndx);
    if (theVoxelMatIndx < 0) {
      // terminate because its in the vacuum
      track.fEkin = 0.0;
      return;
    }
    track.fMatIndx = theVoxelMatIndx;
    float theVoxelMatDensity = Geometry::GetVoxelMaterialDensity(theVoxelMatIndx);
    //
    float totalIMFP = IMFPTotal::GetIMFPPerDensity(track.fEkin, theVoxelMatIndx)*theVoxelMatDensity;
    //
    // P(no-inetaction) = 1.0-mxsecTotal/mxsecGlobalMax
    const float r1 = CuRand::rand();
    float theProb = 1.0-totalIMFP*globalMaxMFP;
    if (r1 < theProb) {
      continue; // with the same globalMaxMFP since the enrgy did not changed !!!
    }
    //
    // otherwise: check which interaction happend P(i) = mxsec-i/mxsecTotal
    // compute cumulated probability of adding Compton prob
    theProb += IMFPCompton::GetIMFPPerDensity(track.fEkin, theVoxelMatIndx)*theVoxelMatDensity*globalMaxMFP;
    if (r1 < theProb) {
      // Compton interaction: Klein-Nishina like
      // the photon scattering angle and post-interafctin energy fraction
        float three_rand[3] = { CuRand::rand(), CuRand::rand(), CuRand::rand() };
      const float theEps = KNTables::SampleEnergyTransfer(track.fEkin, three_rand[2], three_rand[1], three_rand[0]);
      const float kappa  = track.fEkin*kInvEMC2;
      const float phCost = 1.0f-(1.0-theEps)/(theEps*kappa); // 1- (1-cost)
      const float phEner = theEps*track.fEkin;
      const float phPhi  = 2.0f*kPI*CuRand::rand();
      const float elEner = track.fEkin-phEner;
      // insert the secondary e- only if ist energy is above the tracking cut
      // and deposit the corresponding enrgy otherwise
      if (elEner < theElectronCut) {
        Geometry::Score(elEner, track.fBoxIndx[2]);
        //Geometry::Score(elEner, track.fPosition[2]);
      } else {
        // insert secondary e- but first compute its cost
        const float e0 = track.fEkin*kInvEMC2;
        float elCost   = (1.0f+e0)*std::sqrt((1.0f-theEps)/(e0*(2.0f+e0*(1.0f-theEps))));
        //
        Track& aTrack = d_ElectronStack.push_one();
        aTrack.fType         = -1;
        aTrack.fEkin         = elEner;
        aTrack.fMatIndx      = track.fMatIndx;
        aTrack.fPosition[0]  = track.fPosition[0];
        aTrack.fPosition[1]  = track.fPosition[1];
        aTrack.fPosition[2]  = track.fPosition[2];
        aTrack.fBoxIndx[0]   = track.fBoxIndx[0];
        aTrack.fBoxIndx[1]   = track.fBoxIndx[1];
        aTrack.fBoxIndx[2]   = track.fBoxIndx[2];
        const float sint    = std::sqrt((1.0f+elCost)*(1.0f-elCost));
        const float phi     = phPhi+kPI;
        aTrack.fDirection[0] = sint*std::cos(phi);
        aTrack.fDirection[1] = sint*std::sin(phi);
        aTrack.fDirection[2] = elCost;
        RotateToLabFrame(aTrack.fDirection, track.fDirection);
      }
      // update the photon properties:
      // stop the photon if its energy dropepd below the photon absorption threshold
      track.fEkin = phEner;
      if (track.fEkin < theGammaCut) {
         Geometry::Score(track.fEkin, track.fBoxIndx[2]);
        //Geometry::Score(track.fEkin, track.fPosition[2]);
        track.fEkin = 0.0;
        return;
      } else {
        float phSint = std::sqrt((1.0-phCost)*(1.0+phCost));
        float u1 = phSint*std::cos(phPhi);
        float u2 = phSint*std::sin(phPhi);
        float u3 = phCost;
        // rotate new direction from the scattering to the lab frame
        RotateToLabFrame(u1, u2, u3, track.fDirection[0], track.fDirection[1], track.fDirection[2]);
        // update track direction
        track.fDirection[0] = u1;
        track.fDirection[1] = u2;
        track.fDirection[2] = u3;
      }
      continue;
    }

    // compute cumulated probability of adding Pair-production prob
    theProb += IMFPPairProd::GetIMFPPerDensity(track.fEkin, theVoxelMatIndx)*theVoxelMatDensity*globalMaxMFP;
    if (r1 < theProb) {
      // Pair-production interaction:
      const float sumEkin = track.fEkin-2.0*kEMC2;
      // simple uniform share of the enrgy between the e- and e+ going to the
      // same direction as the original photon.
      // no difference between the e- and e+ transport till the end:
      // - when the e+ stops, 2 photons are emitted
      // we will assume that e1 is the e+
      float e1 = CuRand::rand()*sumEkin;
      float e2 = sumEkin-e1;
      // insert the e- and e+ only if their energy is above the tracking cut
      // the e-
      if (e2 < theElectronCut) {
        Geometry::Score(e2, track.fBoxIndx[2]);
        //Geometry::Score(e2, track.fPosition[2]);
      } else {
        Track& aTrack        = d_ElectronStack.push_one();
        aTrack.fType         = -1;
        aTrack.fEkin         = e2;
        aTrack.fMatIndx      = track.fMatIndx;
        aTrack.fPosition[0]  = track.fPosition[0];
        aTrack.fPosition[1]  = track.fPosition[1];
        aTrack.fPosition[2]  = track.fPosition[2];
        aTrack.fBoxIndx[0]   = track.fBoxIndx[0];
        aTrack.fBoxIndx[1]   = track.fBoxIndx[1];
        aTrack.fBoxIndx[2]   = track.fBoxIndx[2];
        aTrack.fDirection[0] = track.fDirection[0];
        aTrack.fDirection[1] = track.fDirection[1];
        aTrack.fDirection[2] = track.fDirection[2];
      }
      // the e+
      if (e1 < theElectronCut) {
          Geometry::Score(e1, track.fBoxIndx[2]);
        //Geometry::Score(e1, track.fPosition[2]);
        PerformAnnihilation(track);
      } else {
        Track& aTrack        = d_ElectronStack.push_one();
        aTrack.fType         = +1;
        aTrack.fEkin         = e1;
        aTrack.fMatIndx      = track.fMatIndx;
        aTrack.fPosition[0]  = track.fPosition[0];
        aTrack.fPosition[1]  = track.fPosition[1];
        aTrack.fPosition[2]  = track.fPosition[2];
        aTrack.fBoxIndx[0]   = track.fBoxIndx[0];
        aTrack.fBoxIndx[1]   = track.fBoxIndx[1];
        aTrack.fBoxIndx[2]   = track.fBoxIndx[2];
        aTrack.fDirection[0] = track.fDirection[0];
        aTrack.fDirection[1] = track.fDirection[1];
        aTrack.fDirection[2] = track.fDirection[2];
      }
      // kill the primary photon
      track.fEkin = 0.0;
      return;
    }

    // if we are here then Photoelectric effect happens that absorbs the photon:
    // - score the current phton energy and stopp the photon
    Geometry::Score(track.fEkin, track.fBoxIndx[2]);
    //Geometry::Score(track.fEkin, track.fPosition[2]);
    track.fEkin = 0.0;
  };
}


__device__ void RotateToLabFrame(float &u, float &v, float &w, float u1, float u2, float u3) {
  float up = u1*u1 + u2*u2;
  if (up>0.) {
    up = std::sqrt(up);
    float px = u;
    float py = v;
    float pz = w;
    u = (u1*u3*px - u2*py)/up + u1*pz;
    v = (u2*u3*px + u1*py)/up + u2*pz;
    w =    -up*px +             u3*pz;
  } else if (u3<0.f) {       // phi=0  teta=pi
    u = -u;
    w = -w;
  }
}

__device__ void RotateToLabFrame(float* dir, float* refdir) {
  RotateToLabFrame(dir[0], dir[1], dir[2], refdir[0], refdir[1], refdir[2]);
}


// It is assumed that track.fEkin > gamma-cut!
// (Interaction is not possible otherwise)
__device__ void PerformBrem(Track& track) {
  const float kPI            = 3.1415926535897932f;
  const float kEMC2          = 0.510991f;
  const float kHalfSqrt2EMC2 = kEMC2 * 0.7071067812f;
  // sample energy transferred to the emitted gamma photon
  float three_rand[3] = { CuRand::rand(), CuRand::rand(), CuRand::rand() };
  const float eGamma = SBTables::SampleEnergyTransfer(track.fEkin,track.fMatIndx, three_rand[2], three_rand[1], three_rand[0]);
 // insert the secondary gamma track into the stack
 Track& aTrack        = d_PhotonStack.push_one();
 aTrack.fType         = 0;
 aTrack.fEkin         = eGamma;
 aTrack.fMatIndx      = track.fMatIndx;
 aTrack.fPosition[0]  = track.fPosition[0];
 aTrack.fPosition[1]  = track.fPosition[1];
 aTrack.fPosition[2]  = track.fPosition[2];
 aTrack.fBoxIndx[0]   = track.fBoxIndx[0];
 aTrack.fBoxIndx[1]   = track.fBoxIndx[1];
 aTrack.fBoxIndx[2]   = track.fBoxIndx[2];
 //
 // compute emission direction (rough approximation in DPM by the mean)
 // and no deflection of the primary e-
 const float dum0    = kHalfSqrt2EMC2/(track.fEkin+kEMC2);
 const float cost    = fmax(-1.0f, fmin(1.0f, 1.0f-dum0*dum0));
 const float sint    = std::sqrt((1.0f+cost)*(1.0f-cost));
 const float phi     = 2.0f*kPI*CuRand::rand();
 aTrack.fDirection[0] = sint*std::cos(phi);
 aTrack.fDirection[1] = sint*std::sin(phi);
 aTrack.fDirection[2] = cost;
 RotateToLabFrame(aTrack.fDirection, track.fDirection);
 // decrease the primary energy:
 track.fEkin = track.fEkin-eGamma;
}

// It is assumed that track.fEkin > 2*electron-cut!
// (Interaction is not possible otherwise)
__device__ void PerformMoller(Track& track) {
  const float kPI     = 3.1415926535897932f;
  const float kEMC2   = 0.510991f;
  const float k2EMC2  = 2.0f*kEMC2;
  float three_rand[3] = { CuRand::rand(), CuRand::rand(), CuRand::rand() };
  const float secEkin = MollerTables::SampleEnergyTransfer(track.fEkin, three_rand[2], three_rand[1], three_rand[0]);
                                                               
  const float cost    = std::sqrt(secEkin*(track.fEkin+k2EMC2)/(track.fEkin*(secEkin+k2EMC2)));
  const float secCost = fmin(1.0f, cost);
  // insert the secondary e- track into the stack
  Track& aTrack = d_ElectronStack.push_one();
  aTrack.fType         = -1;
  aTrack.fEkin         = secEkin;
  aTrack.fMatIndx      = track.fMatIndx;
  aTrack.fPosition[0]  = track.fPosition[0];
  aTrack.fPosition[1]  = track.fPosition[1];
  aTrack.fPosition[2]  = track.fPosition[2];
  aTrack.fBoxIndx[0]   = track.fBoxIndx[0];
  aTrack.fBoxIndx[1]   = track.fBoxIndx[1];
  aTrack.fBoxIndx[2]   = track.fBoxIndx[2];
  const float sint    = std::sqrt((1.0f+secCost)*(1.0f-secCost));
  const float phi     = 2.0f*kPI*CuRand::rand();
  aTrack.fDirection[0] = sint*std::cos(phi);
  aTrack.fDirection[1] = sint*std::sin(phi);
  aTrack.fDirection[2] = secCost;
  RotateToLabFrame(aTrack.fDirection, track.fDirection);
  // decrease primary energy: DMP do not deflect the primary
  track.fEkin -= secEkin;
}


__device__ void PerformMSCAngularDeflection(Track& track, float ekin0) {
  const float kPI  = 3.1415926535897932f;
  float two_rand[2] = { CuRand::rand(), CuRand::rand() };
  const float dum0 = GSTables::SampleAngularDeflection(ekin0, two_rand[1], two_rand[0]);
  const float cost = fmax(-1.0f, fmin(1.0f, dum0));
  const float sint = std::sqrt((1.0-cost)*(1.0f+cost));
  // smaple \phi: uniform in [0,2Pi] <== spherical symmetry of the scattering potential
  const float phi  = 2.0f*kPI*CuRand::rand();
  // compute new direction (relative to 0,0,1 i.e. in the scattering frame)
  float u1 = sint*std::cos(phi);
  float u2 = sint*std::sin(phi);
  float u3 = cost;
  // rotate new direction from the scattering to the lab frame
  RotateToLabFrame(u1, u2, u3, track.fDirection[0], track.fDirection[1], track.fDirection[2]);
  // update track direction
  track.fDirection[0] = u1;
  track.fDirection[1] = u2;
  track.fDirection[2] = u3;
}


__device__ void PerformAnnihilation(Track& track) {
  const float kPI      = 3.1415926535897932f;
  const float kEMC2    = 0.510991f;
  // isotropic direction
  const float cost = 1.0f-2.0f*CuRand::rand();
  const float sint = std::sqrt((1.0f-cost)*(1.0f+cost));
  const float phi  = 2.0f*kPI*CuRand::rand();
  const float rx   = sint*cos(phi);
  const float ry   = sint*sin(phi);
  const float rz   = cost;

  Track& aTrack = d_PhotonStack.push_one();
  aTrack.fType         = 0;
  aTrack.fEkin         = kEMC2;
  aTrack.fMatIndx      = track.fMatIndx;
  aTrack.fPosition[0]  = track.fPosition[0];
  aTrack.fPosition[1]  = track.fPosition[1];
  aTrack.fPosition[2]  = track.fPosition[2];
  aTrack.fBoxIndx[0]   = track.fBoxIndx[0];
  aTrack.fBoxIndx[1]   = track.fBoxIndx[1];
  aTrack.fBoxIndx[2]   = track.fBoxIndx[2];
  aTrack.fDirection[0] = rx;
  aTrack.fDirection[1] = ry;
  aTrack.fDirection[2] = rz;

  Track& aTrack1        = d_PhotonStack.push_one();
  aTrack1.fType         = 0;
  aTrack1.fEkin         = kEMC2;
  aTrack1.fMatIndx      = track.fMatIndx;
  aTrack1.fPosition[0]  = track.fPosition[0];
  aTrack1.fPosition[1]  = track.fPosition[1];
  aTrack1.fPosition[2]  = track.fPosition[2];
  aTrack1.fBoxIndx[0]   = track.fBoxIndx[0];
  aTrack1.fBoxIndx[1]   = track.fBoxIndx[1];
  aTrack1.fBoxIndx[2]   = track.fBoxIndx[2];
  aTrack1.fDirection[0] = -rx;
  aTrack1.fDirection[1] = -ry;
  aTrack1.fDirection[2] = -rz;
}
