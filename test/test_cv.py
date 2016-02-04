from yaff import *
import MTD
import h5py
import numpy as np
import colvar

system = System.from_file('UiO-eeaa.chk')
ff = ForceField.generate(system, 'pars.txt', rcut=15 * angstrom, alpha_scale=3.2, gcut_scale=1.5,smooth_ei=True)


f=h5py.File('traj.h5',mode='w')
hdf=HDF5Writer(f)

xyz = XYZWriter('traj.xyz')
temp = 300 * kelvin
timestep = 0.5 * femtosecond
press = 0*1e6 * pascal

thermo = NHCThermostat(temp, timecon=100.0 * femtosecond)
baro = MTKBarostat(ff, temp, press, timecon=1000.0 * femtosecond, vol_constraint=False)
TBC = TBCombination(thermo, baro)

print 'test angle'
vhill=[]
angleCV=colvar.Angle('angle1',[0,4,95])
vhill.append(MTD.Hills(angleCV,width=np.radians(10),height=5*kjmol))
meta=MTD.Metadynamics1D(ff, timestep,10,5,vhill, hooks=[TBC])
meta.runMeta()

print 'test distance'
vhill=[]
angleCV=colvar.Distance('dist1',[0,4])
vhill.append(MTD.Hills(angleCV,width=2*angstrom,height=5*kjmol))
meta=MTD.Metadynamics1D(ff, timestep,10,5,vhill, hooks=[ TBC])
meta.runMeta()

print ''
print 'test volume'
vhill=[]
angleCV=colvar.Volume()
vhill.append(MTD.Hills(angleCV,width=50*angstrom**3,height=5*kjmol))
meta=MTD.Metadynamics1D(ff, timestep,10,5,vhill, hooks=[TBC])
meta.runMeta()

print 'test cell'
vhill=[]
angleCV=colvar.CellParameter('cell1',[2,2])
vhill.append(MTD.Hills(angleCV,width=0.5*angstrom,height=5*kjmol))
meta=MTD.Metadynamics1D(ff, timestep,10,5,vhill, hooks=[TBC])
meta.runMeta()

