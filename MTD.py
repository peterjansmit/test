'''Ruben Demuynck + Yaff acknowledgement'''

from yaff import *
import numpy as np
from molmod.unit_cells import UnitCell
from molmod.ic import bend_angle
class ForcePartMTD(ForcePart):
	def __init__(self,system,hill,steps):
		self.name='MTD_'+str(hill.atoms)
		self.hill=hill
		self.system=system
		self.B=np.zeros(steps)
		self.H=np.zeros(steps)
		ForcePart.__init__(self,self.name,system)

	def determineF(self,system):
		w2=self.hill.w**2
		N=system.natom
		self.localGPos=np.zeros((N,3))
		self.localVTens=np.zeros((3,3))
		if self.hill.id=='volume':
			cv=system.cell.volume
			g=-np.sum(self.H[:]*self.hill.h*(cv-self.B[:])/w2*np.exp(-(cv-self.B[:])**2/2./w2))
			self.localVTens=np.identity(3)*g*cv
			self.U=np.sum(self.H[:]*self.hill.h*np.exp(-(cv-self.B[:])**2/2./w2))
		elif self.hill.id=='cell':
			rvecs=system.cell.rvecs
			cv=rvecs[self.hill.atoms,self.hill.atoms]
			g=np.zeros((3,3))
			g[self.hill.atoms,self.hill.atoms]=-np.sum(self.H[:]*self.hill.h*(cv-self.B[:])/w2*np.exp(-(cv-self.B[:])**2/2./w2))
			self.localVTens=np.dot(rvecs.T,g)
			self.U=np.sum(self.H[:]*self.hill.h*np.exp(-(cv-self.B[:])**2/2./w2))
		elif self.hill.id=='angle':
			cv,cvderiv=bend_angle(np.array([system.pos[self.hill.atoms[0],:],system.pos[self.hill.atoms[1],:],system.pos[self.hill.atoms[2],:]]),deriv=1)
			g=-np.sum(self.H[:]*self.hill.h*(cv-self.B[:])/w2*np.exp(-(cv-self.B[:])**2/2./w2))
			for i,a in enumerate(self.hill.atoms):
				self.localGPos[a,:]=g*cvderiv[i,:]
			self.U=np.sum(self.H[:]*self.hill.h*np.exp(-(cv-self.B[:])**2/2./w2))
		else:
			raise NotImplementedError

   	def _internal_compute(self, gpos, vtens):
		self.determineF(self.system)
		if vtens is not None:
			vtens+=self.localVTens
		if gpos is not None:
			gpos+=self.localGPos
		return self.U

class MTDHook(VerletHook):
	def __init__(self,hill,steps):
		self.counter=0
		self.updateT=steps
		self.updateC=0
		self.mem=0.0
		self.hill=hill
		VerletHook.__init__(self)
		
	def init(self,iterative):
		pass

	def pre(self,iterative):
		pass
	
	def post(self,iterative):
		self.mem+=self.determineCV(iterative)
		self.counter+=1
		if self.counter%self.updateT==0:
			self.hill.ffPart.B[self.updateC]=self.mem/self.counter
			self.hill.ffPart.H[self.updateC]=1
			self.updateC+=1
			self.counter=0
			self.mem=0

	def determineCV(self,iterative):
		if self.hill.id=='volume':
			return iterative.ff.system.cell.volume		
		elif self.hill.id=='cell':
                        return iterative.ff.system.cell.rvecs[self.hill.atoms,self.hill.atoms]
		elif self.hill.id=='angle':
			return bend_angle(np.array([iterative.ff.system.pos[self.hill.atoms[0],:],iterative.ff.system.pos[self.hill.atoms[1],:],iterative.ff.system.pos[self.hill.atoms[2],:]]))[0]
		else:
			raise NotImplementedError
		

class Hills(object):
	def __init__(self,identifier, atoms=None, width=None, height=None):
		'''identifier 
				are volume, cell lengths, distance, angle and dihedral
		   atoms
			in the latter three cases, one should identify the distance, angle or dihedral using atom numbers (PYTHON STYLE starting at zero)
		   width/height
			make a guess for both the hill height and width minimizing both bias and variance of the free energy profile.
		'''
		self.id=identifier
		self.atoms=atoms
		if width is None or height is None:
			'''In the near future we have to add the variational approach towards metadynamics'''
			raise NotImplementedError
		self.w=width
		self.h=height
		self.ffPart=None

class HillsState(StateItem):
	def __init__(self,hill):
		self.hill=hill
		StateItem.__init__(self,'mtd_'+str(hill.atoms))

	def get_value(self,iterative):
		return self.hill.ffPart.B

class Metadynamics1D(VerletIntegrator):		
	def __init__(self,ff,timestep,MetaSteps,MDSteps,Hills,state=None,hooks=None,velo0=None,temp0=300,scalevel0=True,time0=0.0,ndof=None,counter0=0):
		self.steps=MDSteps*MetaSteps
		state=[]
		for hill in Hills:
			state.append(HillsState(hill))
			hill.ffPart=ForcePartMTD(ff.system,hill,MetaSteps)
			ff.add_part(hill.ffPart)
			hill.Hook=MTDHook(hill,MDSteps)
			hooks.append(hill.Hook)
		VerletIntegrator.__init__(self, ff, timestep, state=state, hooks=hooks, vel0=velo0,temp0=temp0, scalevel0=scalevel0, time0=time0, ndof=ndof, counter0=counter0)
		
	def runMeta(self):
		self.run(self.steps)
