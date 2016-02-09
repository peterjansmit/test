'''Ruben Demuynck + Yaff acknowledgement'''

from yaff import *
import numpy as np
from molmod.unit_cells import UnitCell
from molmod.ic import bend_angle
class ForcePartMTD(ForcePart):
	def __init__(self,system,hill,steps):
		self.name='MTD_'+str(hill.CV.name)
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
		cv=self.hill.CV.get_value(system)
		g=-np.sum(self.H[:]*self.hill.h*(cv-self.B[:])/w2*np.exp(-(cv-self.B[:])**2/2./w2))
		self.localGPos,self.localVTens=self.hill.CV.get_force(g,self.localGPos,self.localVTens,system)
		self.U=np.sum(self.H[:]*self.hill.h*np.exp(-(cv-self.B[:])**2/2./w2))
		return self.U

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
		self.mem+=self.hill.CV.get_value(iterative.ff.system)
		self.counter+=1
		if self.counter%self.updateT==0:
			self.hill.ffPart.B[self.updateC]=self.mem/self.counter
			self.hill.ffPart.H[self.updateC]=1
			self.updateC+=1
			self.counter=0
			self.mem=0

		

class Hills(object):
	def __init__(self,collectiveVariable, width=None, height=None):
		'''collectiveVariable 
				an object from the CollectiveVariable class
		   atoms
			in the latter three cases, one should identify the distance, angle or dihedral using atom numbers (PYTHON STYLE starting at zero)
		   width/height
			make a guess for both the hill height and width minimizing both bias and variance of the free energy profile.
		'''
		self.CV=collectiveVariable
		if width is None:
			'''In the near future we have to add the variational approach towards metadynamics'''
			raise NotImplementedError
		self.w=width
		self.h=height
		self.ffPart=None
		self.Hook=None

class HillsState(StateItem):
	def __init__(self,hill):
		self.hill=hill
		StateItem.__init__(self,'mtd_'+str(hill.CV.name))

	def get_value(self,iterative):
		return self.hill.ffPart.B

class Metadynamics1D(VerletIntegrator):		
	def __init__(self,ff,timestep,MetaSteps,MDSteps,Hills,state=None,hooks=None,velo0=None,temp0=300,scalevel0=True,time0=0.0,ndof=None,counter0=0):
		self.steps=MDSteps*MetaSteps
		print self.steps
		state=[]
		for hill in Hills:
			state.append(HillsState(hill))
			hill.ffPart=ForcePartMTD(ff.system,hill,MetaSteps)
			ff.add_part(hill.ffPart)
			hill.Hook=MTDHook(hill,MDSteps)
			hooks.append(hill.Hook)
		VerletIntegrator.__init__(self, ff, timestep, state=state, hooks=hooks, vel0=velo0,temp0=temp0, scalevel0=scalevel0, time0=time0, ndof=ndof, counter0=counter0)
		
	def runMeta(self):
		print self.steps
		self.run(self.steps)
