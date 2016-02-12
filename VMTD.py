'''Ruben Demuynck + Yaff acknowledgement'''

from yaff import *
import numpy as np
from molmod.unit_cells import UnitCell
from molmod.ic import bend_angle




class ForcePartVMTD(ForcePart):
	def __init__(self,system,hill):
		self.name='MTD'
		self.hill=hill
		self.system=system
		self.B=np.zeros((hills.bins,len(hill.CV)))
		self.H=np.zeros(hills.bins)
		ForcePart.__init__(self,self.name,system)

	def determineF(self,system):
		N=system.natom
		self.localGPos=np.zeros((N,3))
		self.localVTens=np.zeros((3,3))
		cv=[cv.get_value(system) for cv in self.hill.CV]
		kernel=1
		for i,value in enumerate(self.hill.CV):
			kernel*=np.exp(-(cv[i]-self.B[:,i])**2/2./self.hill.w[i]**2)
		for i,value in enumerate(self.hill.CV):
			g=-np.sum(self.H[:]*(cv[i]-self.B[:,i])/self.hill.w[i]**2*kernel)
			self.localGPos,self.localVTens=value.get_force(g,self.localGPos,self.localVTens,system)
		self.U=np.sum(self.H[:]*kernel)
		return self.U

   	def _internal_compute(self, gpos, vtens):
		self.determineF(self.system)
		if vtens is not None:
			vtens+=self.localVTens
		if gpos is not None:
			gpos+=self.localGPos
		return self.U

class VMTDHook(VerletHook):
#VerletHook to keep track of the metadynamics hill position
class optimize(VerletHook):
        def __init__(self,temp,hill,updateT):
                self.alpha=np.zeros(hill.bins)
                self.functDer=np.zeros(hill.bins)
                self.functDerDer=np.zeros(hill.bins)
                self.Beta=1./temp/boltmann
                self.counter=0
		self.updateT=updateT
		self.collectionNumber=0
                self.memory=np.zeros((int(self.updateT),hill.bins))
                self.alphaTime=0
		self.hill=hill
                VerletHook.__init__(self)

        def init(self,iterative):
                self.meanPb=np.zeros(self.hill.bins)
                G=np.zeros(self.hill.bins)
                Grid = np.zeros((len(AGrid),len(CGrid), self.bins))
                for i in range(0,self.hill.bins):
				kernel=np.ones(self.hill.bins)
				for j,cv in enumerate()
	                                kernel[:]*=np.exp(-(self.hill.grid[i,j]-self.hill.grid[:,j])**2/2./self.hill.w[i]**2)
                                Grid[i,:]=kernel[:].copy()
                for i in range(0,self.hill.bins):
                        self.meanPb[i]=np.trapz(Grid[:,i]/float(self.hill.bins))
	def pre(self,iterative):
                pass
        def post(self,iterative):
                self.number+=1
               	self.memory[self.collectionNumber,:]=self.hill.kernel[:].copy()
                self.collectionNumber+=1
                if (self.number)%self.updateT==0:
                	self.alphaTime+=1
               		for i in range(0,self.hill.bins):
                		self.functDer[i]=-np.mean(self.memory[:,i])+self.meanPb[i]
                        	self.functDerDer[i]=self.Beta*(np.var(self.memory[:,i]))/len(self.memory[:,i])
                    	for i in range(0,self.hill.bins):
                    		self.alpha[i]=self.alpha[i]-self.hill.mu*(self.functDer[i]+self.functDerDer[i]*(self.alpha[i]-self.alpha_av[i])) #Gradient descent minimizer
              		self.alphaHist=np.vstack((self.alphaHist,self.alpha))

               		self.alpha_av[:]=((self.alphaTime-1)*self.alpha_av[:]+self.alpha[:])/self.alphaTime
			self.hill.ffPart.H=self.alpha.copy()

                	self.memory[:,:]=0.
			self.collectionNumber=0



class Hills(object):
	def __init__(self,collectiveVariable, width, grid,mu):
		def cartesian(arrays, out=None):
			arrays = [np.asarray(x) for x in arrays]
			dtype = arrays[0].dtype

    			n = np.prod([x.size for x in arrays])
    			if out is None:
        		out = np.zeros([n, len(arrays)], dtype=dtype)

    			m = n / arrays[0].size
    			out[:,0] = np.repeat(arrays[0], m)
    			if arrays[1:]:
        			cartesian(arrays[1:], out=out[0:m,1:])
        			for j in xrange(1, arrays[0].size):
            				out[j*m:(j+1)*m,1:] = out[0:m,1:]
    			return out
		'''collectiveVariable 
				an object from the CollectiveVariable class
		   atoms
			in the latter three cases, one should identify the distance, angle or dihedral using atom numbers (PYTHON STYLE starting at zero)
		   width/height
			make a guess for both the hill height and width minimizing both bias and variance of the free energy profile.
		'''
		if isinstance(collectiveVariable, list):
			self.CV=collectiveVariable
			self.w=width
			self.grid=cartesian(grid)
		else:
			self.CV=[]
			self.w=[]
			self.CV.append(collectiveVariable)
			self.w.append(width)
			self.grid=grid
		if width is None:
			'''In the near future we have to add the variational approach towards metadynamics'''
			raise NotImplementedError
		self.mu=mu
		self.bins=len(self.grid[:,0])
		self.ffPart=None
		self.Hook=None

class HillsStateH(StateItem):
	def __init__(self,hill):
		self.hill=hill
		StateItem.__init__(self,'vmtd_h')

	def get_value(self,iterative):
		return self.hill.ffPart.H

class HillsStateP(StateItem):
	def __init__(self,hill):
		self.hill=hill
		StateItem.__init__(self,'vmtd_p')

	def get_value(self,iterative):
		return self.hill.ffPart.P

class VariationalMTD(VerletIntegrator):
	def __init__(self,ff,timestep,MetaSteps,MDSteps,hill,state=None,hooks=None,velo0=None,temp0=300,scalevel0=True,time0=0.0,ndof=None,counter0=0):
		self.steps=MDSteps
		state= [] if state is None else state
		hooks= [] if state is None else hooks
		state.append(HillsState(hill))
		hill.ffPart=ForcePartMTD(ff.system,hill)
		ff.add_part(hill.ffPart)
		hill.Hook=MTDHook(hill,MDSteps/MetaSteps)
		hooks.append(hill.Hook)
		VerletIntegrator.__init__(self, ff, timestep, state=state, hooks=hooks, vel0=velo0,temp0=temp0, scalevel0=scalevel0, time0=time0, ndof=ndof, counter0=counter0)

	def runMeta(self):
		print self.steps
		self.run(self.steps)
