'''Ruben Demuynck + yaff acknowledgement'''

from yaff import *
import numpy as np
from mpi4py import MPI
from molmod.constants import boltzmann
import MTD

class REHook(VerletHook):
	def __init__(self,rank,comm,n_ensembles,temp,MDsteps,hills,start=0,step=1):
		self.nems=n_ensembles
		self.comm=comm
		self.rank=rank
		self.updateT=MDsteps
		self.counter=0
		self.temp=temp					####FOR NOW WE ASK THE TEMPERATURE IN THE FUTURE WE WOULD PREFER AN AUTOMATIC DETERMINATION OF PRESSURE AND TEMPERATURE AND UMBRELLA
		self.hills=hills
		VerletHook.__init__(self,start,step)

	def init(self,iterative):
		pass
	def pre(self,iterative):
		pass

	def post(self,iterative):
		def compute(pos, rvecs,vel):
                	iterative.pos[:] = pos
                	iterative.rvecs[:]=rvecs
                	iterative.vel[:] = vel
                	iterative.gpos[:] = 0.0
                	iterative.ff.update_rvecs(rvecs)
                	iterative.ff.update_pos(pos)
                	iterative.epot = iterative.ff.compute(iterative.gpos)
                	iterative.acc = -iterative.gpos/iterative.masses.reshape(-1,1)

		self.counter+=1
		if self.counter%self.updateT==0:
			self.comm.Barrier()
			#FOR NOW WE CONSIDER NONE NONE HILLS
			CVS=np.zeros((self.nems,len(self.hills[self.rank].ffPart.B[:])))
			Heavis=np.zeros((self.nems,len(self.hills[self.rank].ffPart.H[:])))
			self.comm.Allgather(self.hills[self.rank].ffPart.B[:],CVS) 
			self.comm.Allgather(self.hills[self.rank].ffPart.H[:],Heavis) 	
			
			Vmeta=np.zeros((self.nems))
			for i in range(0,self.nems):
				if self.hills[i] is None:
					Vmeta[i]=0
				else:
					self.hills[i].B=CVS[i][:]
					self.hills[i].H=Heavis[i][:]
					Vmeta[i]=self.hills[i].ffPart.determineF(iterative.ff.system)
			allInfo=[self.rank,self.temp,iterative.ff.energy,iterative.ff.system.pos,iterative.ff.system.cell.rvecs,iterative.vel,Vmeta] 			##A MORE ELEGANT WAY IS THE CREATION OF A MPI_DATATYPE CORRESPONDING TO ITERATIVE CLASS
			self.rootdata=self.comm.gather(allInfo,root=0)
			if self.rank==0: 
				self.collectAndChange()
			self.comm.Barrier()
			publicdata=self.comm.scatter(self.rootdata,root=0)
			compute(publicdata[3],publicdata[4],publicdata[5])			

	def collectAndChange(self):
		ranks=np.arange(0,self.nems)
		pranks=np.random.permutation(ranks)
		sorted(self.rootdata, key=lambda x: x[0])													##NOT REALLY NECESSARY HENCE GATHER ALWAYS INSTITIONALIZES SORTED DATA
		condition1=np.array([np.exp((1./self.rootdata[i][1]-1./self.rootdata[j][1])*((self.rootdata[j][2]-self.rootdata[j][j])-(self.rootdata[i][2]-self.rootdata[i][i]))/boltzmann+1./self.rootdata[i][1]*(self.rootdata[i][i]-self.rootdata[j][i])/boltzmann+1./self.rootdata[j][1]*(self.rootdata[j][j]-self.rootdata[i][j])/boltzmann) for i,j in zip(ranks,pranks)])
		for i,match in enumerate(np.where(np.random.rand()<condition1)[0]):
			if not pranks[match]==ranks[match]:
				self.rootdata[pranks[match]][3],self.rootdata[ranks[match]][3]=self.rootdata[ranks[match]][3],self.rootdata[pranks[match]][3]
				self.rootdata[pranks[match]][4],self.rootdata[ranks[match]][4]=self.rootdata[ranks[match]][4],self.rootdata[pranks[match]][4]
				self.rootdata[pranks[match]][5],self.rootdata[ranks[match]][5]=np.sqrt(self.rootdata[pranks[match]][2]/self.rootdata[ranks[match]][2])*self.rootdata[ranks[match]][5],np.sqrt(self.rootdata[ranks[match]][2]/self.rootdata[pranks[match]][2])*self.rootdata[pranks[match]][5]
					

class BiasExchange(object):
	def __init__(self,replica,temp,hills,MDsteps,Metasteps,REsteps):
		'''replicas
			a list of VerletIntegrator objects, the size of the list should match the number of processors
		   temp
			a list of the temperatures (for now we only consider replica exchange for different temperatures)
		   MDsteps
			the number of MDsteps between each replica exchange step
		   REsteps
			the number of replica exchange steps
		'''
		###MPI information
		comm = MPI.COMM_WORLD   #Defines the default communicator
		num_procs = comm.Get_size()  #Stores the number of processes in size
		rank = comm.Get_rank()  #Stores the rank (pid) of the current process
		stat = MPI.Status()
                state=[]
                for i,hill in enumerate(hills):
			if i is not rank:
	                        state.append(MTD.HillsState(hill))
	                        hill.ffPart=MTD.ForcePartMTD(replica.ff.system,hill,Metasteps)
        	                hill.Hook=MTD.MTDHook(hill,MDsteps*REsteps/Metasteps)
		replica.hooks.append(REHook(rank,comm,num_procs,temp,MDsteps,hills))

		
		replica.run(MDsteps*REsteps)
