"""Signed LIF S-LCA port of Du et al.'s SLCA_LIF / LIF classes.

Source: XuexingDu/Spiking-LCA, Neuron_models.py, blob
4ace2e2c4b46f68dfccc46738449a8e2f861cebe. Signed lifting uses [A,-A].
See SLCA_NOTES.md for the literal current averaging rule and parameter choices.
"""
import math
import torch
from torch import Tensor
from typing import Tuple


@torch.jit.script
def advance(b:Tensor,gram:Tensor,synaptic:Tensor,average:Tensor,voltage:Tensor,
            previous_spikes:Tensor,last_spike:Tensor,counts:Tensor,
            start:int,steps:int,dt:float,penalty:float,resistance:float,
            tau:float,t_ref:float)->Tuple[Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]:
    n=gram.size(0)
    syn_decay=math.exp(-dt)
    membrane_decay=math.exp(-dt/tau)
    for k in range(start,start+steps):
        t=k*dt
        signed=previous_spikes[:,:n]-previous_spikes[:,n:]
        projected=torch.matmul(signed,gram)
        # Signed Gram minus identity: same-sign inhibition and opposite-sign excitation.
        synaptic=synaptic*syn_decay+torch.cat((projected,-projected),dim=1)-previous_spikes
        # Literal official SLCA_LIF rule: t is physical simulation time, not step index.
        average=average*(t/(t+1.0))+(b-synaptic)/(t+1.0)
        rate=torch.relu(average-penalty)
        # Official R=tau; expose only this family so the inverse gain remains consistent.
        # Default t_ref=0 avoids rate ceilings for the user's signed U[0,4) signals.
        safe_rate=rate.clamp_min(1e-12)
        current=1.0/(resistance*(-torch.expm1((t_ref-1.0/safe_rate)/tau)))
        current=torch.where(rate>1e-7,current,torch.full_like(current,1.0/resistance))
        candidate=voltage*membrane_decay+resistance*current*(1.0-membrane_decay)
        refractory=(t-last_spike)<=t_ref
        candidate=torch.where(refractory,voltage,candidate)
        spikes=(candidate>1.0).to(candidate.dtype)
        last_spike=torch.where(spikes>0,torch.full_like(last_spike,t),last_spike)
        voltage=torch.where(spikes>0,torch.zeros_like(voltage),candidate)
        counts=counts+spikes
        previous_spikes=spikes
    return synaptic,average,voltage,previous_spikes,last_spike,counts


class SignedSLCA(torch.nn.Module):
    def __init__(self,matrix,penalty=0.1,dt=0.01,tau=10.0):
        super().__init__()
        if not 0<dt<=0.1 or penalty<0 or tau<=0:
            raise ValueError('Require 0 < dt <= 0.1, nonnegative penalty and positive tau')
        if not torch.allclose(matrix.norm(dim=0),torch.ones(matrix.shape[1],device=matrix.device,dtype=matrix.dtype),atol=1e-5):
            raise ValueError('S-LCA requires unit-norm matrix columns')
        self.register_buffer('matrix',matrix.clone())
        gram=matrix.t()@matrix
        gram.fill_diagonal_(1.0)
        self.register_buffer('gram',gram)
        self.n=matrix.shape[1]
        self.penalty,self.dt,self.tau=float(penalty),float(dt),float(tau)
        self.resistance=self.tau
        self.t_ref=0.0

    def initial_state(self,y):
        projection=y@self.matrix
        b=torch.cat((projection,-projection),dim=1)
        z=torch.zeros_like(b)
        return b,(z.clone(),z.clone(),z.clone(),z.clone(),torch.full_like(z,-1e7),z.clone())

    def readout(self,counts,steps):
        return (counts[:,:self.n]-counts[:,self.n:])/(steps*self.dt)

    def forward(self,y,steps:int):
        b,state=self.initial_state(y)
        state=advance(b,self.gram,*state,0,steps,self.dt,self.penalty,self.resistance,self.tau,self.t_ref)
        return self.readout(state[-1],steps)

    @torch.no_grad()
    def snapshots(self,y,horizons):
        if not horizons or list(horizons)!=sorted(set(horizons)) or horizons[0]<=0:
            raise ValueError('Horizons must be positive, unique, increasing step counts')
        b,state=self.initial_state(y)
        previous=0
        for steps in horizons:
            state=advance(b,self.gram,*state,previous,steps-previous,self.dt,self.penalty,self.resistance,self.tau,self.t_ref)
            counts=state[-1]
            yield steps,self.readout(counts,steps),counts.sum(dim=1)
            previous=steps

    def synaptic_cost(self,spikes):
        m,n=self.matrix.shape
        return m*n,spikes*(2*n-1)
