#First I imported some useful packages 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

#Here I defined the range of r and t that I used for the simulation.
#For both ranges I used linspace of numpy which creates an array with values equally spaced.
r=np.linspace(0.1,5.1,251) #r between 0.1 and 5.1
t=np.linspace(0,4,2001) #t between 0 and 4 

#We can take dt and dr as the positive difference of consecutive elements of t and r.
dt=t[1]-t[0] #dt=0.002
dr=r[1]-r[0] #dr=0.02

#s and rm are the sigma and the center of the Gaussian of the initial condition for the surface density.
s=0.1
rm=2.55
Sig=(1/(s*np.sqrt(2*np.pi)))*np.exp(-((r-rm)/s)**2) #Initial surface density

#Fixed viscosity term 
nu=0.5

#I defined A to be the matrix for solving the diffusion part with the corresponding beta.
nn=len(r) #dimension of the matrix.
beta=3*nu*dt/(dr**2)
A=np.eye(nn)*(1.0+2.0*beta)+np.eye(nn,k=1)*-beta+np.eye(nn,k=-1)*-beta #tridiagonal matrix

#I defined Signd as an array of dimensions length(t) x length(Sig) in order to save the surface density profile of each timestep.
Signd=np.zeros([len(t),len(Sig)])
Signd[0]=Sig #The initial condition of the surface density is the first element of Signd

#This loop solves the momentum equation by solving the diffusion part first and then the advection as described in equations (13) and (14) of my solutions.  
for n in range(len(t)-1):  #This loop is for each timestep.
    diff=np.linalg.inv(A)@Signd[n] #This solves the diffusion part.
    for j in range(1,len(Sig)-1): #This loop is for each grid cell to solve the advection.
        #Here I take the diffusion part already obtained into the advection solution to obtain the final surface density profile at time t_(n+1)
        Signd[n+1][j]=0.5*(diff[j+1]+diff[j-1])+(9./4)*nu*(dt/dr)*(1/r[j])*(diff[j+1]-diff[j-1]) #I save the result of each timestep in Signd  
    #Here I update the boundaries to consider outflow.
    Signd[n+1][0]=Signd[n+1][1] 
    Signd[n+1][-1]=Signd[n+1][-2]

#Here I defined the figure for the visualization
fig,ax = plt.subplots(figsize=(6,4))

#The function animate updates the figure.
def animate(i):
    ax.clear()
    ax.set_xlim(0,5.2)    
    ax.set_ylim(-0.1,0.8)
    initial, = ax.plot(r,Signd[0],'--' ,color='dodgerblue',label=r'$\tilde{\Sigma}(\tilde{t}=0)$') #I make a plot of the initial surface density at each update in order to compare with the current surface density
    ev, = ax.plot(r,Signd[0::5][i], color='blue',label=r'$\tilde{\Sigma}(\tilde{t})$') #Plot of the current surface density
    ax.text(3.5,0.5,r'$\tilde{t}=$'+str(round(t[0::5][i],3))) #This adds a text which indicates the current time
    ax.set_xlabel(r'$\tilde{r}=r/r_{o}$')
    ax.set_ylabel(r'$\tilde{\Sigma}=\Sigma/\Sigma_{o}$')
    ax.legend()
    return initial, ev
#Note that I specified that the plot updates each every 5 profiles since my timestep is very small, in this way the visualization goes quicker.        
ani = FuncAnimation(fig, animate, frames=400, repeat=True) #This creates the animation. Note that repeat=True, then the animation will restart and won't stop until you close the window desplayed. You can set it to False to close the animation automatically when it ends.
#ani.save("diff.gif", dpi=150, writer=PillowWriter(fps=10)) #You can uncomment this part to save the animation as .gif file 
plt.show()
