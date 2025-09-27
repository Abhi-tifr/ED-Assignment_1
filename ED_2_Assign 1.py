#!/usr/bin/env python
# coding: utf-8

# In[60]:


# Question 1.a

# Residual for Gauss's Law for Electric field 

import numpy as np
import matplotlib.pyplot as plt

beta=[0.01,0.5,0.9999] # Value of beta

z_value=np.linspace(0, 2*np.pi, 400) # value of z from 0 to 2*pi

# defining the difference delta 

def f(beta,z):
    delta=beta*np.cos(z)
    return delta

# Evaluating delta for different value of beta 

for i in beta:
    delta_value=[f(i,z) for z in z_value]
    plt.plot(z_value,delta_value,label=r"$\beta$ = {:.4f}".format(i))
    
# Ploting delta vs z
    
plt.xlabel("z(radian)",fontsize=14)
plt.ylabel(r"$\Delta$",fontsize=14)
plt.title("Plot of $\Delta$ for Gauss's Law for E ")
plt.legend()
plt.grid(True)
plt.show()






#Residual for Ampere's Maxwell Law

c = 1.0
x_value = np.linspace(0, 2*np.pi, 400)  # 0 to 2π


V_vals = [0.01, 0.5, 0.9999] # value of beta 

# defining the difference delta 

def f(V,x):
    delta=(V-V**2)*np.cos(x)
    return delta

# Evaluating delta for different value of beta 

for i in V_vals:
    delta_value=[f(i,x) for x in x_value]
    plt.plot(x_value,delta_value,label=r"$\beta$ = {:.4f}".format(i))
    
# Ploting delta vs x    

plt.xlabel("x (radians)",fontsize=14)
plt.ylabel(r"$\Delta$",fontsize=14)
plt.title("Ampère residual Δ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
        


# In[ ]:





# In[ ]:





# In[61]:


# Question 2.a 

import numpy as np
import matplotlib.pyplot as plt

beta=[0.01,0.5,0.9999] 

for i in beta:
    
    psi=np.arctanh(i) # rapidity
    
    ct_prime_axis=np.linspace(0,1,100)
    x_prime_axis=np.linspace(0,1,100)

    # Transformation of x and ct axis in terms of rapidity 
    
    def f1(ct_prime,x_prime):
        t= ct_prime*np.cosh(psi)+ x_prime*np.sinh(psi) 
        return t

    def f2(ct_prime,x_prime):
        x= ct_prime*np.sinh(psi)+ x_prime*np.cosh(psi)
        return x

    # Transformed ct axis (x_prime=0)
    
    t1=f1(t_prime_axis,0)
    x1=f2(t_prime_axis,0)

    # Tarnsformed x axis (t_prime=0)
    
    t2=f1(0,x_prime_axis)
    x2=f2(0,x_prime_axis)
 
    # Plotting original axes i.e x and ct axis 
    
    plt.axhline(0, color='red', lw=1)  # x-axis
    plt.axvline(0, color='blue', lw=1)  # ct-axis

    plt.text(0.05, 0.9, "ct axis", color='blue', fontsize=12)
    plt.text(0.9, 0.05, "x axis", color='red', fontsize=12)
         
    # Plotting the transformed axes i.e x_prime and ct_prime 
    
    plt.plot(x1,t1,label="ct' axis") 
    plt.plot(x2,t2,label="x' axis ")
    plt.title(r"$\beta$ = {:.4f}".format(i))
    plt.legend()
    plt.grid()       
    plt.show()


# In[ ]:





# In[ ]:





# In[62]:


#Question 3.

import numpy as np
import matplotlib.pyplot as plt


# let the ratio of V  and c  be "beta"

beta=np.arange(0.01,0.9999,0.00001) # storing the differnet value of "beta"


def f(beta):
    y=1/(np.sqrt(1-(beta)**2))  #  let the ratio of tau (given life time ) and tau_lab (life time measured in lab frame) = y
    return y 

y_value=[]   # stores the ratio of tau and t_lab

for i in beta:
    y_value.append(f(i))
    
# Plotting the tau_lab/tau for different value of beta 

plt.plot(beta,y_value)
plt.title(r"$\frac{\tau_{lab}}{\tau}$ vs $\beta$" , fontsize=20)
plt.xlabel(r"$\beta$", fontsize=20)
plt.ylabel(r"$\frac{\tau_{lab}}{\tau}$", fontsize=20)
plt.grid()


# Now let us mark the point on the graph for the given value of beta in the question 

x_point = [0.01,0.5,0.9999] # values of beta given in the question


for i in x_point:
    y=f(i)
    plt.scatter(i,y,color='black', marker='*')
    
    plt.text(i + 0.02, y + 4, f"β = {i}\n" + r"$\frac{\tau_{lab}}{\tau}$" + f" = {y:.2f}",
         fontsize=12,
         backgroundcolor='white')

plt.show()


# In[ ]:




