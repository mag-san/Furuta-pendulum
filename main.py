import numpy as np
import math

##variables##
#arm
theta = 0 #arm angle
m_theta = 2 #arm mass
r = 3 #arm length

#pendulum
alpha = 0 #angle
m_alpha = 3 #mass
L = 5 #length
I = L/2 #half length

#other
g = 9.81 #gravitational acceleration
delta_h = (L/2)*(1-math.cos(alpha)) #height difference center pendulum rel. to ref. pos.

