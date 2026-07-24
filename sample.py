import numpy as np

from guidance.nadir_pointing import NadirPointingGuidance

r = np.array([7000e3, 0, 0])
v = np.array([0, 7500, 0])

g = NadirPointingGuidance()

q, w = g.compute(r, v)

print("Quaternion")
print(q)

from models.dynamics.quaternion import quaternion_to_dcm

C = quaternion_to_dcm(q)

print("\nDCM")
print(C)

print("\nColumns")

print("Body X =", C[:,0])
print("Body Y =", C[:,1])
print("Body Z =", C[:,2])