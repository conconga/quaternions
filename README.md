# quaternions
quaternions for navigation

# some examples
## initialize the object with quaternions:
```
    q = CQUATERNIONS([ 0.2, 0.3, 0.4, 0.5 ])
```
## initialize the object with tait-bryan angles (phi, theta, psi):
```
    q = CQUATERNIONS([ 0,0,0 ])
```
## convert from quaternions to tait-bryan angles and show in [degrees]:
```
    rad2deg = 180./3.14159265359
    print( CQUATERNIONS([ 10./rad2deg, 0, 0 ]).to_tbra().to_deg() )
    print( CQUATERNIONS([ 0, 10./rad2deg, 0 ]).to_tbra().to_deg() )
    print( CQUATERNIONS([ 0, 0, 10./rad2deg ]).to_tbra().to_deg() )
```
## convert from quaternions to the rotation matrix:
```
    print(CQUATERNIONS([0,0,0]).to_C())
    print(CQUATERNIONS([90/rad2deg,0,0]).to_C())
```
## transform a vector from one frame to another:
```
    vector = [1.,0,0]
    print("phi = +45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([45/rad2deg,0,0])*vector).__str__()))
    print("tta = +45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([0,45/rad2deg,0])*vector).__str__()))
    print("psi = +45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([0,0,45/rad2deg])*vector).__str__()))
    print("psi = -45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([0,0,-45/rad2deg])*vector).__str__()))
```
## initialize from tait-bryan angles and rotation matrix:
```
    tbra = CTAITBRYAN([20, 30, 40])
    tbra = CTAITBRYAN( [i/rad2deg for i in tbra] )
    C     = CQUATERNIONS(tbra).to_C()
    Q     = CQUATERNIONS(C)
    print("result = {:s}".format( [i*rad2deg for i in Q.to_tbra()].__str__() ))
```
## cascade of rotation quaternions:
```
    q_a2b = CQUATERNIONS((-10./rad2deg, 33./rad2deg, -55./rad2deg))
    q_b2c = CQUATERNIONS((44./rad2deg, -38./rad2deg, 77./rad2deg))

    print("C_a2c = C_b2c . C_a2b =")
    print((q_b2c * q_a2b).to_C())
    print(np.dot(q_b2c.to_C(), q_a2b.to_C()))
    print("  = ...")
    print((q_b2c * q_a2b).to_C())


    q_a2b = CQUATERNIONS((30/rad2deg,1,0.5))
    Ra2b  = q_a2b.to_C()

    q_b2c = CQUATERNIONS((20/rad2deg,-1,-0.3))
    Rb2c  = q_b2c.to_C()

    q_a2c = CQUATERNIONS(np.dot(Rb2c, Ra2b))
    print("q_a2c = {:s}".format(str(q_a2c)))
    print("      = {:s}".format(str(q_b2c * q_a2b)))
    print("tbra(q_a2c) =")
    print(q_a2c.to_tbra())
    print((q_b2c * q_a2b).to_tbra())
```
## integral and derivative:
```
    def eqdiff(q,t,w):
        qI2b = CQUATERNIONS(q)
        dqdt = qI2b.dqdt(qI2b * w)
        return dqdt

    y = odeint(eqdiff, list(qI2b), [0,t], (w,))[1,:]
    tbra = CQUATERNIONS(y).to_tbra()
```
