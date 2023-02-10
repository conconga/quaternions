#!/usr/bin/python3
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
# author:      Luciano Augusto Kruk
#
# description: Package of functions for quaternions
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>

import numpy as np
import math  as mt
from numpy import zeros,sin,cos,empty,sqrt;


#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
#  Tait-Bryan Angles:
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
class CTAITBRYAN:

    def __init__(self, *I):
        if len(I) == 1:
            I = I[0]

        self.phi = I[0]
        self.tta = I[1]
        self.psi = I[2]

    def __iter__(self):
        yield self.phi
        yield self.tta
        yield self.psi

    def __call__(self):
        return (self.phi, self.tta, self.psi)

    def to_deg(self):
        rad2deg = 180./mt.pi
        return (self.phi * rad2deg, self.tta * rad2deg, self.psi * rad2deg)

    def __getitem__(self, i):
        return (self.phi, self.tta, self.psi)[i]

    def __repr__(self):
        return "tbra( {:1.01e}, {:1.01e}, {:1.01e} )".format(self.phi, self.tta, self.psi)

#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
#  Quaternions:
#>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>--<<..>>
class CQUATERNIONS:

    def __init__(self, I):

        if 1 == 0:
            print("type      = {:s}".format(str(type(I))))
            print("__str__() = {:s}".format(I.__str__()))
            if isinstance(I, np.ndarray):
                print("shape     = {:s}".format(str(I.shape)))


        if isinstance(I, np.ndarray):
            if I.shape == (3,3):
                self.q = self._from_C(I)
            elif I.shape == (4,):
                self.q = list(I)
            else:
                print("error #1 (shape = {:s})".format(str(I.shape)))
                print(I)

        elif hasattr(I, "__iter__"):
            if isinstance(I, CTAITBRYAN) or len(I) == 3:
                self.q = self._tbra2Q(I)

            elif len(I) == 4:
                self.q = list(I)

            else:
                print("error #2");


        else:
            print("error #3")


    def __repr__(self):
        return "q( {:1.1e}, {:1.1e}, {:1.1e}, {:1.1e} )".format(self.q[0], self.q[1], self.q[2], self.q[3])

    def __iter__(self):
        for i in self.q:
            yield i

    def _tbra2Q(self, tbra):
        """
        Navigation -- from tait-bryan to Q.

        : tbra      : [phi, tta, psi] [rad]
        : output    : CQUATERNIONS
        """

        half_phi   = 0.5*tbra[0]
        half_theta = 0.5*tbra[1]
        half_psi   = 0.5*tbra[2]

        return [
            (cos(half_phi)*cos(half_theta)*cos(half_psi)) + (sin(half_phi)*sin(half_theta)*sin(half_psi)),
            (sin(half_phi)*cos(half_theta)*cos(half_psi)) - (cos(half_phi)*sin(half_theta)*sin(half_psi)),
            (cos(half_phi)*sin(half_theta)*cos(half_psi)) + (sin(half_phi)*cos(half_theta)*sin(half_psi)),
            (cos(half_phi)*cos(half_theta)*sin(half_psi)) - (sin(half_phi)*sin(half_theta)*cos(half_psi))
        ];

    def to_tbra(self):
        """
        Navigation -- from Q to tait-bryian angles.

        : output   : phi   [rad]
        : output   : theta [rad]
        : output   : psi   [rad]
        """
        q = self.q

        phi   = mt.atan2(2.0*((q[2]*q[3])+(q[0]*q[1])), (q[0]**2.0)-(q[1]**2.0)-(q[2]**2.0)+(q[3]**2.0));
        psi   = mt.atan2(2.0*((q[1]*q[2])+(q[0]*q[3])), (q[0]**2.0)+(q[1]**2.0)-(q[2]**2.0)-(q[3]**2.0));

        try:
            theta = mt.asin(2.0*((q[0]*q[2])-(q[1]*q[3])));
        except ValueError:
            print("ERRO: norm(Q) = {:f}".format(np.sqrt(np.sum(q**2))))
            theta = 0;

        return CTAITBRYAN(phi, theta, psi)

    def to_C(self):
        """
        Navigation -- from Q to C.

        If Q represents the transformation from 'a' to 'b', the matrix
        'C' represents 'Ca2b'.

        : output   : C
        """
        q = self.q

        C = np.empty((3,3));
        C[0,0] = (q[0]**2.0) + (q[1]**2.0) - (q[2]**2.0) - (q[3]**2.0);
        C[0,1] = 2.0 * ((q[1]*q[2]) + (q[0]*q[3]));
        C[0,2] = 2.0 * ((q[1]*q[3]) - (q[0]*q[2]));

        C[1,0] = 2.0 * ((q[1]*q[2]) - (q[0]*q[3]));
        C[1,1] = (q[0]**2.0) - (q[1]**2.0) + (q[2]**2.0) - (q[3]**2.0);
        C[1,2] = 2.0 * ((q[2]*q[3]) + (q[0]*q[1]));

        C[2,0] = 2.0 * ((q[1]*q[3]) + (q[0]*q[2]));
        C[2,1] = 2.0 * ((q[2]*q[3]) - (q[0]*q[1]));
        C[2,2] = (q[0]**2.0) - (q[1]**2.0) - (q[2]**2.0) + (q[3]**2.0);

        return C

    def _from_C(self, C):
        """
        from C to Q.
        """

        # <<------------------------------>>
        # << calculates C to tait-bryan:  >>
        # <<------------------------------>>
        assert(C[2,2] != 0)
        assert(C[0,0] != 0)
        assert(C[0,2]>=-1 and C[0,2]<=1)

        phi   = np.arctan2(C[1,2], C[2,2])
        theta = np.arcsin(-C[0,2])
        psi   = np.arctan2(C[0,1], C[0,0])

        # <<------------------------------>>
        # << calculates tait-bryan to Q:  >>
        # <<------------------------------>>
        return self._tbra2Q(CTAITBRYAN(phi, theta, psi))


    def __mul__(self, I):
        """
        return = C(q) x vector
        return = q1 x q2
        """

        if isinstance(I, CQUATERNIONS):
            return self._q1_prod_q2(I)
        else:
            ret = np.dot(self.to_C(), I)

            if isinstance(I, list):
                return list(ret)
            else:
                return ret


    def _q1_prod_q2(self, q2):
        """
        Navigation -- multiplies two quaternions

        q_a2c = q_b2c * q_a2b
        R_a2c = R_b2c * R_a2b

        output: np.array quaternion (self.q * q2)
        """
        q1 = self.q
        q2 = q2.q
        q3 = [
            (q1[0]*q2[0])-(q2[1]*q1[1])-(q2[2]*q1[2])-(q2[3]*q1[3]),
            (q2[0]*q1[1])+(q2[1]*q1[0])+(q2[2]*q1[3])-(q2[3]*q1[2]),
            (q2[0]*q1[2])+(q2[2]*q1[0])-(q2[1]*q1[3])+(q2[3]*q1[1]),
            (q2[0]*q1[3])+(q2[3]*q1[0])+(q2[1]*q1[2])-(q2[2]*q1[1])
        ]

        return CQUATERNIONS(q3)


    def dqdt(self, w):
        """
        The derivative of the quaternions is $\dot{q} = 1/2 .B(w).q$
        This funtion returns $\dot{q}$.
        """

        if isinstance(w, np.ndarray):
            w = np.squeeze(w)

        K      = 1e1
        cq     = np.asarray(self.q).reshape((4,1))
        epslon = 1.0 - np.sum(cq**2.0)

        B = np.asarray([
            [   0, -w[0], -w[1], -w[2]],
            [w[0],     0,  w[2], -w[1]],
            [w[1], -w[2],     0,  w[0]],
            [w[2],  w[1], -w[0],     0]
        ])

        dq = (0.5 * np.dot(B,cq)) + (K*epslon*cq)

        return list(dq.squeeze())

#####################################################
if __name__ == "__main__":
    print("init() with quaternions")
    q = CQUATERNIONS([ 0.2, 0.3, 0.4, 0.5 ])
    print(q)

    print("init() with tait-bryan")
    q = CQUATERNIONS([ 0,0,0 ])
    print(q)

    print("tests with 'to_tbra()'")
    rad2deg = 180./3.14159265359
    print( CQUATERNIONS([ 10./rad2deg, 0, 0 ]).to_tbra().to_deg() )
    print( CQUATERNIONS([ 0, 10./rad2deg, 0 ]).to_tbra().to_deg() )
    print( CQUATERNIONS([ 0, 0, 10./rad2deg ]).to_tbra().to_deg() )

    print("tests with to_C()")
    print(CQUATERNIONS([0,0,0]).to_C())
    print("phi = 90[deg]")
    print(CQUATERNIONS([90/rad2deg,0,0]).to_C())

    vector = [1.,0,0]
    print("vector = {:s}".format(vector.__str__()))
    print("phi = +45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([45/rad2deg,0,0])*vector).__str__()))
    print("tta = +45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([0,45/rad2deg,0])*vector).__str__()))
    print("psi = +45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([0,0,45/rad2deg])*vector).__str__()))
    print("psi = -45[deg], C(q) x vector = {:s}".format((CQUATERNIONS([0,0,-45/rad2deg])*vector).__str__()))

    print("from C to Q")
    tbra = CTAITBRYAN([20, 30, 40])
    print("from tbra = {:s}".format(str(tbra)))
    tbra = CTAITBRYAN( [i/rad2deg for i in tbra] )
    C     = CQUATERNIONS(tbra).to_C()
    Q     = CQUATERNIONS(C)
    print("result = {:s}".format( [i*rad2deg for i in Q.to_tbra()].__str__() ))

    print("q1 x q2")
    print(CQUATERNIONS([0,0,0]))
    q1 = CQUATERNIONS( [0,90/rad2deg,0] )
    q2 = CQUATERNIONS( [0,0,-90/rad2deg] )
    q3 = CQUATERNIONS( [0,-90/rad2deg,0] )
    q4 = CQUATERNIONS( [90/rad2deg,0,0] )
    print(q4*q3*q2*q1)

    q_a2b = CQUATERNIONS((-10./rad2deg, 33./rad2deg, -55./rad2deg))
    q_b2c = CQUATERNIONS((44./rad2deg, -38./rad2deg, 77./rad2deg))

    print("C_a2c = C_b2c . C_a2b =")
    print((q_b2c * q_a2b).to_C())
    print(np.dot(q_b2c.to_C(), q_a2b.to_C()))
    print("  = ...")
    print((q_b2c * q_a2b).to_C())

    print("yielding..")
    q = CQUATERNIONS((3,4,5))
    print(list(q))
    print(list(q))
    print(list(q))

    #----------------------#
    # some dynamic tests:
    #----------------------#
    from   scipy.integrate import odeint;
    from   numpy           import dot;
    print()

    #  I: inertial frame
    #  b: body frame
    qI2b = CQUATERNIONS((0,0,0))

    # angular rotation between I and b:
    # \omega_{Ib}^I
    w = np.asarray([2./rad2deg,  0,   0]).reshape((3,1))

    def eqdiff(q,t,w):
        qI2b = CQUATERNIONS(q)
        dqdt = qI2b.dqdt(qI2b * w)
        return dqdt

    # a vector described at I:
    F = np.asarray([0,0,1]).reshape((3,1))
    print("F = ")
    print(F.T)

    for t in [1,5,20,90]:
        # after t seconds, the quaternions should be:
        y = odeint(eqdiff, list(qI2b), [0,t], (w,))[1,:]
        # with these tait-bryan angles:
        tbra = CQUATERNIONS(y).to_tbra()

        # and described at b:
        F_b = (CQUATERNIONS(y) * F).squeeze()
        print("F_b(phi = {:1.03f}) = [{:1.03f} {:1.03f} {:1.03f}]".format(rad2deg*tbra[0], F_b[0], F_b[1], F_b[2]))

    #----------------------#
    # transformation tests:
    #----------------------#
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

#####################################################
