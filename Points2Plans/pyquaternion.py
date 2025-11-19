"""Minimal shim for `pyquaternion.Quaternion` used in tests.

This provides just enough functionality for `relational_dynamics.utils.math_util`
to use Quaternion(matrix=...), Quaternion.random(), and Quaternion(...).transformation_matrix.
This is NOT a full replacement for the pyquaternion package.
"""
import numpy as np


def _rotation_matrix_from_quat(w, x, y, z):
    # compute 3x3 rotation matrix from quaternion (w,x,y,z)
    R = np.zeros((3,3))
    R[0,0] = 1 - 2*(y*y + z*z)
    R[0,1] = 2*(x*y - z*w)
    R[0,2] = 2*(x*z + y*w)
    R[1,0] = 2*(x*y + z*w)
    R[1,1] = 1 - 2*(x*x + z*z)
    R[1,2] = 2*(y*z - x*w)
    R[2,0] = 2*(x*z - y*w)
    R[2,1] = 2*(y*z + x*w)
    R[2,2] = 1 - 2*(x*x + y*y)
    return R


class Quaternion:
    def __init__(self, *args, **kwargs):
        # support Quaternion(matrix=...) or Quaternion(w, x, y, z)
        if 'matrix' in kwargs:
            M = kwargs['matrix']
            # extract rotation 3x3 from homogeneous matrix
            R = np.array(M, dtype=float)[:3, :3]
            # convert rotation matrix to quaternion using trace method
            t = np.trace(R)
            if t > 1e-8:
                s = 0.5 / np.sqrt(t + 1.0)
                w = 0.25 / s
                x = (R[2,1] - R[1,2]) * s
                y = (R[0,2] - R[2,0]) * s
                z = (R[1,0] - R[0,1]) * s
            else:
                # fallback for near-zero trace
                diag = np.argmax([R[0,0], R[1,1], R[2,2]])
                if diag == 0:
                    s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                    w = (R[2,1] - R[1,2]) / s
                    x = 0.25 * s
                    y = (R[0,1] + R[1,0]) / s
                    z = (R[0,2] + R[2,0]) / s
                elif diag == 1:
                    s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                    w = (R[0,2] - R[2,0]) / s
                    x = (R[0,1] + R[1,0]) / s
                    y = 0.25 * s
                    z = (R[1,2] + R[2,1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                    w = (R[1,0] - R[0,1]) / s
                    x = (R[0,2] + R[2,0]) / s
                    y = (R[1,2] + R[2,1]) / s
                    z = 0.25 * s
            self.w = float(w)
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
        else:
            # accept Quaternion(w, x, y, z) or Quaternion(x, y, z, w)
            if len(args) == 4:
                # assume (w, x, y, z)
                self.w, self.x, self.y, self.z = [float(a) for a in args]
            else:
                raise ValueError('Quaternion requires either matrix= or 4 components')

    @property
    def transformation_matrix(self):
        R = _rotation_matrix_from_quat(self.w, self.x, self.y, self.z)
        T = np.eye(4)
        T[:3, :3] = R
        return T

    @staticmethod
    def random():
        # return identity quaternion for simplicity
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    def __repr__(self):
        return f"Quaternion(w={self.w}, x={self.x}, y={self.y}, z={self.z})"
