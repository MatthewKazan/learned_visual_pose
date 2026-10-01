## Pose notation:
T_AB transforms a point expressed in frame B into frame A

p_A = T_AB p_B

makes composition intuitive
T_WC = T_WB @ T_BC

Transform representation:
T = [ R  t ]
[ 0  1 ]

Twist ordering:
xi = [rho, phi]
translation first, rotation second

Rotation convention:
right-handed coordinate systems

Perturbation convention:
implement both eventually,
but start with left perturbations

T' = Exp(delta_xi) T

## Camera conventions:

Camera coordinate frame (OpenCV convention):
+X right
+Y down
+Z into the scene (forward from the camera)

Pixel coordinates:
u grows to the right
v grows downward from the top-left corner of the image

Intrinsics:
fx, fy = focal length in pixels
cx, cy = principal point in pixels (image location of the optical center)

Projection:
u = fx * X/Z + cx
v = fy * Y/Z + cy

Back-projection (pixel + depth to 3D):
X = Z * (u - cx) / fx
Y = Z * (v - cy) / fy

project() takes points in the camera frame, not world frame.
Callers do camera.project(T_CW * P_W) to project a world-frame point.

Distortion:
not modeled — assumes undistorted intrinsics (e.g. TartanAir renders).

Other camera conventions are converted at the frame source and never leak above it:
TartanAir poses are NED (T_WC @ T_ned_cv, data_utils/dataset.py).
Habitat cameras are +X right, +Y up, looking down -Z (T_WC @ T_hab_cv, a flip
of Y and Z, data_utils/hm3d_sequence.py). Habitat depth is planar, so it is
OpenCV Z unchanged.
