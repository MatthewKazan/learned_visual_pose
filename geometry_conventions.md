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