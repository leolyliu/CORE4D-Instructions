### SMPL-X

SMPL-X model parameters (55 * 3 DoF):

* 21*3 DoF body pose
* 15*3 DoF left hand pose
* 15*3 DoF right hand pose
* 1*3 DoF jaw pose
* 1*3 DoF left eye pose
* 1*3 DoF right eye pose
* 1*3 DoF global orientation (pelvis orientation, note: pelvis is not (0, 0, 0) in SMPL-X canonical space!)
* 1*3 DoF global translation

(0, 0, 0): breast, not pelvis (pelvis defines the orientation)!

number of joints: 55

betas: 10 DoF

number of PCA components: 6

number of expression coefficients: 10

* vertices shape = (10475, 3)
* joints shape = (127, 3)
