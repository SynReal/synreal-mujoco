import numpy as np

class step_skipper:

    def __init__(self ):
        self.last_rigid_body_transform = None
        self.last_piece_pos = None

        self.curr_rigid_body_transform = None
        self.curr_piece_pos = None

        self.rigid_body_ref_pos = None;

        self.dt = 0

        self.norm_threshold = 1e-3  # velocity bellow 1mm/s consider stay still
        self.box_threshold = 1e-2   # box within 1cm consider near

    def set_rigidbody_refpos(self, pos):
        self.rigid_body_ref_pos = pos

    def set_pos(self, piece_pos, rigidbody_mat, rigidbody_translate, dt):

        self.last_rigid_body_transform = self.curr_rigid_body_transform
        self.last_piece_pos = self.curr_piece_pos

        self.curr_rigid_body_transform = ( rigidbody_mat, rigidbody_translate)
        self.curr_piece_pos = piece_pos

        self.dt = dt

    def safe_to_skip(self):
        return  \
        self.last_piece_pos != None  \
        and self.last_rigid_body_transform != None \
        and not self._is_aabb_near_between_cloth_rigidbody()  \
        and  self._cloth_is_almost_stay_still()


    def get_curr_rigid_body_transform(self):
        return self.curr_rigid_body_transform

    def get_last_rigid_body_transform(self):
        return self.last_rigid_body_transform

    def _is_aabb_near_between_cloth_rigidbody(self):
        box_rb = self._compute_aabb_rigidbody()
        box_cloth = self._compute_aabb_cloth()
        ret = self._box_near(*box_cloth, *box_rb)
        #print(f'box intersected: {ret}')
        return ret

    def _cloth_is_almost_stay_still(self):

        norm = self._compute_cloth_norm()
        #print(f'cloth norm: {norm} {self.norm_threshold}')
        return norm < self.norm_threshold

    def _compute_cloth_norm(self):
        x0 = np.concatenate(self.last_piece_pos,axis=0)
        x1 = np.concatenate(self.curr_piece_pos,axis=0)
        velocity = np.linalg.norm( (x0 - x1) / self.dt ,axis=1)
        return np.max(velocity)

    def _compute_aabb_rigidbody(self):
        stack_x = []
        for x,r,t in zip(self.rigid_body_ref_pos,*self.curr_rigid_body_transform):
            r = r.reshape(3,3) # check this
            stack_x.append(self._compute_rigid_body_current_pos(x,r,t))

        stack_x = np.concatenate(stack_x,axis=0)

        return self._compute_aabb(stack_x)

    def _compute_aabb_cloth(self):
        return self._compute_aabb(np.concatenate(self.curr_piece_pos,axis=0))

    def _compute_aabb(self,x):
        min_x=np.min(x,axis=0)
        max_x=np.max(x,axis=0)
        return min_x, max_x

    def _box_near(self, min0, max0, min1, max1):
        n=len(min0)
        for i in range(n):
            if min0[i] > max1[i] + self.box_threshold or max0[i] < min1[i] - self.box_threshold :
                return False
        return True

    def _compute_rigid_body_current_pos(self, x, rot, translate):
        return translate +  x @ rot
