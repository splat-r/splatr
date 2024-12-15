import numpy as np
class CheckSuccessfulAction():
    '''
    (Hack) Check action success by comparing RGBs. 
    TODO: replace with a network
    '''
    def __init__(self, rgb_init, H, W, perc_diff_thresh = 0.05, controller=None, use_GT_action_success=False):
        '''
        rgb_init: the rgb image from the spawn viewpoint W, H, 3
        This class does a simple check with the previous image to see if it completed the action 
        '''
        self.rgb_prev = rgb_init
        self.perc_diff_thresh = perc_diff_thresh
        self.H = H
        self.W = W
        self.controller = controller
        self.use_GT_action_success = use_GT_action_success

    def update_image(self, rgb):
        self.rgb_prev = rgb

    def check_successful_action(self, rgb):
        if self.use_GT_action_success:
            success = self.controller.last_event.metadata["lastActionSuccess"]
        else:
            num_diff = np.sum(np.sum(self.rgb_prev.reshape(self.W*self.H, 3) - rgb.reshape(self.W*self.H, 3), 1)>0)
            # diff = np.linalg.norm(self.rgb_prev - rgb)
            # print(num_diff)
            print("val : ", num_diff)
            print("thresh : ", self.perc_diff_thresh*self.W*self.H)
            if num_diff < self.perc_diff_thresh*self.W*self.H:
                success = False
            else:
                success = True
            # self.rgb_prev = rgb
        return success

    def check_successful_action_pos(self, rgb):
        raise NotImplementedError("TODO: implement check_successful_action_pos")

