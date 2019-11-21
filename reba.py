# -*- coding: utf-8 -*-
# ---------------------

import numpy as np
import utils


class RebaScore:
    '''
    Class to compute REBA metrics

    Pose:
          [0]: Head
          [1]: Neck
          [2, 3, 4, 14]: Left arm + (optional)left hand
          [5, 6, 7, 15]: Right arm + (optional)right hand
          [8, 9, 10]: Left leg
          [11, 12, 13]: Right leg
    '''
    def __init__(self):
        # Table A ( Neck X Trunk X Legs)
        self.table_a = np.zeros((3, 5, 4))
        # Table B ( UpperArm X LowerArm X Wrist)
        self.table_b = np.zeros((6, 2, 3))
        # Table C ( ScoreA X ScoreB)
        self.table_c = np.zeros((12, 12))

        # Body Params
        self.body = {'neck_angle': 0, 'neck_side': False,
                     'trunk_angle': 0, 'trunk_side': False,
                     'legs_walking': False, 'legs_angle': 0,
                     'load': 0}

        # Arms Params
        self.arms = {'upper_arm_angle': 0, 'shoulder_raised': False, 'arm_abducted': False, 'leaning': False,
                     'lower_arm_angle': 0,
                     'wrist_angle': 0, 'wrist_twisted': False}

        # Init lookup tables
        self.init_table_a()
        self.init_table_b()
        self.init_table_c()


    def init_table_a(self):
        '''
        Table used to compute upper body score

        :return: None
        '''
        self.table_a = np.array([
                                [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
                                [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
                                [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
                                ])

    def init_table_b(self):
        '''
        Table used to computer lower body score

        :return: None
        '''
        self.table_b = np.array([
                                [[1, 2, 2], [1, 2, 3]],
                                [[1, 2, 3], [2, 3, 4]],
                                [[3, 4, 5], [4, 5, 5]],
                                [[4, 5, 5], [5, 6, 7]],
                                [[6, 7, 8], [7, 8, 8]],
                                [[7, 8, 8], [8, 9, 9]],
                                ])

    def init_table_c(self):
        '''
        Table to compute score_c

        :return: None
        '''
        self.table_c = np.array([
                                [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
                                [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
                                [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
                                [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
                                [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
                                [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
                                [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
                                [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
                                [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
                                [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
                                [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
                                [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                                ])

    def set_body(self, values):
        # type: (np.ndarray) -> None
        '''
        Set body params

        :param values: [neck_angle, neck_side, trunk_angle, trunk_side,
                        legs_walking, legs_angle, load]

        :return: None
        '''
        assert len(values) == len(self.body)

        for i, (key, _) in enumerate(self.body.items()):
            self.body[key] = values[i]

    def set_arms(self, values):
        # type: (np.ndarray) -> None
        '''
        Set arms params

        :param values:  [upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                        lower_arm_angle, wrist_angle, wrist_twisted]

        :return: None
        '''
        assert len(values) == len(self.arms)

        for i, (key, _) in enumerate(self.arms.items()):
            self.arms[key] = values[i]

    def compute_score_a(self):
        # type: (RebaScore) -> (np.ndarray, np.ndarray)
        '''
        Compute score A
        >>> rebascore = RebaScore()
        >>> rebascore.set_body(np.array([10, 0, 20, 0, 1, 50, 0]))
        >>> rebascore.compute_score_a()
        (4, array([1, 2, 3]))

        :return: Score A, [neck_score, trunk_score, leg_score]
        '''
        neck_score, trunk_score, leg_score, load_score = 0, 0, 0, 0

        # Neck position
        if 10 <= self.body['neck_angle'] <= 20 :
            neck_score +=1
        else:
            neck_score +=2
        # Neck adjust
        neck_score +=1 if self.body['neck_side'] else 0

        # Trunk position
        if 0 <= self.body['trunk_angle'] <= 1:
            trunk_score +=1
        elif self.body['trunk_angle'] <= 20:
            trunk_score +=2
        elif 20 <= self.body['trunk_angle'] <= 60:
            trunk_score +=3
        elif self.body['trunk_angle'] > 60:
            trunk_score +=4
        # Trunk adjust
        trunk_score += 1 if self.body['trunk_side'] else 0

        # Legs position
        leg_score += 2 if self.body['legs_walking'] else 1
        # Legs adjust
        if 30 <= self.body['legs_angle'] <= 60:
            leg_score += 1
        elif self.body['legs_angle'] > 60:
            leg_score += 2

        # Load
        if 5 <= self.body['load'] <= 10:
            load_score += 1
        elif self.body['load'] > 10:
            load_score += 2

        assert neck_score > 0 and trunk_score > 0 and leg_score > 0

        score_a = self.table_a[neck_score-1][trunk_score-1][leg_score-1]
        return score_a, np.array([neck_score, trunk_score, leg_score])

    def compute_score_b(self):
        # type: (RebaScore) -> (np.ndarray, np.ndarray)
        '''
        Compute score B
        >>> rebascore = RebaScore()
        >>> rebascore.set_arms(np.array([45, 0, 0, 0, 70, 0, 1]))
        >>> rebascore.compute_score_b()
        (2, array([2, 1, 2]))

        :return: scoreB, [upper_arm_score, lower_arm_score, wrist_score]
        '''
        upper_arm_score, lower_arm_score, wrist_score = 0, 0, 0

        # Upper arm position
        if -20 <= self.arms['upper_arm_angle'] <= 20:
            upper_arm_score +=1
        elif self.arms['upper_arm_angle'] <= 45:
            upper_arm_score +=2
        elif 45 <= self.arms['upper_arm_angle'] <= 90:
            upper_arm_score +=3
        elif self.arms['upper_arm_angle'] > 90:
            upper_arm_score +=4

        # Upper arm adjust
        upper_arm_score += 1 if self.arms['shoulder_raised'] else 0
        upper_arm_score += 1 if self.arms['arm_abducted'] else 0
        upper_arm_score -= 1 if self.arms['leaning'] else 0

        # Lower arm position
        if 60 <= self.arms['lower_arm_angle'] <= 100:
            lower_arm_score += 1
        else:
            lower_arm_score += 2

        # Wrist position
        if -15 <= self.arms['wrist_angle'] <= 15:
            wrist_score += 1
        else:
            wrist_score += 2

        # Wrist adjust
        wrist_score += 1 if self.arms['wrist_twisted'] else 0

        assert lower_arm_score > 0 and wrist_score > 0

        score_b = self.table_b[upper_arm_score-1][lower_arm_score-1][wrist_score-1]
        return score_b, np.array([upper_arm_score, lower_arm_score, wrist_score])

    def compute_score_c(self, score_a, score_b):
        # type: (np.ndarray, np.ndarray) -> (np.ndarray, str)
        '''
        Compute score C

        :param score_a:  Score A
        :param score_b:  Score B

        :return: Score C, caption
        '''
        reba_scoring = ['Negligible Risk',
                         'Low Risk. Change may be needed',
                         'Medium Risk. Further Investigate. Change Soon',
                         'High Risk. Investigate and Implement Change',
                         'Very High Risk. Implement Change'
                         ]

        score_c = self.table_c[score_a-1][score_b-1]
        ix = self.score_c_to_5_classes(score_c)
        caption = reba_scoring[ix]

        return score_c, caption

    @staticmethod
    def score_c_to_5_classes(score_c):
        # type: (np.ndarray) -> int
        '''
        Score C to 5 risk-classes

        :param score_c:  Score C
        :return: Risk-class
        '''
        if score_c == 1:
            ret = 0
        elif 2 <= score_c <= 3:
            ret = 1
        elif 4 <= score_c <= 7:
            ret = 2
        elif 8 <= score_c <= 10:
            ret = 3
        else:
            ret = 4

        return ret

    @staticmethod
    def get_body_angles_from_pose_left(pose, verbose=False):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get body angles from pose (look at left)

        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging

        :return: Body params (neck_angle, neck_side, trunk_angle, trunk_side,
                legs_walking, legs_angle, load)
        '''
        pose = np.expand_dims(np.copy(pose), 0)

        neck_angle, neck_side, trunk_angle, trunk_side, \
        legs_walking, legs_angle, load = 0, 0, 0, 0, 0, 0, 0

        if verbose:
            utils.show_skeleton(pose, title="GT pose")
            _pose, _ = utils.rotate_pose(np.copy(pose), rotation_joint=8)
            utils.show_skeleton(_pose, title="GT pose left")

        # Neck position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8)
        pose -= pose[:, 1]
        neck_angle = np.rad2deg(np.arctan2(pose[0, 0, 1], pose[0, 0, 0]) - (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Neck angle: "+ str(round(neck_angle, 2)))

        # Neck bending
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi/2)
        neck_side_angle = abs(np.rad2deg(np.arctan2(pose[0, 0, 1], pose[0, 0, 0]) - (np.pi / 2)))
        neck_side = 1 if neck_side_angle > 20 else 0

        if verbose:
            utils.show_skeleton(pose, title="Neck side angle: " + str(round(neck_side_angle, 2)))

        # Trunk position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi/2)
        pose -= (pose[:, 8] + pose[:, 11]) /2
        trunk_angle = np.rad2deg(np.arctan2(pose[0, 1, 1], pose[0, 1, 0]) - (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Trunk angle: " + str(round(trunk_angle, 2)))

        # Trunk bending
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi/2)
        trunk_side_angle = abs(np.rad2deg(np.arctan2(pose[0, 1, 1], pose[0, 1, 0]) - (np.pi / 2)))
        trunk_side = 1 if trunk_side_angle > 30 else 0

        if verbose:
            utils.show_skeleton(pose, title="Trunk side angle: " + str(round(trunk_side_angle, 2)))

        # Legs position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
        pose -= pose[:, 8]
        legs_angle = -np.rad2deg(np.arctan2(pose[0, 9, 1], pose[0, 9, 0]) + (np.pi/2))
        step_size = abs(np.linalg.norm(pose[0, 10, :2] - pose[0, 13, :2]))
        legs_walking = 1 if step_size > 0.1 else 0

        if verbose:
            title = "Leg angle: " + str(round(legs_angle, 2)) + " Step size: " + str(round(step_size, 2))
            utils.show_skeleton(pose, title=title)

        return np.array([neck_angle, neck_side, trunk_angle, trunk_side,
                           legs_walking, legs_angle, load])


    @staticmethod
    def get_body_angles_from_pose_right(pose, verbose=False):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get body angles from pose (look at right)

        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging

        :return: Body params (neck_angle, neck_side, trunk_angle, trunk_side,
                              legs_walking, legs_angle, load)
        '''
        pose = np.expand_dims(np.copy(pose), 0)

        neck_angle, neck_side, trunk_angle, trunk_side, \
        legs_walking, legs_angle, load = 0, 0, 0, 0, 0, 0, 0

        if verbose:
            utils.show_skeleton(pose, title="GT pose")
            _pose, _ = utils.rotate_pose(np.copy(pose), rotation_joint=8)
            _pose, _ = utils.rotate_pose(_pose, rotation_joint=8, m_coeff=np.pi)
            utils.show_skeleton(_pose, title="GT pose Right")

        # Neck position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8)
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi)
        pose -= pose[:, 1]
        neck_angle = np.rad2deg((np.pi / 2) - np.arctan2(pose[0, 0, 1], pose[0, 0, 0]))

        if verbose:
            utils.show_skeleton(pose, title="Neck angle: " + str(round(neck_angle, 2)))

        # Neck bending
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi / 2)
        neck_side_angle = abs(np.rad2deg(np.arctan2(pose[0, 0, 1], pose[0, 0, 0]) - (np.pi / 2)))
        neck_side = 1 if neck_side_angle > 20 else 0

        if verbose:
            utils.show_skeleton(pose, title="Neck side angle: " + str(round(neck_side_angle, 2)))

        # Trunk position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
        pose -= (pose[:, 8] + pose[:, 11]) / 2
        trunk_angle = np.rad2deg((np.pi / 2) - np.arctan2(pose[0, 1, 1], pose[0, 1, 0]))

        if verbose:
            utils.show_skeleton(pose, title="Trunk angle: " + str(round(trunk_angle, 2)))

        # Trunk bending
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi / 2)
        trunk_side_angle = abs(np.rad2deg(np.arctan2(pose[0, 1, 1], pose[0, 1, 0]) - (np.pi / 2)))
        trunk_side = 1 if trunk_side_angle > 30 else 0

        if verbose:
            utils.show_skeleton(pose, title="Trunk side angle: " + str(round(trunk_side_angle, 2)))

        # Legs position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
        pose -= pose[:, 11]
        legs_angle = np.rad2deg((np.pi / 2) + np.arctan2(pose[0, 12, 1], pose[0, 12, 0]))
        step_size = abs(np.linalg.norm(pose[0, 10, :2] - pose[0, 13, :2]))
        legs_walking = 1 if step_size > 0.1 else 0

        if verbose:
            title = "Leg angle: " + str(round(legs_angle, 2)) + " Step size: " + str(round(step_size, 2))
            utils.show_skeleton(pose, title=title)

        return np.array([neck_angle, neck_side, trunk_angle, trunk_side,
                legs_walking, legs_angle, load])

    @staticmethod
    def get_arms_angles_from_pose_left(pose, verbose=False):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get arms angles from pose (look at left)

        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging

        :return: Body params (upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                              lower_arm_angle, wrist_angle, wrist_twisted)
        '''
        pose = np.expand_dims(np.copy(pose), 0)
        if verbose:
            utils.show_skeleton(pose, title="GT pose")

        # Upper Arm position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8)
        pose -= pose[:, 2]
        upper_arm_angle = -np.rad2deg(np.arctan2(pose[0, 3, 1], pose[0, 3, 0]) + (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Upper Arms angle: " + str(round(upper_arm_angle, 2)))

        # Upper Arm Adjust
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi/2)
        shoulder_step = pose[:, 2, 1] - pose[:, 1, 1]
        arm_abducted_angle = np.rad2deg(np.arctan2(pose[0, 3, 1], pose[0, 3, 0]) + (np.pi / 2))
        shoulder_raised = 1 if shoulder_step > 0.02 else 0
        arm_abducted = 1 if abs(arm_abducted_angle) > 45 else 0

        if verbose:
            print(shoulder_raised)
            utils.show_skeleton(pose, title="Upper Arms abducted: " + str(round(arm_abducted_angle, 2)))

        # Leaning
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
        pose -= (pose[:, 8] + pose[:, 11]) / 2
        trunk_angle = np.rad2deg(np.arctan2(pose[0, 1, 1], pose[0, 1, 0]) - (np.pi / 2))
        leaning = 1 if trunk_angle > 30 else 0

        if verbose:
            utils.show_skeleton(pose, title="Leaning angle: " + str(round(trunk_angle, 2)))

        # Lower Arm position
        pose -= pose[:, 3]
        lower_arm_angle = -np.rad2deg(np.arctan2(pose[0, 4, 1], pose[0, 4, 0]) + (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Lower Arms angle: " + str(round(lower_arm_angle, 2)))

        # Wrist position
        wrist_angle = 0
        wrist_twisted = 0

        if pose.shape[1] > 14:
            pose -= pose[:, 4]
            wrist_angle = np.rad2deg(np.arctan2(pose[0, 14, 1], pose[0, 14, 0]) - (np.pi / 2) )

            if verbose:
                utils.show_skeleton(pose, title="Wrist Angle: " + str(round(wrist_angle, 2)))

            pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
            wrist_twisted_angle = abs(np.rad2deg(np.arctan2(pose[0, 14, 1], pose[0, 14, 0]) - (np.pi / 2)))
            wrist_twisted = 1 if wrist_twisted_angle > 30 else 0

        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                lower_arm_angle, wrist_angle, wrist_twisted])


    @staticmethod
    def get_arms_angles_from_pose_right(pose, verbose=False):
        # type: (np.ndarray, bool) -> np.ndarray
        '''
        Get arms angles from pose (look at right)

        :param pose: Pose (Joints coordinates)
        :param verbose: If true show each pose for debugging

        :return: Body params (upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                               lower_arm_angle, wrist_angle, wrist_twisted)
        '''
        pose = np.expand_dims(np.copy(pose), 0)
        if verbose:
            utils.show_skeleton(pose, title="GT pose")

        # Upper Arm position
        pose, _ = utils.rotate_pose(pose, rotation_joint=8)
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi)
        pose -= pose[:, 5]
        upper_arm_angle = np.rad2deg(np.arctan2(pose[0, 6, 1], pose[0, 6, 0]) + (np.pi / 2))

        if verbose:
            utils.show_skeleton(pose, title="Upper Arms angle: " + str(round(upper_arm_angle, 2)))

        # Upper Arm Adjust
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=np.pi / 2)
        shoulder_step = pose[:, 5, 1] - pose[:, 1, 1]
        arm_abducted_angle = np.rad2deg((np.pi / 2) + np.arctan2(pose[0, 6, 1], pose[0, 6, 0]))
        shoulder_raised = 1 if shoulder_step > 0.02 else 0
        arm_abducted = 1 if abs(arm_abducted_angle) > 45 else 0

        if verbose:
            print(shoulder_raised)
            utils.show_skeleton(pose, title="Upper Arms abducted: " + str(round(arm_abducted_angle, 2)))

        # Leaning
        pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
        pose -= (pose[:, 8] + pose[:, 11]) / 2
        trunk_angle = np.rad2deg((np.pi / 2) - np.arctan2(pose[0, 1, 1], pose[0, 1, 0]))
        leaning = 1 if trunk_angle > 60 else 0

        if verbose:
            utils.show_skeleton(pose, title="Leaning angle: " + str(round(trunk_angle, 2)))

        # Lower Arm position
        pose -= pose[:, 6]
        lower_arm_angle = np.rad2deg((np.pi / 2) + np.arctan2(pose[0, 7, 1], pose[0, 7, 0]) )

        if verbose:
            utils.show_skeleton(pose, title="Lower Arms angle: " + str(round(lower_arm_angle, 2)))

        # Wrist position
        wrist_angle = 0
        wrist_twisted = 0

        if pose.shape[1] > 14:
            pose -= pose[:, 7]
            wrist_angle = np.rad2deg((np.pi / 2) + np.arctan2(pose[0, 15, 1], pose[0, 15, 0]))

            if verbose:
                utils.show_skeleton(pose, title="Wrist Angle: " + str(round(wrist_angle, 2)))

            pose, _ = utils.rotate_pose(pose, rotation_joint=8, m_coeff=-np.pi / 2)
            wrist_twisted_angle = abs(np.rad2deg((np.pi / 2) + np.arctan2(pose[0, 15, 1], pose[0, 15, 0])))
            wrist_twisted = 1 if wrist_twisted_angle > 30 else 0


        return np.array([upper_arm_angle, shoulder_raised, arm_abducted, leaning,
                         lower_arm_angle, wrist_angle, wrist_twisted])


if __name__ == '__main__':
    rebaScore = RebaScore()

    rebaScore.set_body(np.array([10, 0, 20, 0, 1, 50, 0]))
    score_a, partial_a = rebaScore.compute_score_a()

    rebaScore.set_arms(np.array([45, 0, 0, 0, 70, 0, 1]))
    score_b, partial_b = rebaScore.compute_score_b()

    score_c, caption = rebaScore.compute_score_c(score_a, score_b)

    print(score_a, partial_a)
    print(score_b, partial_b)
    print(score_c, caption)


