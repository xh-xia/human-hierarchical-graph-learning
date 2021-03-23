

import numpy as np

FINGER_COMBINATIONS_5 = np.array([[ True, False, False, False, False],
                                  [False,  True, False, False, False],
                                  [False, False,  True, False, False],
                                  [False, False, False,  True, False],
                                  [False, False, False, False,  True],
                                  [ True,  True, False, False, False],
                                  [ True, False,  True, False, False],
                                  [ True, False, False,  True, False],
                                  [ True, False, False, False,  True],
                                  [False,  True,  True, False, False],
                                  [False,  True, False,  True, False],
                                  [False,  True, False, False,  True],
                                  [False, False,  True,  True, False],
                                  [False, False,  True, False,  True],
                                  [False, False, False,  True,  True]]) # 15 combos
FINGER_COMBO_5_VACANT = np.full(FINGER_COMBINATIONS_5.shape,False,dtype=bool)
FINGER_COMBO_5x5 = np.concatenate((np.concatenate((FINGER_COMBINATIONS_5,FINGER_COMBO_5_VACANT), axis=1)\
                                  ,np.concatenate((FINGER_COMBO_5_VACANT,FINGER_COMBINATIONS_5), axis=1)), axis=0)
def trash():
    print(FINGER_COMBO_5x5)




if __name__=="__main__":
    trash()
