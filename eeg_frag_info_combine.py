# class_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "UNKNOWN"}
# epoch_len in s

import itertools

def eeg_frag_info(y_pred, epoch_len):
    weight = {0:0, 1:3, 2:4, 3:6, 4:3, 5:-1}
    label = {0:0, 1:3, 2:7, 3:12, 4:3, 5:-1}

    sleep = 0.0   # total # of stages defined as "asleep"
    deep = 0.0    # total # of deep sleep stage N3
    rem = 0.0     # total # of REM stages
    tran = []     # Transition label vectors
    twt = 0       # Total weighted transition

    # "previous" wt and label. Initialized to wt/lab of first element so first
    # transition guarenteed to be 0 and will not affect total score
    w_prev = weight[y_pred[0]]
    l_prev = label[y_pred[0]]

    # list traverse
    for i in y_pred:
        w_curr = weight[i]                  # wt of current stage
        l_curr = label[i]                   # modified label of stages

        if w_curr != -1:                    # if wt = -1: unknown, do not count
            wt = w_prev - w_curr            # calculate current transition wt
            if wt > 0 :                    
                twt += wt                   # add to total if transition wt > 0
        w_prev = w_curr                     # prev <- curr

        if l_curr != -1:                    # if not unknown
            tran.append(l_prev - l_curr)    # transition label = prev - curr
        else:
            tran.append(0)                  # if unknown, treat as no transition
        l_prev = l_curr                     # prev <- curr

        if i != 0 and i != 5:               # not wake or unknown
            sleep +=1                       # total sleep stage + 1
            if i == 3:                      # deep sleep
                deep += 1                   # total deep stage + 1
            elif i == 4:                    # REM
                rem += 1                    # total REM stage + 1

    tst = round (sleep * epoch_len / 3600, 2)   # Total sleep time in hrs   
    
    # score and message based on deep sleep
    deep_ratio = round(deep*100/sleep, 2)
    deep_score = get_score(deep_ratio, 13, 100, 50)

    # score and message based on REM
    rem_ratio = round(rem*100/sleep)
    rem_score = get_score(rem_ratio, 20, 25, 50)
    
    # weighted transition rate (sleep fragmentation index)
    wtr = round (twt / tst, 2)  
    if wtr < 10:
        wtr_msg = ("The Sleep Fragmentation Index represents sleep fragmentation " +
                  "that could negatively affect the overall sleep quality. " +
                  "You have a low score so you've likely had a peaceful sleep. ")
    elif wtr < 20:
        wtr_msg = ("The Sleep Fragmentation Index represents sleep fragmentation " +
                  "that could negatively affect the overall sleep quality. " +
                  "You have a moderately high score so you've likely experienced " +
                  "some disturbance during your sleep.")
    else:
        wtr_msg = ("The Sleep Fragmentation Index represents sleep fragmentation " +
                  "that could negatively affect the overall sleep quality. " +
                  "You have a very high score so you've likely experienced " +
                  "quite a lot of disturbance during your sleep. If you felt " +
                  "tired despite a long night of sleep, this may be why!")

    
    # Combine msgs & scores
    score = deep_score + rem_score
    overall_msg = make_msg(deep_ratio, rem_ratio, tst)
    epoch_msg = get_epoch_msg(tran, y_pred)

    # overall score, overall msg, total sleep time,
    # weighted transition rate, comment for overall transition rate,
    # description for each epoch (stage + transition)
    return score, overall_msg, tst, wtr, wtr_msg, epoch_msg
    
def get_epoch_msg(tran, y_pred):
    # library for stage description
    wake    = "You were awake. \n"
    rem     = ("You were experiencing REM (rapid eye movement) sleep.\n" +
               "Your eyes were moving rapidly (hence the name) and your " +
               "brain was being really active. You were most likely to dream " +
               "during this stage.\n")
    n1      = ("You were experiencing shallow sleep.\n" +
               "This a transitional stage where your sleep was very shallow " +
               "and could be very easily interrupted. You might even have a sense of " +
               "awareness and didn't feel asleep as all.")
    n2      = ("You were experiencing light sleep.\n"+
               "This stage constitutes the majority of your sleep. During this " +
               "stage your body started to relax and your brain was processing " +
               "your memories from the day. You could still be easily awoken by " +
               "disruptive stimuli.\n")
    n3      = ("You were experiencing deep sleep.\n" +
               "This is where you get most of the rest you need. Your muscles and brain " +
               "were fully relaxed and you would only wake up due to intensive stimuli." +
               "You were also very unlikely to dream during this stage. \n")
    unknow  = "Unkown: Sorry, we were unable to detect your current sleep stage\n"
    
    stage_lib = {0: wake, 1: n1, 2:n2, 3:n3, 4:rem}

    # library for transition description
    msg0 = "No stage transition\n"
    msg1 = "Waking up from REM sleep\n"
    msg2 = "Waking up from shallow sleep\n"
    msg3 = "Waking up from light sleep\n"
    msg4 = "Waking up from deep sleep\n"
    msg5 = "Transitioning from shallow sleep to REM sleep\n"
    msg6 = "Transitioning from light sleep to REM sleep\n"
    msg7 = "Transitioning from deep sleep to REM sleep"
    msg8 = "Transitioning from light sleep to shallow sleep\n"
    msg9 = "Transitioning from deep sleep to shallow sleep\n"
    msg10 = "Transitioning from deep sleep to light sleep\n"
    msg11 = "Falling into deeper sleep\n"
    
    trans_lib = {0: msg0, 1: msg2, 3: msg3, 7: msg3,
                 12: msg4, 2: msg5, 6: msg6, 11: msg7,
                 4: msg8, 9: msg9, 5: msg10, -1: msg11}
        
    # result holder
    epoch_msg = []

    for i, j in itertools.izip(tran, y_pred):
        if i < 0:
            epoch_msg.append(trans_lib[-1] + stage_lib[j])
        else:
            epoch_msg.append(trans_lib[i] + stage_lib[j])
    return epoch_msg

def get_score(ratio, lower, upper, weight):
    if ratio < lower:
        score = ratio / lower * weight
    elif ratio > upper:
        score = (1 - (ratio - upper) / upper) * weight
    else:
        score = weight
    return score

def make_msg(deep_ratio, rem_ratio, tst):
    deep_lib = {0:("Looks like you've had a nice deep sleep, " +
                   "so you should feel quite refreshed, "),
                2:("Looks like you didn't get quite enough deep sleep, ")}
    
    rem_lib = {0: ("you had the right amount of REM sleep so you must " +
                   "be feeling calm and energetic for the day!\n"),
               2: ("you had too much REM sleep which might make you become " +
                   "angry or irritable more easily than usual. If so, take " +
                   "a deep breathe and enjoy a nice cup of coffee with some " +
                   "of your favourite music!\n"),
               3: ("you had too little REM sleep which could make you lazy and less " +
                   "able to focus... Probably not the perfect day for those hard tasks! \n")}
    
    tst_lib = {0: (", which is appropriate for most healthy adults. Keep it on! \n"),
               1: (". Seriously you need to get more sleep, it's important!!\n"),
               2: (". That's too much sleep... Get up and enjoy the sunshine!\n")}

    if deep_ratio < 8:
        deep = 2
    else:
        deep = 0
    
    if rem_ratio > 30:
        rem = 2
    elif rem_ratio < 15:
        rem = 3
    else:
        rem = 0

    if tst > 11:
        tstm = 2
    elif tst < 6:
        tstm = 1 
    else:
        tstm = 0
    tst_msg = ("You slept for a total of %.2f hours" % tst
               + tst_lib[tstm])

    if abs(rem - deep) > 1:
        msg = (tst_msg + deep_lib[deep] + "but " + rem_lib[rem] )
    else:
        msg = (deep_lib[deep] + "and " + rem_lib[rem] )
    return msg

