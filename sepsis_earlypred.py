
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import sepsis_supervised_hmm_DP2 as shd
import sepsis_perform_eval as spe
import random
from sklearn.metrics import confusion_matrix


# ## Early Prediction

# In[377]:

# ### Functions

def loaddf(filepath, filename):
    df = pd.read_csv(filepath+filename, header=0)
    return df

def savedf(df, filepath, filename):
    print('* save file: '+filename)
    time = datetime.now().strftime('%m_%d_%H')
    df.to_csv(filepath+filename+'_'+time+'.csv', index = False)
    
def sortbytime(df):
    pd.to_numeric(df.MinutesFromArrival, errors='coerce')
    df = df.sort_values(by=['VisitIdentifier', 'MinutesFromArrival'])
    df = df.reset_index(drop=True)
    return df

# foward + normal value filling
def removeNull(df, traindf, savefile_mode, filename):
    print('\n* carry forward features')
    # 1. Carryforward for vital signs until the new measurement happended for all the features 
    feat = catfeat+confeat
    df[feat] = df.groupby('VisitIdentifier')[feat].ffill()

    # 2. Set normal value
    print('* set normal values for nulls: mean for continuous feature/ 0 for binary feature')
    for f in confeat:  # continuous 
        df[f] = df[f].fillna(traindf[f].mean())
    for f in catfeat:
        df[f] = df[f].fillna(0)

    # 3. save the data files
    if savefile_mode:
        savedf(df, filepath, filename)

    return df


# ### Statistic Info
def statLabels(df, label, title):
    # Event Level
    totlen = np.size(df,0)
    #nnlen = np.size(df[pd.isnull(df[label])],0) # not null label length
    poslen = np.size(df[df[label]==1],0) # positive label length
    neglen = np.size(df[df[label]==0],0) 

    # Visit Level
    posvid = df[df['ShockFlag']==1]['VisitIdentifier'].unique().tolist()
    posvidlen = len(posvid)
    
    negvid = df[df['ShockFlag']==0]['VisitIdentifier'].unique().tolist()
    negvid = [x for x in negvid if x not in posvid]
    negvidlen = len(negvid)
    
    totvid = df['VisitIdentifier'].unique().tolist()
    totvidlen = len(totvid)

    statcols = ['Pos', 'Neg','total']
    statrows = ['Visit', 'Event']
    data = [[posvidlen, negvidlen, totvidlen],
            [poslen, neglen, totlen]]
    print(title)
    print(pd.DataFrame(data,statrows,statcols))
    
    return posvid, negvid, totvid


# divide training/testing dataset
def getDataset(df, savemode):

    dfpos = df[df.ShockFlag == 1]
    posvids = dfpos.VisitIdentifier.sample(len(dfpos)).unique().tolist()
    negvids = [x for x in df.VisitIdentifier.sample(len(df)).unique().tolist() if x not in posvids]

    trainposvid = posvids[0:500]  # pos train: 500
    trainnegvid = negvids[0:2830] # neg train: 2830
    
    testposvid = [x for x in posvids if x not in trainposvid][0:100] # pos test: 100
    testnegvid = [x for x in negvids if x not in trainnegvid][0:570] # neg test: 570

    trainvid = trainposvid + trainnegvid
    testvid = testposvid + testnegvid
    traindf = df[df['VisitIdentifier'].isin(trainvid)]
    testdf = df[df['VisitIdentifier'].isin(testvid)]

    if savemode:
        savedf(traindf, filepath, 'sp_traindf')
        savedf(testdf, filepath, 'sp_testdf')
    return traindf, testdf, trainvid, testvid, trainposvid, trainnegvid, testposvid, testnegvid


# ### Standardization
def standardize(traindf, testdf, savefile_mode):
    print('* standardize')

    trainmean = traindf[numfeat].mean()
    trainstd = traindf[numfeat].std()
    traindf.loc[:, numfeat] = (traindf.loc[:, numfeat] - trainmean[numfeat]) / trainstd[
        numfeat]  # standardize the whole numeric data with train mean & std
    testdf.loc[:, numfeat] = (testdf.loc[:, numfeat] - trainmean[numfeat]) / trainstd[numfeat]

    # df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    # save the data files
    if savefile_mode:
        savedf(traindf, filepath, 'traindf_std')
        savedf(testdf, filepath, 'testdf_std')

    return traindf, testdf

def divideDataRate(label_rate, unlabel_n, trainposvid, trainnegvid):
    trainparts = []
    random.shuffle(trainposvid)
    random.shuffle(trainnegvid)
    label_posvidnum = int(len(trainposvid)*label_rate)
    label_negvidnum = int(len(trainnegvid)*label_rate)
    trainparts.append(trainposvid[0:label_posvidnum]
                           +trainnegvid[0:label_negvidnum])

    posvidpartnum = int((len(trainposvid) - label_posvidnum)/unlabel_n)
    negvidpartnum = int((len(trainnegvid) - label_negvidnum)/unlabel_n)
    
    for i in range(unlabel_n):
        posidx = label_posvidnum + i*posvidpartnum
        negidx = label_negvidnum + i*negvidpartnum
        trainparts.append(trainposvid[posidx:posidx+posvidpartnum]
                           +trainnegvid[negidx:negidx+negvidpartnum])
        
    return trainparts


def divideData(n, trainposvid, trainnegvid):
    trainparts = []
    posvidpartnum = int(len(trainposvid)/n)
    negvidpartnum = int(len(trainnegvid)/n)
    for i in range(n):
        posidx = i*posvidpartnum
        negidx = i*negvidpartnum
        trainparts.append(trainposvid[posidx:posidx+posvidpartnum]
                           +trainnegvid[negidx:negidx+negvidpartnum])
        
    return trainparts

# calculate positive number at Visit level in prediction & original labels
def calPosVisit(seq_lengths, pred, orglabs):
    idx = 0
    predVisit = []
    orgVisit = []
    for i in range(len(seq_lengths)):
        if i > 0:
            idx += seq_lengths[i - 1]
        if 1 in pred[idx:idx + seq_lengths[i]]:
            predVisit.append(1)
        else:
            predVisit.append(0)

        if 1 in orglabs[idx:idx + seq_lengths[i]]:
            orgVisit.append(1)
        else:
            orgVisit.append(0)

#     print("VisitLevel: original posnum = " + str(np.sum(np.equal(orgVisit, 1))) 
#           +", pred posnum = " + str(np.sum(np.equal(predVisit, 1))))
    return predVisit, orgVisit

def conf_measures(resultdf, orgVisit, predVisit):
    conf = confusion_matrix(orgVisit, predVisit)
    accuracy = (conf[0][0] + conf[1][1]) / conf.sum()
    precision = conf[1][1]/(conf[1][1]+conf[1][0])
    recall = conf[1][1]/(conf[1][1]+conf[0][1])
    f1 = 2*precision*recall/(precision+recall)
    #print("accuracy: "+str(accuracy)+", precision: "+str(precision)+", recall: "+str(recall)+", f1: "+str(f1)+"\n")
    return [conf, accuracy, precision, recall, f1]

## Single Feature Exploration
def featEx(confeat, catfeat):
    featscoredf = pd.DataFrame(columns = ['features','conf','accuracy', 'precision', 'recall', 'f1'])
    
    for f in confeat:
        train_f1 = hmmtest(traindf, testdf, fixfeat, [f], 'Gaussian', featscoredf)

    for f in catfeat:
        train_f1 = hmmtest(traindf, testdf, fixfeat, [f], 'Multinomial' ,featscoredf)

    # combinations
    return featsocredf

def getMinutesDist(testdf):
    predTime = []
    labelTime = []
    testvid = testdf.VisitIdentifier.unique().tolist()
    for vid in sorted(testvid):
        ptimelist = testdf.loc[testdf.VisitIdentifier==vid].loc[testdf.pred == 1].MinutesFromArrival.tolist()
        ltimelist = testdf.loc[testdf.VisitIdentifier==vid].loc[testdf.ShockFlag == 1].MinutesFromArrival.tolist()

        if len(ptimelist) >0:
            predTime.append(ptimelist[0])
        else:
            predTime.append(np.nan)
        if len(ltimelist) > 0:
            labelTime.append(ltimelist[0])
        else:
            labelTime.append(np.nan)

    minutesDist = np.zeros(len(predTime))
    for i in range(len(predTime)):
        minutesDist[i] = predTime[i] - labelTime[i]

    TP_minutesDist = minutesDist[np.isnan(minutesDist)== False]
    TP_hourDist = TP_minutesDist/60
    
    return TP_minutesDist, TP_hourDist # true positive minutes distances

def labels_byvid(testpred, test_seq_lengths):
    testpred_list = []
    idx = 0
    for i in range (len(test_seq_lengths)):
        testpred_list.append(testpred[idx:idx+test_seq_lengths[i]])
        idx = i*test_seq_lengths[i]

    return testpred_list

def getEarlyPredRate(TP_hourDist):
    TP_earlypred = []
    for i in range(24): # 24 hours
        TP_earlypred.append(len(TP_hourDist[TP_hourDist<=(-1)*i]))

    TP_earlypredrate = np.array(TP_earlypred)/len(TP_hourDist)
    return TP_earlypredrate

def learn_predict(df, model, lobservs, lseq_lengths, ltrain_labs, test_observs, test_seq_lengths, test_labs):
    #model.fit(lobservs, lseq_lengths)
    ltrain_pred = model.predict(lobservs, lseq_lengths)
    ltrainPredVisit, ltrainVisit = calPosVisit(lseq_lengths, ltrain_pred, ltrain_labs)
    tr_info = conf_measures(df, ltrainVisit, ltrainPredVisit)
    #print("Train - acc: "+str(tr_info[1])+", prec: "+str(tr_info[2])+", recall: "+str(tr_info[3])+", f1: "+str(tr_info[4]))
    
    test_pred = model.predict(test_observs, test_seq_lengths)
    testPredVisit, testVisit = calPosVisit(test_seq_lengths, test_pred, test_labs)
    te_info = conf_measures(df, testVisit, testPredVisit)
    #print("Test - acc: "+str(te_info[1])+", prec: "+str(te_info[2])+", recall: "+str(te_info[3])+", f1: "+str(te_info[4]))
    return model, tr_info, te_info, test_pred


# Supervised learning test   
def hmmtest_super(trainparts, traindf, testdf, fixfeat, feat, hmm_mode):
    ltrainvid = []
    utrainvid = []
    train_whole = traindf[fixfeat + feat] 
    test_data = testdf[fixfeat + feat]
    
    df = pd.DataFrame(columns = ['features','labelnum','unlabelnum','init_prob','trans_prob','emission_prob',
                               'tr_confmat','tr_acc', 'tr_prec', 'tr_recall', 'tr_f1', 
                                'te_confmat','te_acc', 'te_prec', 'te_recall', 'te_f1', 'TP_hourDist', 'earlyRate']) 
    
    for i in range(len(trainparts)):
        ltrainvid += trainparts[i]
        ltrain_data = train_whole[train_whole.VisitIdentifier.isin(ltrainvid)]

        [model, lobservs, lseq_lengths, ltrain_labs, observ_dict, uniq_observs] = shd.hmm_setup(ltrain_data, "ShockFlag", "VisitIdentifier", hmm_mode)
        [test_observs, test_seq_lengths, test_labs] = shd.hmm_pred_setup(test_data, "ShockFlag", "VisitIdentifier",hmm_mode, observ_dict)
           
        print("\n** i= "+str(i))
        model, train_info, test_info, test_pred = learn_predict(df, model, lobservs, lseq_lengths, ltrain_labs, test_observs, test_seq_lengths, test_labs)                         
        testdf['pred'] = test_pred
        
        # Early prediction Testing codes by hours
        TP_minutesDist, TP_hourDist = getMinutesDist(testdf)
        earlyRate = getEarlyPredRate(TP_hourDist)
        print("* early prediction rate: "+str(earlyRate))
        
        df.loc[len(df)] = [feat, len(ltrainvid), len(utrainvid), model.startprob_, model.transmat_, model.emissionprob_,
                                   train_info[0], train_info[1], train_info[2],train_info[3], train_info[4],
                                   test_info[0],test_info[1], test_info[2],test_info[3], test_info[4], 
                           TP_hourDist.mean(), earlyRate]

    return df

# Semisupervised learning test    
def hmmtest_semi(trainparts, traindf, testdf, fixfeat, feat, hmm_mode):
    ltrainvid = []
    train_whole = traindf[fixfeat + feat] 
    test_data = testdf[fixfeat + feat]
    #train_whole['ShockPred'] = traindf['ShockFlag']
    
    df = pd.DataFrame(columns = ['features','labelnum','unlabelnum','init_prob','trans_prob','emission_prob',
                               'tr_confmat','tr_acc', 'tr_prec', 'tr_recall', 'tr_f1', 
                                 'u_confmat','u_acc', 'u_prec', 'u_recall', 'u_f1', 
                                'te_confmat','te_acc', 'te_prec', 'te_recall', 'te_f1', 'TP_hourDist', 'earlyRate'])  
    
    
    for i in range(len(trainparts)):
        ltrainvid += trainparts[i]
        ltrain_data = train_whole[train_whole.VisitIdentifier.isin(ltrainvid)]
        if i < len(trainparts)-1:
            utrainvid = trainparts[i+1]
            utrain_data = train_whole[train_whole.VisitIdentifier.isin(utrainvid)]
        else:
            utrinvid = []

        if i == 0:
            [model, lobservs, lseq_lengths, ltrain_labs, observ_dict, uniq_observs] = shd.hmm_setup(ltrain_data, "ShockFlag", "VisitIdentifier", hmm_mode)
            [uobservs, useq_lengths, utrain_labs]  = shd.hmm_pred_setup(utrain_data, "ShockFlag", "VisitIdentifier", hmm_mode, observ_dict)
            [test_observs, test_seq_lengths, test_labs] = shd.hmm_pred_setup(test_data, "ShockFlag", "VisitIdentifier",hmm_mode, observ_dict)
              
        print("\n** i= "+str(i))
        prev_train_recall = 0
        for k in range(10): # loop until convergence
            print("k= "+str(k))
            #[model, lobservs, lseq_lengths, ltrain_labs, observ_dict, uniq_observs] = shd.hmm_setup(ltrain_data, "ShockPred", "VisitIdentifier", hmm_mode)
            model.fit(lobservs, lseq_lengths)
            model, train_info, test_info, test_pred = learn_predict(df,model,lobservs, lseq_lengths, ltrain_labs, test_observs, test_seq_lengths, test_labs)                         

            if i < len(trainparts)-1:
                print(" Unlabeled classification...")
                utrain_pred = model.predict(uobservs, useq_lengths)
                uPredVisit, utrainVisit = calPosVisit(useq_lengths, utrain_pred, utrain_labs)
                u_info = conf_measures(df, utrainVisit, uPredVisit)

                # merge assigned unlabelled data to labelled data for the next turn 
                lobservs =  np.array(lobservs.tolist() + uobservs.tolist()) 
                lseq_lengths = np.array(lseq_lengths.tolist() + useq_lengths.tolist())
                ltrain_labs = np.array(ltrain_labs.tolist() + utrain_pred.tolist())
                # update assigned labels for unlabeled data
                utrainIdx = traindf[traindf.VisitIdentifier.isin(utrainvid)].index
                #traindf.set_value(utrainIdx, 'ShockFlag', utrain_pred)
                #ltrain_data = train_whole[train_whole.VisitIdentifier.isin(ltrainvid+utrainvid)]
                
            train_recall = train_info[3]
            print("* train_recall:"+str(train_recall)+", test_recall:"+str(test_info[3]))
            
            if train_recall == prev_train_recall:
                break
            prev_train_recall = train_recall

        testdf['pred'] = test_pred

        # Early prediction Testing codes by hours
        TP_minutesDist, TP_hourDist = getMinutesDist(testdf)
        earlyRate = getEarlyPredRate(TP_hourDist)
        print("* early prediction rate: "+str(earlyRate))
            
        df.loc[len(df)] = [feat, len(ltrainvid), len(utrainvid), model.startprob_, model.transmat_, model.emissionprob_,
                                   train_info[0], train_info[1], train_info[2],train_info[3], train_info[4],
                                   u_info[0], u_info[1], u_info[2], u_info[3], u_info[4],
                                   test_info[0],test_info[1], test_info[2],test_info[3], test_info[4], TP_hourDist.mean(), earlyRate]
       
    return df

def hmmtest(traindf, testdf, ehour, fixfeat, feat, hmm_mode, df, shockmode):
    #print(feat)
    train_data = traindf[fixfeat + feat]
    test_data = testdf[fixfeat + feat]
    
    [model, observs, seq_lengths, train_labs, observ_dict, uniq_observs] = shd.hmm_setup(train_data, shockmode, "VisitIdentifier", hmm_mode)
    [test_observs, test_seq_lengths, test_labs] = shd.hmm_pred_setup(test_data, "ShockFlag", "VisitIdentifier", hmm_mode, observ_dict)
    
    model, train_info, test_info, test_pred = learn_predict(df,model,observs, seq_lengths, train_labs, test_observs, test_seq_lengths, test_labs)  
    testdf['pred'] = test_pred
    
#     print(" * init_prob:\n"+ str(model.startprob_))    
#     print(" * emission_prob:\n"+str(model.emissionprob_))
#     print(" * trans_pob:\n"+str(model.transmat_))
        
    #if df != False:
    trainvid = traindf.VisitIdentifier.unique().tolist()
    TP_minutesDist, TP_hourDist = getMinutesDist(testdf)
    er = getEarlyPredRate(TP_hourDist)
    print(er)
    df.loc[len(df)]=[feat, ehour, len(trainvid), 0, model.startprob_, model.transmat_,model.emissionprob_,
                 train_info[0], train_info[1], train_info[2],train_info[3], train_info[4],
                 test_info[0],test_info[1], test_info[2],test_info[3], test_info[4], TP_hourDist.mean(),
                 er[0], er[1], er[2], er[3], er[4], er[5], er[6], er[7], er[8], er[9], er[10], er[11], er[12],
                er[13], er[14], er[15], er[16], er[17], er[18], er[19], er[20], er[21], er[22], er[23],0]
    return test_pred, test_labs, test_seq_lengths, test_info

def getEarlyHourData(df, hour):   
    posvid = df[df.ShockFlag ==1].VisitIdentifier.unique().tolist()
    #edf['ShockPred'] = df['ShockFlag']
    edf = df.copy(deep=True)
    ermapcnt = 0
    #dropIdx = []

    for vid in sorted(posvid):
        vdf = edf.loc[edf.VisitIdentifier==vid]
        startIdx = vdf.index[0]
        endIdx = vdf.index[-1]
        firstShockIdx = vdf.loc[vdf.ShockFlag == 1].index[0]
        firstShockMinutes = vdf.loc[firstShockIdx].MinutesFromArrival
        ermaps = vdf.loc[vdf.MinutesFromArrival>firstShockMinutes-hour*60].index.tolist()
        
        ermapIdx = ermaps[0]
        ermapmin = vdf.loc[ermapIdx].MinutesFromArrival
        #dropIdx += drops
        if ermapIdx > startIdx and ermapIdx < firstShockIdx:  # set ShockFlag == 1 for early prediction
            ermapIdxs = [x+ermapIdx for x in range(firstShockIdx-ermapIdx)]
            edf.set_value(ermapIdxs, 'ShockFlag', 1)
            ermapcnt += 1
            #print(str(vid)+"-FirstShockMin: "+str(firstShockMinutes)+", ermapIdx: "+str(ermapmin))

    #edf.drop(dropIdx, inplace= True) # Never drop for HMM
    edfposvid = len(edf[edf.ShockFlag ==1].VisitIdentifier.unique().tolist())
    #print("posvid: "+str(len(posvid))+"->"+str(edfposvid))
    return edf, ermapcnt 

def order_cols(traindf, testdf)
    col = ['VisitIdentifier', 'MinutesFromArrival', 'HeartRate', 'RespiratoryRate',
           'PulseOx', 'SystolicBP', 'DiastolicBP', 'MAP', 'Temperature', 'Bands',
           'BUN', 'Lactate', 'Platelet', 'Creatinine', 'BiliRubin', 'WBC',
           'Procalcitonin', 'CReactiveProtein', 'SedRate', 'PcrFluCulture',
           'AntiInfectiveFlag', 'GlasgowCommaScore', 'VasopressorFlag',
           'ChOxSrcVentTrachPAP', 'OxSrcVentTrachPAP', 'FIO2', 'SBP_b', 'MAP_b',
           'Procalcitonin_b', 'Bands_b', 'SedRate_b', 'HeartRate_b', 'Lactate_b',
           'Platelet_b', 'GCS_b', 'Creatinine_b', 'RR_b', 'FIO2_b', 'PulseOx_b',
           'BiliRubin_b', 'Temp_b', 'WBC_b', 'BUN_b', 'ShockFlag']
    traindf = traindf[col]
    testdf = testdf[col]

def hist_TP_hourDist()
        # histogram 
        plt.hist(TP_hourDist, 100, facecolor='blue')# alpha=0.75)
        plt.xlabel('hour distance between prediction and shock labels')
        plt.ylabel('Number of Instances')
        plt.grid(True)
        #plt.savefig('hist_missing.png', dpi=1000)
        plt.show()


        ## Analysis
def analysis():
    erlogdf = loaddf('', 'ep_log_04_24_10.csv')
    erdfcols= ['features','hbs0','hbs1','hbs2','hbs3','hbs4','hbs6','hbs8','hbs12','hbs16','hbs20','hbs23']
    erdf = pd.DataFrame(columns = erdfcols) 

    # 1. measurement
    ers =  ['er0','er1','er2','er3','er4','er6','er8','er10','er12','er16','er20','er23']
    f1_ers = ['f1_'+str(er) for er in ers]

    for er in ers:    
        col_name = 'f1_'+str(er)
        erlogdf[col_name] = erlogdf.f1 * erlogdf.er

    # 2. get features and with the max f1_er at each hbs (hours before shock) 
    bestFeat = []
    for fe in f1_er:
        row = erlogdf[erlogdf[fe].idxmax()]
        erdf[len(erdf)] = row[erdfcols]   


if __name__ == "__main__":

    unlabel_n = 8
    label_rate = 0.4
    hmm_mode = 'Multinomial'
    
    filepath = '//Volumes/lin-res19.csc.ncsu.edu/ykim32/sepsis/data/hmm/'

    # load datafiles, get train_posvid, train_negvid
    if True: 
        traindf = loaddf(filepath, "sp_train_04_24_13.csv")
        testdf = loaddf(filepath, "sp_test_04_24_13.csv")
        _ = statLabels(traindf, "ShockFlag", "[train]")
        _ = statLabels(testdf, "ShockFlag", "[test]")
        trainposvid = traindf[traindf.ShockFlag ==1].VisitIdentifier.unique().tolist()
        trainnegvid = [x for x in traindf.VisitIdentifier.unique().tolist() if x not in trainposvid]
        hmmdf = pd.DataFrame(columns = ['features','ehour','labelnum','unlabelnum','init_prob','trans_prob','emission_prob',
                           'tr_confmat','tr_acc', 'tr_prec', 'tr_recall', 'tr_f1', 
                            'te_confmat','te_acc', 'te_prec', 'te_recall', 'te_f1',
                            'TP_hourDist', 'er0','er1','er2','er3','er4','er5','er6',
                               'er7','er8','er9','er10','er11','er12','er13','er14','er15','er16',
                               'er17','er18','er19','er20','er21','er22','er23', 'ermapcnt']) 
      
    fixfeat = ['VisitIdentifier', 'ShockFlag']
    feat = ['VasopressorFlag', 'SBP_b', 'MAP_b', 'Temp_b']
    # hmm test for diagnosis
    if False: 
    test_pred, test_labs, test_seq_lengths, test_info = hmmtest(traindf, testdf, 0, fixfeat, feat, hmm_mode, hmmdf)
    
    # Semi-supervised learning: increasing the number pf unlabeled data
    if False:
        # divide the training data to N parts => trainparts[0]: labelled, trainparts[1-N]: unlablle
        trainparts = divideDataRate(label_rate, unlabel_n, trainposvid, trainnegvid)
        semidf = hmmtest_semi(trainparts, traindf, testdf, fixfeat, feat, "Multinomial") # "Gaussian"

    # Supervised learing test: increasing the number of labeled data
    if False:
        trainparts = divideDataRate(0.2, 10, trainposvid, trainnegvid)
        superdf = hmmtest_super(trainparts, traindf, testdf, fixfeat, feat, hmm_mode)
        savedf(superdf, '', 'superdf')
    
    # Single HMM learing test: with the fixed number of labeled data
    if False:
        test_pred, test_labs, test_seq_lengths, test_info = hmmtest(traindf, testdf, fixfeat, feat, hmm_mode, hmmdf)
    
        # Early prediction Testing codes by hours
        TP_minutesDist, TP_hourDist = getMinutesDist(testdf)
        earlyRate = getEarlyPredRate(TP_hourDist)
        
        # Early prediction Testing codes by time steps
        dist = spe.predict_dist(1, test_labs, test_pred, test_seq_lengths)
        #TP_stepDist = dist[dist < 900]

        # get the labels by vid
        testpred_list = labels_byvid(test_pred, test_seq_lengths)
        testlabel_list = labels_byvid(test_labs, test_seq_lengths) 
            
    # single_HMM
    if False:
        single_feat = [ ['VasopressorFlag'],['SBP_b'],['MAP_b'],['OxSrcVentTrachPAP'],['Lactate_b'],
           ['Bands_b'], ['Temp_b'], ['HeartRate_b'], ['RR_b'], ['PulseOx_b'], 
           ['BiliRubin_b'], ['FIO2_b'],['WBC_b'], ['BUN_b'],['Creatinine_b'],
           ['SedRate_b'], ['GCS_b'], ['GlasgowCommaScore'], ['Platelet_b'], ['Procalcitonin_b']]

        for f in feat:
            print(f)

            for i in [0, 2,4,6,8,10,12,16,20,24]:
                print(str(i)+"-before hours")
                # make dataset for early prediction
                edf, ermapcnt = getEarlyHourData(traindf[fixfeat+['MinutesFromArrival']+f], i)
                print("ermap count: "+str(ermapcnt))
                test_pred, test_labs, test_seq_lengths, test_info = hmmtest(edf, testdf,i, fixfeat, f, hmm_mode, hmmdf, "ShockFlag")
                hmmdf.set_value(len(hmmdf)-1, 'ermapcnt', ermapcnt)

        savedf(hmmdf, '', 'ep_log')
        
    # multi-HMM
    if False:
        multi_feat = [
              ['VasopressorFlag', 'SBP_b', 'MAP_b', 'BiliRubin_b']
        ]

        for f in feat:
            print(f)
            for i in [0, 2,4,6,8,10,12,16,20,24]:
                print(str(i)+"-before hours")
                edf, ermapcnt = getEarlyHourData(traindf[fixfeat+['MinutesFromArrival']+f], i)
                print("ermap count: "+str(ermapcnt))
                test_pred, test_labs, test_seq_lengths, test_info = hmmtest(edf, testdf,i, fixfeat, f, hmm_mode, hmmdf, "ShockFlag")
                print("te_f1: "+str(test_info[3]))
                hmmdf.set_value(len(hmmdf)-1, 'ermapcnt', ermapcnt)

            savedf(hmmdf, '', 'ep_log')
        
#hmmdf[['features','ehour','te_f1', 'er0','er1','er2','er3','er4','er6','er8','er10','er12','er16','er20','er23']]

