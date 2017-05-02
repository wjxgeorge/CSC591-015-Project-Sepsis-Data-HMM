
# coding: utf-8

# In[9]:

import collections
import numpy as np
import pandas
import math
import sklearn

import sepsis_perform_eval as spe

from sklearn.metrics import confusion_matrix
from hmmlearn import hmm
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from random import shuffle

def feat2observ (features):
    #transform observations with features in form of matrix into a sequence of observations in form of tuple
    features = np.asarray(features)
    feat_tps = list(map(tuple, features))
    return feat_tps

def observ_transpose(observations):
    # NOT USED
    #transpose a sequence of observations in form of tuple
    #called after invoke feat2observ
    
    ob_buff = np.empty(len(observations), dtype = object)
    ob_buff[:] = observations
    ob_buff.shape = (len(observations), 1)
    
    return ob_buff

def hmm_multi_param(features, labels):
    #parameter setup for Multinomial HMM
    #incoperate 1-layer Dirichlet Process to allow new emission symbols
    features_tmp = np.ascontiguousarray(features).view(np.dtype((np.void, features.dtype.itemsize * features.shape[1])))
    [__, idx, f_count] = np.unique(features_tmp, return_index = True, return_counts = True)
    uniq_feat = features[idx]
    uniq_feat = feat2observ(uniq_feat)
    #print(f_count)
    #print(uniq_feat)
    uniq_observs = dict(zip(uniq_feat, range(len(uniq_feat))))
    
    observs = feat2observ(features)
    
    [uniq_lab, count] = np.unique(labels, return_counts = True)
    count = count.astype('d')
    emit_prob = np.zeros((len(uniq_lab), len(uniq_feat)+1))
    
    for i in range(len(observs)):
        ob = uniq_observs[observs[i]]
        lb = np.where(uniq_lab == labels[i])[0][0]
        
        emit_prob[lb, ob] += 1
    
    divider = np.sum(emit_prob, axis = 1)
    divider.shape = (len(divider), 1)
    emit_prob = np.divide(emit_prob, divider)
    
    #print(emit_prob)
    #print(np.sum(emit_prob, axis = 1))
    #print(np.count_nonzero(np.sum(emit_prob, axis = 0)))
    
    return [uniq_feat, uniq_observs, observs, emit_prob]

def hmm_norm_param(features, labels):
    #parameter setup for Gaussian HMM
    uniq_lab = np.unique(labels)
    feat_num = features.shape[1]
    
    mean_mat = np.zeros((len(uniq_lab), feat_num))
    full_cov_mat = np.zeros((len(uniq_lab), feat_num, feat_num))
    
    for i in range(len(uniq_lab)):
        target = features[labels == uniq_lab[i], :]
        #print(np.sum(target, axis = 0))
        mean_mat[i] = np.mean(target, axis = 0)
        full_cov_mat[i] = np.cov(target.T)
    
    #print(mean_mat)
    #print(full_cov_mat[1])
    #print(np.linalg.eigvalsh(full_cov_mat[1]))
    #print(np.allclose(full_cov_mat[0], full_cov_mat[0].T))
    return [mean_mat, full_cov_mat]

def hmm_init(ids, features, labels, hmm_type):
    #Supported HMM types:
    #    1. Gaussian Model
    #    2. Multinomial
    #generate parameters for HMM:
    #    1. Initial Probabilities
    #    2. Transition Probabilities
    #    3. Mean parameters for each state (GM)
    #    4. Covariance parameters for each state (GM)
    #    5. Emission Probabilities (Multinomial)
    
    #initialize start probabilities
    start_prob = np.array([])
    [uniq_lab, count] = np.unique(labels, return_counts = True)
    count = count.astype('d')
    labs_length = len(labels)
    for i in range(len(uniq_lab)):
        prob = count[i]/labs_length
        start_prob = np.append(start_prob, prob)
    
    #initialize transition probabilities
    trans_prob = np.zeros((len(uniq_lab), len(uniq_lab)))
    for i in range(len(labels)-1):
        if ids[i] != ids[i+1]:
            continue
        
        start = np.where(uniq_lab == labels[i])[0][0]
        end = np.where(uniq_lab == labels[i+1])[0][0]
        trans_prob[start, end] += 1
    
    divider = np.sum(trans_prob, axis = 1)
    divider.shape = (len(divider), 1)
    trans_prob = np.divide(trans_prob, divider)
    
    if (hmm_type == "Multinomial"):
        optional_param = hmm_multi_param(features, labels)
    elif (hmm_type == "Gaussian"):
        optional_param = hmm_norm_param(features, labels)
    
    return [start_prob, trans_prob, optional_param]

def hmm_data_preprocess(data, label_tag, visit_id_tag):
    #separate data labels and features
    #calculate sequences' length for multi-sequence input for HMM
    
    labels = np.array(data.loc[:,label_tag])
    event_ids = np.array(data.loc[:, visit_id_tag])
    feature = data.drop([visit_id_tag, label_tag], axis = 1)
    feature = feature.as_matrix()
    
    # TO DO: handle situation if the event ids are not sorted
    [__, count] = np.unique(event_ids, return_counts = True)
    seq_lengths = count
    
    return [event_ids, feature, labels, seq_lengths]

def hmm_multi_proprocess(model, labels, beta = 1):
    #Implement Dirichlet Process after model training
    [uniq_lab, count] = np.unique(labels, return_counts = True)
    count = count.astype('d')
    
    tar = np.argmax(count)
    for i in range(model.emissionprob_.shape[1]-1):
        new_prob = model.emissionprob_[tar, i]
        new_prob = new_prob*count[tar]/float(count[tar]+beta)
        model.emissionprob_[tar, i] = new_prob
        
    model.emissionprob_[tar, model.emissionprob_.shape[1]-1] = beta/float(count[tar]+beta)
    return model

def hmm_setup(train_data, label_tag, visit_id_tag, hmm_type):
    #integration of HMM initialization and parameter setting
    #return trained model
    [train_ids, train_feats, train_labs, seq_lengths] = hmm_data_preprocess(train_data, label_tag, visit_id_tag)
    
    if (hmm_type == "Multinomial"):
        [start_prob, trans_prob, [uniq_observs, observ_dict, train_observ, emit_prob]] = hmm_init(train_ids, train_feats, train_labs, hmm_type)
    elif (hmm_type == "Gaussian"):
        [start_prob, trans_prob, [mean_mat, full_cov_mat]] = hmm_init(train_ids, train_feats, train_labs, hmm_type)
    
    n_states = len(np.unique(train_labs))
    
    if (hmm_type == "Multinomial"):
        model = hmm.MultinomialHMM(n_components = n_states, init_params = "")
    elif (hmm_type == "Gaussian"):
        model = hmm.GaussianHMM(n_components = n_states, covariance_type = "full", init_params = "")
    
    model.transmat_ = trans_prob
    model.startprob_ = start_prob
    
    if (hmm_type == "Multinomial"):
        model.emissionprob_ = emit_prob
        
        print(model.transmat_)
        #print(model.emissionprob_)
        
        observs = np.array(list(map(lambda x: observ_dict[x], train_observ)))
        observs = observs.reshape(-1, 1)
        
        model.fit(observs, seq_lengths)
        
        model = hmm_multi_proprocess(model, train_labs, beta = 1)
        
        return [model, observs, seq_lengths, train_labs, observ_dict, uniq_observs]
    
    elif (hmm_type == "Gaussian"):
        model.means_ = mean_mat
        model.covars_ = full_cov_mat
        
        observs = train_feats
        
        model.fit(observs, seq_lengths)
        
        return [model, observs, seq_lengths, train_labs]

def hmm_pred_setup(data, label_tag, visit_id_tag, hmm_type, observ_dict = None):
    [ids, feats, labs, seq_lengths] = hmm_data_preprocess(data, label_tag, visit_id_tag)
    
    if hmm_type == "Gaussian":
        return [feats, seq_lengths, labs]
    elif hmm_type == "Multinomial":
        if observ_dict is None:
            raise ValueError('observation dict should not be None')
        
        ob_feat = feat2observ(feats)
        observs = np.zeros(len(ob_feat), dtype = int)
        dict_size = len(observ_dict)
        for i in range(len(observs)):
            if ob_feat[i] in observ_dict:
                observs[i] = observ_dict[ob_feat[i]]
            else:
                observs[i] = dict_size
        
        observs = observs.reshape(-1, 1)
        
        return [observs, seq_lengths, labs]

def check_sequence(seq_lengths, labs, pred):
    start = 0
    for i in range(len(seq_lengths)):
        print("Visit: "+str(i))
        print(labs[start:start+seq_lengths[i]])
        print(pred[start:start+seq_lengths[i]])
        print("**************************************")
        start += seq_lengths[i]
    
def main_function():
    train_data = pandas.read_csv("data/spde_train.csv")
    test_data = pandas.read_csv("data/spde_test.csv")
    #print(train_data.head())
    
    train_data = train_data.drop("MinutesFromArrival", axis = 1)
    test_data = test_data.drop("MinutesFromArrival", axis = 1)
    
    # Revised Start
    event_ids = np.array(train_data.loc[:, "VisitIdentifier"])
    [__, count] = np.unique(event_ids, return_counts = True)
    allseq_lengths = count
    wholevid = sorted(set(event_ids))
    
    tmpacc = []
    for i in range(10):
    
        parts = 5
        #labIdxset = np.array_split(range(0, len(wholevid)), parts)
        Idxlist = list(range(0, len(wholevid)))
        shuffle(Idxlist)
        labIdxset = np.array_split(Idxlist, parts)
        _data = train_data[train_data['VisitIdentifier'].isin([wholevid[j] for j in labIdxset[0].tolist()])]
        
        acc = []
        ## Revise End
        for i in range(1, parts + 1):

            ## train_data = _data ## Revised Here
            print(_data.shape)

            [model, observs, seq_lengths, train_labs, observ_dict, uniq_observs] = hmm_setup(_data, "ShockFlag", "VisitIdentifier", "Multinomial")
            #[model, observs, seq_lengths, train_labs] = hmm_setup(train_data, "ShockFlag", "VisitIdentifier", "Gaussian")
            print(model.transmat_)
            #print(model.emissionprob_)
            #print(np.sum(model.emissionprob_, axis = 1))
            #print(np.count_nonzero(np.sum(model.emissionprob_, axis = 0)))
            pred = model.predict(observs, seq_lengths)
            print(confusion_matrix(train_labs, pred))
            [pre, act] = spe.calPosVisit(seq_lengths, pred, train_labs)
            print(confusion_matrix(act, pre))
            #check_sequence(seq_lengths, train_labs, pred)
            tmp = spe.predict_dist(1, train_labs, pred, seq_lengths)
            tmp = tmp[tmp >= 0]
            #print(tmp)
            print(np.mean(tmp))

            [test_ob, seq_lengths, test_labs] = hmm_pred_setup(test_data, "ShockFlag", "VisitIdentifier", "Multinomial", observ_dict)
            pred = model.predict(test_ob, seq_lengths)
            print(model.score(test_ob, seq_lengths))
            print(confusion_matrix(test_labs, pred))
            #check_sequence(seq_lengths, test_labs, pred)
            [pre, act] = spe.calPosVisit(seq_lengths, pred, test_labs)
            tempacc = sum(np.array(pre) == np.array(act))/len(act) ## Edit Here
            acc.append(tempacc) ## Edit Here
            print(confusion_matrix(act, pre))
            tmp = spe.predict_dist(1, test_labs, pred, seq_lengths)
            print(tmp)
            print(len(tmp))
            print(np.mean(tmp))

            ## Revised Start
            if i < parts:

                tmp_df = train_data[train_data['VisitIdentifier'].isin([wholevid[j] for j in labIdxset[i].tolist()])]
                _data = _data.append(tmp_df)

            ## Revised End

        #print(acc)
        #plt.plot(acc)
        tmpacc.append(acc)
            
    avacc = np.average(np.array(tmpacc), axis=0)
    print(avacc)
    plt.plot(avacc)
        

if __name__ == "__main__":
    main_function()
