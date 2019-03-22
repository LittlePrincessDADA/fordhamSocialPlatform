import numpy as np
import math

def compute_measure(predicted_label, true_label):
	t_idx = (predicted_label == true_label) # truely predicted
	f_idx = np.logical_not(t_idx)

	p_idx = (true_label > 0)
	n_idx = np.logical_not(p_idx)

	tp = np.sum( np.logical_and(t_idx, p_idx)) # TP
	tn = np.sum( np.logical_and(t_idx, n_idx)) # TN

	fp = np.sum(n_idx) - tn
	fn = np.sum(p_idx) - tp

	tp_fp_tn_fn_list=[]
	tp_fp_tn_fn_list.append(tp)
	tp_fp_tn_fn_list.append(fp)
	tp_fp_tn_fn_list.append(tn)
	tp_fp_tn_fn_list.append(fn)
	tp_fp_tn_fn_list=np.array(tp_fp_tn_fn_list)

	tp=tp_fp_tn_fn_list[0]
	fp=tp_fp_tn_fn_list[1]
	tn=tp_fp_tn_fn_list[2]
	fn=tp_fp_tn_fn_list[3]

	with np.errstate(divide='ignore'):
		sen = (1.0*tp)/(tp+fn)
	with np.errstate(divide='ignore'):
		spc = (1.0*tn)/(tn+fp)
	with np.errstate(divide='ignore'):
		if tp+fp == 0:
			ppr = 9999999999
		else:
			ppr = (1.0*tp)/(tp+fp)
	with np.errstate(divide='ignore'):
		if (tn+fn) == 0:
			npr = 9999999999
		else:
			npr = (1.0*tn)/(tn+fn)


	acc = (tp+tn)*1.0/(tp+fp+tn+fn)
	ans=[]
	ans.append(acc)
	ans.append(sen)
	ans.append(spc)
	ans.append(ppr)
	ans.append(npr)

	return ans

