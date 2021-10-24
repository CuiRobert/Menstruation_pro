
import numpy as np
import pandas as pd



def pred_max_prob(df_result, keyword):
    df_result = df_result
    print("image nums:",len(df_result))

    case2img = {}  # 存储的是case_id :[图片编号1,图片编号2，图片编号3]
    for i, row in df_result.iterrows():
        case_id = row['orgname'].split('/')[-2]
        if case_id in case2img:
            case2img[case_id].append(row['newname'])
        else:
            case2img[case_id] = [row['newname']]

    case2prob = {}  # 存储case,每张图片经softmax函数的值 case_id:[[img1_prob,img2_prob],[img1_prob1,img1_prob2]]
    for key, value in case2img.items():
        list_prob_temp = [[], []]
        for i in value:
            list_prob_temp[0].append(df_result[df_result['newname'] == i]['predprob_0'].values[0])
            list_prob_temp[1].append(df_result[df_result['newname'] == i]['predprob_1'].values[0])
        case2prob[key] = list_prob_temp

    case2result = {}  # 存储 case_id:预测结果（0 or 1）
    for i, value in case2prob.items():
        a = np.where(value == np.max(value))
        case2result[i] = a[0].item()

    correct_JJ = 0
    correst_QT = 0
    count_label_1 = 0
    for _, value in case2result.items():
        if keyword in _:
            count_label_1 += 1
        if keyword in _ and value == 1:
            correct_JJ += 1
        if keyword not in _ and value == 0:
            correst_QT += 1
    print('cases nums:',len(case2result))
    print('总的准确率acc：%.4f\t (%d + %d)\t/%d '%((correct_JJ + correst_QT) / len(case2result), correct_JJ, correst_QT, len(case2result)))
    print('标签为1的准确率(绝经期或者增生期) sen：%.4f\t(%d/%d)'%(correct_JJ / count_label_1,correct_JJ,count_label_1))
    print('标签为0的准确率：%.4f\t(%d/%d)spe:'%((correst_QT / (len(case2result) - count_label_1)),correst_QT,(len(case2result) - count_label_1)))



def main():
    #  ../result/训练集概率值.xlsx
    df_dict3= pd.read_excel('../result/测试集概率值.xlsx')
    pred_max_prob(df_dict3,keyword='增')

if __name__ == '__main__':

    main()