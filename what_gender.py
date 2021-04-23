# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:41:57 2019

@author: Ni He
"""

import pandas as pd

class whatgender(object):
    def __init__(self):
        self.stats = {'m': 0, 'f': 0, 'total': 0}
        self.name_freq = {}  
        # 字典name_freq键名是一个名字，键值是男名和女名的个数， i.e. {'坚':{'m':89, 'f':20}} 表示名字中有'坚'的人当中，男人89个，女人20个
        self.sur = pd.read_csv('surn.csv', encoding = 'gb2312')
        # 读入中文的姓氏

    def first_name(self, name):
        if len(name) <= 2:
            return name[1:]  # 名字太短不可能是复姓
        if self.sur['surname'].isin([str(name[:2])]).any(): # 判断是复姓
            return name[2:]  # 返回名字，除姓氏外的其他字，复姓情况： i.e. 欧阳娜娜 返回 娜娜
        else:
            return name[1:]  # 返回名字，除姓氏外的其他字： i.e. 张华兵 返回 华兵
    
    def add_name_freq(self, name, sex):        
        for char in self.first_name(name):
            # 如果这个名字已经被记录，那么修正（增加）该名字的的统计量，即是男人还是女人的比例
            if char in self.name_freq.keys():
                if sex == '男':
                    self.name_freq[str(char)]['m'] += 1
                else:
                    self.name_freq[str(char)]['f'] += 1
            else: # 如果该名字没有被记录，则增加键值
                if sex == '男':
                    self.name_freq[str(char)]={'m':1, 'f':0}
                else:
                    self.name_freq[str(char)]={'m':0, 'f':1}
        
    def load_data(self, ndata): 
        for index, row in ndata.iterrows():
            # 读取每一个名字作为训练数据，生成 name_freq的字典
            self.add_name_freq(row[0],row[1])
            # 读取之后，完成stats的统计，总共的男名数量，女名数量，以及总数
            if row[1] == '男':
                self.stats['m'] += 1
            else:
                self.stats['f'] += 1
        self.stats['total'] = self.stats['m'] + self.stats['f']
                
        
    def estimate_gender(self, name, gender = 'm'):   # Input name and predicted gender, return a precentage     
        '''
        贝叶斯公式: P(Y|X) = P(X|Y) * P(Y) / P(X)
        当X条件独立时, P(X|Y) = P(X1|Y) * P(X2|Y) * ...
        P（Y） 男或者女的概率， 因为名字非男即女，因此就默认成男人的概率。
        X1，X2是名字中的每一个字。
        /P(X) 该项是归一项，暂时不做考虑，不影响对男女名的判断，但是影响输出是否为一个0-1之间的概率
        P(X1|Y)P(X2|Y)P(Y) is needed to be compared with different Y (genders)
        '''
        # prob 为先验概率
        prob = self.stats['m']/self.stats['total'] if gender == 'm' \
        else self.stats['f']/self.stats['total'] # 根据所有样本来确定先验概率
        prob = 0.5  # 简单设定先验概率
        
        for char in self.first_name(name): # 遍历名字中的每一个字，认为每一个字是独立的，此处忽略了名字是一个有意义的词语的情况
            if str(char) in self.name_freq: # 正常情况，该名字出现过，有过统计
                # self.name_freq.get(str(char))[str(gender)] / self.stats.get(str(gender))
                # self.name_freq.get(str(char))[str(gender)] 返回一个数字，即名字中含有char的男人的数量
                # 将名字中有该字的男人的数量除以所有男人的数量，得到一个概率 P(Xi|Y)，即男人中名字里面有这个字的概率
                prob *= self.name_freq.get(str(char))[str(gender)] / self.stats.get(str(gender))
            else:
                # 碰见一个在训练数据中找不到的字，这个时候显示一下，该字实在生僻
                print('Oops! %s (%s) has not been covered by our training data.'% (char, name))
                # 概率乘以1，表示没有贡献，忽略该字的影响
                prob *= 1
        return prob
    
    def check_name(self, name, display = 1):
        # 由于未考虑归一项，即 prob_f + prob_m 可能不等于 1，因此需要都做一遍
        prob_f = self.estimate_gender(name, 'f')
        prob_m = self.estimate_gender(name, 'm')

        if display == 1:
            if prob_m > prob_f:
                print('%s should be a male with confidence level of %s'.format(name, round(1. * prob_m / (prob_m + prob_f),2)))
            else:
                print('%s should be a female with confidence level of %s'.format(name, round(1. * prob_f / (prob_m + prob_f),2)))
        else:
            # 通过比男名概率和女名概率的大小来判断是男人还是女人名字
            if prob_m > prob_f:
                return 'm'
            elif prob_m <= prob_f:
                return 'f'
    
    def data_split(self, data, m, f):
        # 将数据分为两个部分，训练和测试。 m是给男人留的测试数量，f是给女人名留的测试数量
        test = pd.concat([data[data['sex']=='男'][:m],data[data['sex']=='女'][:f]],ignore_index=True)
        train = pd.concat([data[data['sex']=='男'][m:],data[data['sex']=='女'][f:]],ignore_index=True)
        return train, test
    
    def testing(self,test, m, f):
        # 使用训练集来测试，并记录结果 mm 表示 男人的名字被成功的判断成男人， mf 表示 女人的名字被错误判断成男人
        self.test_res = {'mm':0, 'ff':0, 'mf':0, 'fm':0, 'm': m, 'f': f}
        for index, row in test.iterrows():
            res = self.check_name(row[0], display = 0)
            if res == 'm' and row[1] == '男':
                self.test_res['mm'] += 1
            elif res == 'm' and row[1] == '女':
                self.test_res['mf'] += 1
            elif res == 'f' and row[1] == '女':
                self.test_res['ff'] += 1
            elif res == 'f' and row[1] == '男':
                self.test_res['fm'] += 1
        print('正确率: Male4Male: %s, Female4Female: %s, 错误率： Male4Female: %s, Female4Male: %s.' %\
              (self.test_res['mm']/self.test_res['m'], self.test_res['ff']/self.test_res['f'],self.test_res['mf']/self.test_res['m'],self.test_res['fm']/self.test_res['f']))

import time
def dur(op=None, clock=[time.time()]):
  if op != None:
    duration = time.time() - clock[0]
    print('%s finished. Duration %.2f seconds.' % (op, duration))
  clock[0] = time.time()
  
if __name__ == '__main__':
    checkgender = whatgender()
    data = pd.read_csv('name.csv')
    m = 10000
    f = 10000
    train, test = checkgender.data_split(data, m, f)
    #------- Trainning 
    dur()
    checkgender.load_data(train)
    dur('Gender learning')
    #--------Testing by testing data
    dur()
    checkgender.testing(test, m, f)
    dur('Gender Testing')
    # ------ Use your own data
    checkgender.check_name(input('请在冒号后输入一个名字：'))
    
    
