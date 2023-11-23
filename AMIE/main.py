import duckdb
import pandas as pd
import pandas
import numpy as np
import os
import shutil
from numpy import unravel_index
import multiprocessing
import time
import itertools
import argparse
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
def worker_function(range_x):
  con = duckdb.connect(database=':memory:')
  benchmark_name = 'Biomarker'
  files_list = ['train.txt', 'test.txt', 'amie-' + benchmark_name + '.txt']
  for f in files_list:
    if f.find('amie-') == -1:
      data = pd.read_csv(r'./Data/' + benchmark_name + '/' + f, sep='\t', header=None, names=['s', 'p', 'o'])
      df = pd.DataFrame(data)
      df = df.where((pd.notnull(df)), None)

      con.execute("DROP TABLE IF EXISTS " + f.replace('.txt', '') + "_triples ")
      con.execute("CREATE TABLE " + f.replace('.txt', '') + "_triples AS SELECT * FROM df")
    else:
      data = pd.read_csv(r'./Data/' + benchmark_name + '/' + f, sep="\t|=>", header=None,
                         names=['Body', 'Head', 'Head_Coverage', 'Std_Confidence', 'PCA_Confidence', 'Positive_Examples',
                                'Body_size', 'PCA_Body_size', 'Functional_variable', 'n', 'nn', 'nnn']
                         , engine="python", encoding="ISO-8859-1")
      df = pd.DataFrame(data)
      df = df.where((pd.notnull(df)), None)

      con.execute("DROP TABLE IF EXISTS amie_rules")
      con.execute("CREATE TABLE amie_rules AS SELECT * FROM df")
  test_triples=con.execute("select distinct * from test_triples ").fetch_df().to_numpy()
  relations=con.execute("select distinct p from train_triples ").fetch_df().to_numpy()[:,0]
  dic_rel2range={}
  dic_rel2range_rules={}
  for r in relations:
    dic_rel2range[r] = con.execute("select distinct o from train_triples where p='" + str(r) + "'").fetch_df().to_numpy()[:, 0]
    rules_relation=con.execute("select distinct * from amie_rules where Head like '%"+r+"  ?%' order by PCA_Confidence desc").fetch_df().to_numpy()
    if rules_relation.shape[0]==0:
      rules_relation_head=con.execute("select distinct Head from amie_rules where Head like '%"+r+"%' order by PCA_Confidence desc").fetch_df().to_numpy()[:,0]
      dic_rel2range_rules[r]=[]
      for h in rules_relation_head:
        dic_rel2range_rules[r].append(h.split('  ')[2])
    else:
      dic_rel2range_rules[r]=dic_rel2range[r]
  mr = 0
  mrr = 0
  hit1 = 0
  hit3 = 0
  hit10 = 0
  for x in range(range_x[0],range_x[1],1):
    t=test_triples[x]
    #print(x,t)
    #print(x)
    o_list=dic_rel2range[t[1]]
    o_list_rules=dic_rel2range_rules[t[1]]
    pca_confidence=np.zeros((len(o_list)))
    rules_t=con.execute("select distinct * from amie_rules where Head like '%"+str(t[1])+"%' order by PCA_Confidence desc").fetch_df().to_numpy()
    prediction_rank_list=[None]*len(o_list)
    prediction_rank_list_index=0
    ranked_t_object=len(o_list)
    isObjectFound=False
    rank=len(o_list)
    if not(t[2] in o_list_rules):
      rank=rank+1
      mr=mr+rank
      mrr=mrr+float(1/rank)
      with open('./prediction_result.txt', 'a') as file:
        # 2. Write data to the file
        file.write(str([t, rank, 'It does not exist in range of relations']))
        file.write('\n')
      continue
    for rule in rules_t:

      rule_variable=rule[8]
      rule_head=rule[1]
      rule_body=rule[0]
      rule_body_splited=rule_body.split("  ")
      rule_head_splited=rule_head.split("  ")
      ##for Right Link prediction
      if not(rule_variable==rule_head_splited[0][1:]):
        continue
      rule_body=rule_body.replace(rule_variable,t[0])
      new_o_list=list()
      if not(rule_head_splited[2][0]=='?'): #if object rule is not variable
        new_o_list=[rule_head_splited[2]]
      else:
        new_o_list=list(o_list)
        if t[2] in new_o_list:
          aaaaa=1
          #new_o_list.remove(t[2])
          #new_o_list=[t[2]]+new_o_list
        else:
          rank=rank+1
          break
      for ranked_object in prediction_rank_list:
        if ranked_object in new_o_list:
          new_o_list.remove(ranked_object)
      for available_object in new_o_list :
        new_rule_body=rule_body.replace(rule_head_splited[2],available_object)
        new_rule_body_splited=new_rule_body.split("  ")
        is_body_covered=True
        dic_var2entity={}
        for j in range(0,len(new_rule_body_splited)-1,3):
          if not(j+2<len(new_rule_body_splited)):
            break
          s1=new_rule_body_splited[j].replace("'","''")
          p1=new_rule_body_splited[j+1].replace("'","''")
          o1=new_rule_body_splited[j+2].replace("'","''")
          if s1[0]=='?':
            available_s_list=con.execute("select distinct s from train_triples where p = '"+str(p1)+"' and o = '"+str(o1)+"'").fetch_df().to_numpy()[:,0]
            if not(s1 in dic_var2entity.keys()):
              dic_var2entity[s1]=available_s_list
            else:
              dic_var2entity[s1]=intersection(dic_var2entity[s1],available_s_list)
            if len(dic_var2entity[s1])==0:
              is_body_covered=False
              break
            continue

          if o1[0]=='?':
            available_o_list=con.execute("select distinct o from train_triples where p = '"+str(p1)+"' and s = '"+str(s1)+"'").fetch_df().to_numpy()[:,0]
            if not(o1 in dic_var2entity.keys()):
              dic_var2entity[o1]=available_o_list
            else:
              dic_var2entity[o1]=intersection(dic_var2entity[o1],available_o_list)
            if len(dic_var2entity[o1])==0:
              is_body_covered=False
              break
            continue
          available_triple_list=con.execute("select distinct * from train_triples where p = '"+str(p1)+"' and s = '"+str(s1)+"' and o = '"+str(o1)+"' ").fetch_df().to_numpy()[:,0]
          if len(available_triple_list)==0:
            is_body_covered=False
            break
        if is_body_covered:
          if not(available_object in prediction_rank_list):
            prediction_rank_list[prediction_rank_list_index]=available_object
            prediction_rank_list_index=prediction_rank_list_index+1
            if available_object==t[2]:
              isObjectFound=True
              rank=prediction_rank_list_index
              break
      if isObjectFound:
        break
    mr=mr+rank
    mrr=mrr+float(1/rank)
    if rank==1:
      hit1=hit1+1
    if rank<=3:
      hit3=hit3+1
    if rank<=10:
      hit10=hit10+1
    #print(t,rank,prediction_rank_list)
    with open('./prediction_result.txt', 'a') as file:
      #2. Write data to the file
      file.write(str([t,rank,prediction_rank_list[0:rank]]))
      file.write('\n')
  result=[range_x,mr,mrr,hit1,hit3,hit10]
  #print(result)
  with open('./range_result.txt', 'a') as file:
    # 2. Write data to the file
    file.write(str(result))
    file.write('\n')
  return result
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Description of your program')
  parser.add_argument('--benchmark', type=str, help='Your Benchmark')
  args = parser.parse_args()
  benchmark_name = args.benchmark if args.benchmark else input("Enter Benchmark: ")

  con = duckdb.connect(database=':memory:')
  files_list = ['test.txt']
  data = pd.read_csv(r'./Data/' + benchmark_name + '/test.txt', sep='\t', header=None, names=['s', 'p', 'o'])
  df = pd.DataFrame(data)
  df = df.where((pd.notnull(df)), None)
  con.execute("DROP TABLE IF EXISTS test_triples ")
  con.execute("CREATE TABLE test_triples AS SELECT * FROM df")
  test_triples = con.execute("select distinct * from test_triples ").fetch_df().to_numpy()
  #print(len(test_triples))
  pool = multiprocessing.Pool(processes=6)
  # Input data
  len_test_data = len(test_triples)
  done_range_start = []

  data_multiprocess = []
  for i in range(0, len_test_data + 100, 100):
    if not (i in done_range_start):
      if i + 100 >= len_test_data:
        data_multiprocess.append(list([i, len_test_data]))
        break
      data_multiprocess.append(list([i, i + 100]))
  #print(len(done_range_start))
  #print(len(data_multiprocess))
  #print(data_multiprocess)
  results = pool.map(worker_function, data_multiprocess)
  #print(results)
  # Close the pool to free resources
  pool.close()
  pool.join()

  hit1=0
  hit3=0
  hit10=0
  mr=0
  mrr=0
  for r in results:
    hit1 += r[3]
    hit3 += r[4]
    hit10 += r[5]
    mr += r[1]
    mrr += r[2]
  print("mr,mrr,hit1,hit3,hit10:")
  print(mr/len_test_data,mrr/len_test_data,hit1/len_test_data,hit3/len_test_data,hit10/len_test_data)





