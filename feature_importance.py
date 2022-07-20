from all_algos import All_in_one
from collections import Counter
import argparse
import warnings
warnings.filterwarnings("ignore")

class Feature_importance:
    def __init__(self,output:str):
        self.output = output
    
    def scores_dict(self):    
        models = ['lr','knn','svr','pca','sgdr','plsr1']
        cases = ['c1','c2']

        imp_scores = []
        for case in cases:
            for model in models:
                aio = All_in_one(case,model,self.output,3)
                imp_score_train = aio.permutation_importance(True)
                imp_score_test = aio.permutation_importance(False)
                model_dict = dict(Counter(imp_score_train) + Counter(imp_score_test))
                imp_scores.append(model_dict)
#                 print(model_dict)
        return imp_scores
    
    def average_score(self):
        imp_scores = self.scores_dict()
        sum_dict = imp_scores[0]
        for dict_ in imp_scores[1:]:
            temp_dict = dict(Counter(sum_dict) + Counter(dict_))
            sum_dict = temp_dict
        total = 12
        final_dict = {k: v / total for k, v in sum_dict.items()}
        return final_dict

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Feature Importance for ETSC')
  parser.add_argument('-o','--output',type=str,help='Calculate the feature importance of an output')
  args = parser.parse_args()
  output = args.output
  final_dict = Feature_importance(output)
  print(final_dict.average_score())