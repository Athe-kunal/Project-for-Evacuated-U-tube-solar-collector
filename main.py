from all_algos import All_in_one
import argparse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

class quick_testing:
    def __init__(self,model,test_size):
        self.model = model
        self.test_size = test_size
        self.tester()
    def tester(self):
        cases = ['c1','c2']
        if self.model == 'plsr2':
            outputs = ['all']
        else:
            outputs = ['temperature','heat','efficiency']
        for case in cases:
            for output in outputs:
                model_name = All_in_one(case,self.model,output,self.test_size)
                predictions,y_val,r2_score,model,error = model_name.train_model()
                plt.plot()
                print(f"The train data r2 score of {output} is { r2_score} for case {case}")
                # print(f"The cross validation score of {output} is {r2_score} and error {error} for case {case} " )
                r2_score_test,y_test,test_preds = model_name.test_model(model)
                print(f"The test data r2 score of {output} is { r2_score_test} for case {case}")
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Testing the Machine learning Algorithms for ETSC')
  parser.add_argument('-m','--model_name',type=str,help="Please specify the name of the model")
  parser.add_argument('-t','--test_size',type=int,help='Test size')
  args = parser.parse_args()
  
  model_name = args.model_name
  test_size = args.test_size

  quick_testing(model_name,test_size)

