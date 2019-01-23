import pandas as pd
import os
#from matplotlib import pyplot as plt
class Spooky_NLP():
	"""docstring for ClassName"""
	def __init__(self, train_data_path=os.path.join(os.getcwd(),'data','train.csv')):
		self.train_data_path = train_data_path
		self.df = self.load_data()
	
	def load_data(self):
		return pd.read_csv(self.train_data_path)

	def analyze_raw_data(self):
		self.df.plot()		

	def execute(self):
		print(self.df.head())

if __name__ == '__main__':
	print('hello world')
	path = os.getcwd()
	my_nlp = Spooky_NLP(os.path.join(path,'data','train.csv'))
	my_nlp.execute()