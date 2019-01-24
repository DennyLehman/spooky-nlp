import pandas as pd
import os
import matplotlib.pyplot as plt
class Spooky_NLP():
	"""docstring for ClassName"""
	def __init__(self, train_data_path=os.path.join(os.getcwd(),'data','train.csv')):
		self.train_data_path = train_data_path
		self.train = self.load_data()
	
	def load_data(self):
		return pd.read_csv(self.train_data_path)

	def analyze_raw_data(self):

		authors = self.train.groupby('author').count().reset_index()
		print(self.train.groupby(['author']).count()['id'])

		#authors = pd.DataFrame(self.df.groupby(['author']).count())
		
		print(authors.columns)
		authors.plot.bar(x='author',y='id',rot=0, color=['blue','red','black']) #.plot.bar(x='author')		

		all_words_counts = self.get_all_words(self.train)
		all_words_counts.plot.bar()

		# analyze most popular words

	def get_all_words(self,df):
		all_words = df.text.str.split(expand=True).unstack().value_counts()
		#all_words = df.loc[0:50,'text'].str.split(expand=True).unstack()
		return all_words

	def execute(self):

		#print(self.df.head())
		self.analyze_raw_data()

if __name__ == '__main__':
	print('hello world')
	path = os.getcwd()
	my_nlp = Spooky_NLP(os.path.join(path,'data','train.csv'))
	my_nlp.execute()
    