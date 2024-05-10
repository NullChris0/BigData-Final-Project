from autogluon.tabular import TabularPredictor
from typing import Optional
import io
import pandas as pd

class DSaver(object):
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.train: Optional[pd.DataFrame] = None
        self.test: Optional[pd.DataFrame] = None
        self.predictor: Optional[TabularPredictor] = None
    def load_csv(self, file):
        self.data = pd.read_csv(file)
    def load_xls(self, file):
        self.data = pd.read_excel(file)
    def load_feather(self, file):
        self.data = pd.read_feather(file)
        
    @property
    def info(self):
        # https://blog.csdn.net/Constantdropping/article/details/110184710
        # 存储为字符串，创建一个StringIO，便于在内存中写入字符串
        buf = io.StringIO()
        
        # 数据属性写入
        self.data.info(buf=buf) 
        # 读取写到的数据，并转换成dataframe
        re = buf.getvalue() 
        # df = pd.DataFrame(re.split("\n"), columns=['info'])
        
        # # 根据保存字符串的格式，使用df.loc[]定位所要获取内容的位置
        # df_info = df.loc[3:len(df)-4, 'info'].str.split(n=1, expand=True).reset_index(drop=True)
        
        # # 创建一个新的属性list用于保存获取到的内容，我这里保存打印的最后一列内容
        # att = []
        # for i in df_info[1]:
        #     att.append(i.split()[-1])
        return re
    @property
    def find_string(self):
        if self.data is not None:
            output = self.data.select_dtypes(exclude=['number']).columns.tolist()
            return ', '.join(output)
        else:
            return ''
        
    def show_head(self, n):
        if self.data is not None:
            return self.data.head(n)
        else:
            return pd.DataFrame()


dATA =  DSaver()