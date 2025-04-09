import pandas as pd
import numpy as np
import joblib
import os

class FeatureEngineer:
    def __init__(self, mode='train'):
        self.mode = mode
        self.params_path = 'model_params'
        self._initialize_params()
        
    def _initialize_params(self):
        os.makedirs(self.params_path, exist_ok=True)
        
        self.major_mapping = {
            '计算机类': ['计算机科学与技术', '软件工程', '信息安全', '人工智能'],
            '电子信息类': ['电子工程', '通信工程', '自动化'],
            '商科类': ['工商管理', '市场营销', '会计学', '金融学'],
            '工程类': ['机械工程', '土木工程', '材料工程'],
            '文科类': ['汉语言文学', '英语', '新闻学', '法学']
        }
        
        self.company_scale = {
            '创业公司':1, '中小型企业':2, '国企':3, '外企':4,
            '四大会计师事务所':5, '行业龙头':5, '世界500强':6, 
            '上市科技公司':6, '金融机构':4, '研究院所':3,
            '事业单位':3, '自由职业':1, '传媒机构':2, 
            '教育机构':2, '独角兽企业':5, '央企':3
        }
        
        self.gpa_map = {'2.0-3.0':2.5, '3.0-3.5':3.25, '3.5+':3.75}
        self.scholarship_map = {'无':0, '校奖':1, '省奖':2, '国奖':3}
        
    def transform(self, df):
        df = self._basic_features(df)
        df = self._process_english(df)
        df = self._process_internship(df)
        df = self._process_academic(df)
        df = self._winsorize(df)
        df = self._finalize_columns(df)
        
        if self.mode == 'train':
            self._save_params(df)
            
        return df
    
    def _basic_features(self, df):
        df = df.copy()
        reverse_map = {v:k for k,lst in self.major_mapping.items() for v in lst}
        df['专业大类'] = df['主修专业'].map(reverse_map)
        df['四六级分数'] = pd.to_numeric(df['四六级成绩'].str.extract(r'(\d+)')[0], errors='coerce')
        return df
    
    def _process_english(self, df):
        if self.mode == 'train':
            self.english_bins = np.quantile(df['四六级分数'].dropna(), [0.25, 0.5, 0.75])
            np.save(os.path.join(self.params_path, 'english_bins.npy'), self.english_bins)
        else:
            self.english_bins = np.load(os.path.join(self.params_path, 'english_bins.npy'))
            
        bins = [-np.inf] + self.english_bins.tolist() + [np.inf]
        labels = ['较差', '一般', '良好', '优秀']
        df['英语等级'] = pd.cut(df['四六级分数'], bins=bins, labels=labels, include_lowest=True)
        return df
    
    def _process_internship(self, df):
        df['企业规模'] = df['实习企业规模'].map(self.company_scale).fillna(2)
        df['实习质量'] = (df['实习岗位相关性'].map({'无关':0, '部分相关':1, '完全相关':2}) 
                       * np.log1p(df['企业规模']))
        return df
    
    def _process_academic(self, df):
        df['GPA'] = df['GPA分段'].map(self.gpa_map)
        df['奖学金等级'] = df['奖学金级别'].map(self.scholarship_map)
        df['学术潜力'] = np.sqrt(df['GPA']) * (df['奖学金等级'] + 1)
        return df
    
    def _winsorize(self, df):
        cols = ['四六级分数', '挂科门数']
        
        if self.mode == 'train':
            self.winsorize_limits = {}
            for col in cols:
                q1 = df[col].quantile(0.05)
                q3 = df[col].quantile(0.95)
                self.winsorize_limits[col] = (q1, q3)
            joblib.dump(self.winsorize_limits, 
                       os.path.join(self.params_path, 'winsorize_limits.pkl'))
        else:
            self.winsorize_limits = joblib.load(
                os.path.join(self.params_path, 'winsorize_limits.pkl'))
            
        for col in cols:
            q1, q3 = self.winsorize_limits[col]
            df[col] = df[col].clip(lower=q1, upper=q3)
        return df
    
    def _finalize_columns(self, df):
        drop_cols = ['四六级成绩', '主修专业', 'GPA分段', 
                    '奖学金级别', '实习企业规模', '实习岗位相关性']
        df = df.drop(drop_cols, axis=1)
        
        if self.mode == 'train':
            pd.Series(df.columns).to_csv(
                os.path.join(self.params_path, 'column_order.csv'), index=False)
            
        return df
    
    def _save_params(self, df):
        pass  # 参数已在各步骤中保存