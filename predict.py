import pandas as pd
import numpy as np
import joblib
from utils.features import FeatureEngineer

class SalaryPredictor:
    def __init__(self):
        self.model = joblib.load('model_params/salary_predictor.pkl')
        self.fe = FeatureEngineer(mode='predict')
        self.column_order = pd.read_csv('model_params/column_order.csv').squeeze()
        
    def predict(self, input_data):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        processed = self.fe.transform(df)
        processed = processed.reindex(columns=self.column_order, fill_value=0)
        log_pred = self.model.predict(processed)
        return np.expm1(log_pred)

if __name__ == "__main__":
    import sys
    
    predictor = SalaryPredictor()
    
    # 示例数据
    sample_data = {
        '主修专业': '计算机科学与技术',
        '岗位类型': '技术岗',
        '四六级成绩': 'CET6_600',
        'GPA分段': '3.5+',
        '奖学金级别': '国奖',
        '实习企业规模': '世界500强',
        '实习岗位相关性': '完全相关',
        '挂科门数': 0
    }
    
    # 命令行模式
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        predictions = predictor.predict(df)
        print("预测结果：")
        print(pd.DataFrame({
            '预测起薪': predictions.round(2),
            '输入特征': df.to_dict('records')
        }))
    else:
        # 单条预测
        result = predictor.predict(sample_data)[0]
        print(f"预测起薪：¥{result:.2f} 元/月")