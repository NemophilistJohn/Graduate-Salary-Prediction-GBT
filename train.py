# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import os
import seaborn as sns
from scipy import stats
from utils.features import FeatureEngineer

# 设置全局中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use('ggplot')  # 使用更好的绘图风格

# 自定义颜色方案
COLOR_PALETTE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

def build_model():
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['专业大类', '岗位类型', '英语等级']),
        ('quantile', QuantileTransformer(output_distribution='normal'), ['GPA', '四六级分数']),
        ('scale', StandardScaler(), ['实习质量', '学术潜力', '挂科门数'])
    ])
    
    return Pipeline([
        ('preprocess', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            random_state=42,
            early_stopping=True,
            scoring='neg_mean_absolute_error'
        ))
    ])

def enhanced_evaluation(model, X_test, y_test):
    """综合模型评估"""
    predictions = model.predict(X_test)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(predictions)
    
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_test, predictions),
        'MedAE': median_absolute_error(y_true, y_pred),
        '预测偏差率(%)': np.mean((y_pred - y_true)/y_true)*100,
        '预测标准差': np.std(y_pred - y_true)
    }
    
    error_percentiles = np.percentile(y_pred - y_true, [5, 25, 50, 75, 95])
    metrics.update({
        f'误差P{perc}': val for perc, val in zip([5,25,50,75,95], error_percentiles)
    })
    
    return pd.DataFrame([metrics], index=['指标值'])

def create_advanced_visualizations(model, X_test, y_test, feature_names):
    """生成全套高级可视化分析"""
    os.makedirs('results/visualizations', exist_ok=True)
    predictions = model.predict(X_test)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(predictions)
    residuals = y_pred - y_true
    
    # 1. 预测值-真实值回归图
    plt.figure(figsize=(10, 8), dpi=120)
    sns.jointplot(x=y_true, y=y_pred, kind='reg', 
                 line_kws={'color': COLOR_PALETTE[1]},
                 scatter_kws={'alpha':0.3, 'color': COLOR_PALETTE[0]})
    plt.gcf().suptitle("预测值与真实值回归分析", y=1.02)
    plt.xlabel("真实起薪（元）")
    plt.ylabel("预测起薪（元）")
    plt.savefig('results/visualizations/regression_analysis.png', bbox_inches='tight')
    
    # 2. 残差分析组合图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 残差分布
    sns.histplot(residuals, kde=True, ax=axes[0,0], color=COLOR_PALETTE[0])
    axes[0,0].set_title("残差分布直方图")
    
    # Q-Q图
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title("残差Q-Q图")
    
    # 残差与预测值关系
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[1,0], alpha=0.5, color=COLOR_PALETTE[1])
    axes[1,0].axhline(0, color=COLOR_PALETTE[3], linestyle='--')
    axes[1,0].set_title("残差 vs 预测值")
    
    # 误差百分比分析
    error_percent = (residuals / y_true) * 100
    sns.boxplot(x=error_percent, ax=axes[1,1], color=COLOR_PALETTE[2])
    axes[1,1].set_title("相对误差百分比分布")
    axes[1,1].set_xlabel("误差百分比(%)")
    
    plt.suptitle("残差综合分析", y=1.02)
    plt.savefig('results/visualizations/residual_analysis.png', bbox_inches='tight')
    
    # 3. 特征重要性桑基图
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x=result.importances_mean[sorted_idx][-15:], 
               y=feature_names[sorted_idx][-15:], 
               palette=COLOR_PALETTE)
    plt.title("Top 15 重要特征（排列重要性）", fontsize=14)
    plt.xlabel("重要性得分")
    plt.savefig('results/visualizations/feature_importance.png', bbox_inches='tight')
    
    # 4. 学习曲线分析
    plt.figure(figsize=(12, 8))
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_test, y_test, cv=5,
        scoring='neg_mean_absolute_error',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    
    plt.plot(train_sizes, -train_scores.mean(axis=1), 'o-', 
            color=COLOR_PALETTE[0], label='训练误差')
    plt.plot(train_sizes, -test_scores.mean(axis=1), 'o-', 
            color=COLOR_PALETTE[1], label='验证误差')
    plt.fill_between(train_sizes, 
                    -train_scores.mean(axis=1) - train_scores.std(axis=1),
                    -train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.1, color=COLOR_PALETTE[0])
    plt.fill_between(train_sizes,
                    -test_scores.mean(axis=1) - test_scores.std(axis=1),
                    -test_scores.mean(axis=1) + test_scores.std(axis=1),
                    alpha=0.1, color=COLOR_PALETTE[1])
    plt.xlabel("训练样本量", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title("模型学习曲线", fontsize=14)
    plt.legend()
    plt.savefig('results/visualizations/learning_curve.png', bbox_inches='tight')

def main():
    # 初始化环境
    os.makedirs('model_params', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)
    
    # 数据加载与处理
    df = pd.read_csv('data/student_data.csv')
    X = df.drop('起薪', axis=1)
    y = np.log1p(df['起薪'])
    
    # 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征工程
    fe = FeatureEngineer(mode='train')
    X_train = fe.transform(X_train) 
    X_test = fe.transform(X_test)
    
    
    # 确保列顺序一致
    column_order = pd.read_csv('model_params/column_order.csv').squeeze()
    X_train = X_train[column_order]
    X_test = X_test[column_order]
    
    # 模型训练与调优
    model = build_model()
    param_grid = {
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7],
        'regressor__l2_regularization': [0, 0.1, 0.5],
        'regressor__max_iter': [1000, 2000]
    }
    
    searcher = GridSearchCV(model, param_grid, 
                           scoring='neg_mean_absolute_error',
                           cv=5, n_jobs=-1)
    searcher.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = searcher.best_estimator_
    
    # 综合评估
    print("\n=== 模型评估结果 ===")
    eval_df = enhanced_evaluation(best_model, X_test, y_test)
    print(eval_df.T.to_string(float_format="%.2f"))
    
    # 高级可视化
    feature_names = best_model.named_steps['preprocess'].get_feature_names_out()
    create_advanced_visualizations(best_model, X_test, y_test, feature_names)
    
    # 保存结果
    eval_df.to_csv('results/metrics_report.csv', index=False)
    joblib.dump(best_model, 'model_params/salary_predictor.pkl')
    print("\n=== 结果保存 ===")
    print(f"模型文件: model_params/salary_predictor.pkl")
    print(f"评估报告: results/metrics_report.csv")
    print(f"可视化图表: results/visualizations/ 目录")

if __name__ == "__main__":
    main()