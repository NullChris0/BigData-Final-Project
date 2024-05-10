from utils import *
from saver import dATA

dATA.load_csv('./data/kaggle_house_pred_train.csv')
merge_new('./data/kaggle_house_pred_test.csv')
drop_missings(0.3)
clear2full()
numeralization('MSZoning, Street, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, SaleType, SaleCondition', False, True)
plot_corr()
normalize(['MiscVal',
'MoSold',
'YrSold',
'PoolArea'])
split_data('2010-1-1', 'Date')
load_new('SalePrice', './AutogluonModels/ag-20240507_161942')