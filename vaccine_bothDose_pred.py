import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

class VaccinePredictor:

    def __init__(self,):

        self.data0 = pd.read_csv('2021-05-25_trends.csv')
        # Pre-processing the self.data1
        self.data1 = self.data0[self.data0['Location']=='US']
        self.data1 = self.data1[self.data1.notna()]
        self.data1 = self.data1[['Date', 'Admin_Dose_1_Day_Rolling_Average']]
        self.data1 = self.data1.dropna()
        self.data1 = self.data1.set_index('Date')
        self.data1.index = pd.to_datetime(self.data1.index)
        # Fit a Gausssian Curve
        self.gaussian_model1, cov = optimize.curve_fit(self.gaussian_f,
            xdata=np.arange(len(self.data1['Admin_Dose_1_Day_Rolling_Average'])), 
            ydata=self.data1['Admin_Dose_1_Day_Rolling_Average'].values)

        print('Gaussian fit successfull on 1st dose with parameters:')
        print(self.gaussian_model1)

        # Pre-processing the self.data2
        self.data2 = self.data0[self.data0['Location']=='US']
        self.data2 = self.data2[self.data2.notna()]
        self.data2 = self.data2[['Date', 'Admin_Dose_2_Day_Rolling_Average']]
        self.data2 = self.data2.dropna()
        self.data2 = self.data2.set_index('Date')
        self.data2.index = pd.to_datetime(self.data2.index)
        # Fit a Gausssian Curve
        self.gaussian_model2, cov = optimize.curve_fit(self.gaussian_f,
            xdata=np.arange(len(self.data2['Admin_Dose_2_Day_Rolling_Average'])), 
            ydata=self.data2['Admin_Dose_2_Day_Rolling_Average'].values)

        print('Gaussian fit successfull on 2nd dose with parameters:')
        print(self.gaussian_model2)

    def gaussian_f(self,X, a, b, c):
        y = a * np.exp(-0.5 * ((X-b)/c)**2)
        return y

    # Plot parametric fitting.
    def utils_plot_parametric(self,data, zoom=30, figsize=(15,5)):
        ## interval
        data["residuals"] = data["ts"] - data["model"]
        data["conf_int_low"] = data["forecast"] - 1.96*data["residuals"].std()
        data["conf_int_up"] = data["forecast"] + 1.96*data["residuals"].std()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        
        ## entire series
        data["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting", color="black")
        data["model"].plot(ax=ax[0], color="green")
        data["forecast"].plot(ax=ax[0], grid=True, color="red")
        ax[0].fill_between(x=data.index, y1=data['conf_int_low'], y2=data['conf_int_up'], color='b', alpha=0.3)
    
        ## focus on last
        first_idx = data[pd.notnull(data["forecast"])].index[0]
        first_loc = data.index.tolist().index(first_idx)
        zoom_idx = data.index[first_loc-zoom]
        data.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black", 
                                    title="Zoom on the last "+str(zoom)+" observations")
        data.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")
        data.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")
        ax[1].fill_between(x=data.loc[zoom_idx:].index, y1=data.loc[zoom_idx:]['conf_int_low'], 
                        y2=data.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
        plt.show()
        return data[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]

    def forecast_curve(self,ts, f, model, pred_ahead=None, freq="D", zoom=30, figsize=(15,5)):
        '''
        Forecast unknown future.
        :parameter
            :param ts: pandas series
            :param f: function
            :param model: list of optim params
            :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
            :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
            :param zoom: for plotting
        '''
        ## fit
        X = np.arange(len(ts))
        fitted = f(X, model[0], model[1], model[2])
        data = ts.to_frame(name="ts")
        data["model"] = fitted
        
        ## index
        index = pd.date_range(start='2021-05-25',periods=pred_ahead,freq=freq)
        index = index[1:]
        ## forecast
        Xnew = np.arange(len(ts)+1, len(ts)+1+len(index))
        preds = f(Xnew, model[0], model[1], model[2])
        data = data.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
        
        ## plot
        self.utils_plot_parametric(data, zoom=zoom)
        return data

    def write_forecast_csv(self,pred_ahead=120):

        preds1 = self.forecast_curve(self.data1["Admin_Dose_1_Day_Rolling_Average"], self.gaussian_f, self.gaussian_model1, 
            pred_ahead=pred_ahead, freq="D", zoom=7)
        preds2 = self.forecast_curve(self.data2["Admin_Dose_2_Day_Rolling_Average"], self.gaussian_f, self.gaussian_model2, 
            pred_ahead=pred_ahead, freq="D", zoom=7)
        
        all_df1 = pd.concat([preds1['model'].dropna(),preds1['forecast'].dropna()])
        all_df1 = pd.DataFrame(all_df1)
        all_df1.columns = ['Dose1']
        all_df1.index.names = ['Date']
        # print(all_df1)
        all_df2 = pd.concat([preds2['model'].dropna(),preds2['forecast'].dropna()])
        all_df2 = pd.DataFrame(all_df2)
        all_df2.columns = ['Dose2']
        all_df2.index.names = ['Date']
        # print(all_df2)
        all_df = pd.concat([all_df1,all_df2],axis=1)
        # print(all_df)
        all_df.to_csv('vaccine_forecasts_both_dose.csv')

if __name__ == "__main__":

    PredObj = VaccinePredictor()
    PredObj.write_forecast_csv()